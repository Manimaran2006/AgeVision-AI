"""
V3 Age Estimation Inference — ConvNeXt-Small + DLDL + TTA + Face Crop
"""

import os
import sys
import json
import math
import argparse
import numpy as np
import cv2
from PIL import Image
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import timm
from validator import ImageValidator
from transformers import pipeline

AGE_GROUPS = [
    ("Child", 0, 12), ("Teenager", 13, 19),
    ("Young Adult", 20, 35), ("Adult", 36, 59), ("Senior", 60, 200),
]

def get_age_group(age):
    for name, low, high in AGE_GROUPS:
        if low <= float(age) <= high:
            return name
    return "Unknown"

# ── Face Crop ──
HAAR_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
_cascade = None

def get_cascade():
    global _cascade
    if _cascade is None and os.path.exists(HAAR_PATH):
        _cascade = cv2.CascadeClassifier(HAAR_PATH)
    return _cascade

def crop_face(image_bgr, padding=0.20):
    cascade = get_cascade()
    h, w = image_bgr.shape[:2]
    if cascade is not None:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=4, minSize=(40, 40))
        if len(faces) > 0:
            faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)
            fx, fy, fw, fh = faces[0]
            px, py = int(fw * padding), int(fh * padding)
            x1, y1 = max(0, fx-px), max(0, fy-py)
            x2, y2 = min(w, fx+fw+px), min(h, fy+fh+py)
            cropped = image_bgr[y1:y2, x1:x2]
            if cropped.size > 0:
                return cropped
    mn = min(h, w)
    sy, sx = (h-mn)//2, (w-mn)//2
    return image_bgr[sy:sy+mn, sx:sx+mn]

# ── Model ──
class FaceAgeModel(nn.Module):
    def __init__(self, num_classes=101, backbone_name="convnext_small.fb_in22k"):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=False, num_classes=0)
        in_features = self.backbone.num_features
        self.age_head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )
        self.register_buffer("age_indices", torch.arange(num_classes, dtype=torch.float32))

    def forward(self, x):
        features = self.backbone(x)
        return self.age_head(features)

    def predict_age(self, x):
        logits = self.forward(x)
        probs = F.softmax(logits, dim=-1)
        expected = torch.sum(probs * self.age_indices, dim=-1)
        variance = torch.sum(probs * (self.age_indices - expected.unsqueeze(-1)) ** 2, dim=-1)
        std = torch.sqrt(torch.clamp(variance, min=1e-4))
        return expected, std, probs

# Keep backward compatibility alias
AgeEstimationModel = FaceAgeModel

# ── Predictor with TTA ──
class AgePredictor:
    def __init__(self, model_path="models/best_age_model.pt", device=None, metrics_path=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        # Auto-download from GitHub Releases if missing on cloud servers
        if not os.path.exists(model_path):
            os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
            release_url = "https://github.com/KaRTh1-s/AgeVision-AI/releases/download/v1.0.0/best_age_model.pt"
            print(f"[INFO] Model not found locally. Downloading from GitHub Release: {release_url}...")
            try:
                import urllib.request
                urllib.request.urlretrieve(release_url, model_path)
                print(f"[OK] Downloaded model weights to {model_path}")
            except Exception as e:
                print(f"[WARN] Could not auto-download model weights: {e}")

        # Detect backbone from checkpoint
        backbone_name = "convnext_small.fb_in22k"
        if os.path.exists(model_path):
            ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
            if "backbone" in ckpt:
                backbone_name = ckpt["backbone"]
            elif "config" in ckpt and "backbone_name" in ckpt["config"]:
                backbone_name = ckpt["config"]["backbone_name"]

        self.model = FaceAgeModel(num_classes=101, backbone_name=backbone_name).to(self.device)
        self.model.eval()
        self.validator = ImageValidator(device=self.device)
        
        # Load pre-trained gender classifier
        try:
            self.gender_pipe = pipeline("image-classification", model="rizvandwiki/gender-classification", device=0 if torch.cuda.is_available() else -1)
            print("[OK] Loaded pre-trained gender classification model")
        except Exception as e:
            print(f"[WARN] Failed to load gender model: {e}")
            self.gender_pipe = None

        if os.path.exists(model_path):
            state = ckpt.get("model_state_dict", ckpt)
            self.model.load_state_dict(state, strict=False)
            print(f"[OK] Loaded: {model_path} (backbone={backbone_name})")

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        self.tta_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=1.0),  # Always flip
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def predict(self, image_input, use_face_crop=True, use_tta=True):
        # Load image as BGR
        if isinstance(image_input, (str, Path)):
            image_bgr = cv2.imread(str(image_input))
            if image_bgr is None:
                pil = Image.open(str(image_input)).convert("RGB")
                image_bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        elif isinstance(image_input, Image.Image):
            image_bgr = cv2.cvtColor(np.array(image_input.convert("RGB")), cv2.COLOR_RGB2BGR)
        elif isinstance(image_input, np.ndarray):
            if len(image_input.shape) == 2:
                image_bgr = cv2.cvtColor(image_input, cv2.COLOR_GRAY2BGR)
            elif image_input.shape[2] == 4:
                image_bgr = cv2.cvtColor(image_input, cv2.COLOR_RGBA2BGR)
            elif image_input.shape[2] == 3:
                image_bgr = cv2.cvtColor(image_input, cv2.COLOR_RGB2BGR)
            else:
                image_bgr = image_input
        else:
            raise ValueError(f"Unsupported: {type(image_input)}")

        # ── Pre-inference Validation ──
        is_valid, msg, box = self.validator.validate(image_bgr)
        if not is_valid:
            return {"error": msg}

        # Use the original Haar crop that the model was trained on!
        # MTCNN crops too tightly, which ruins the accuracy since the model expects 20% padding.
        if use_face_crop:
            image_bgr = crop_face(image_bgr)
            
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # Original prediction
        t1 = self.transform(image_rgb).unsqueeze(0).to(self.device)
        with torch.no_grad():
            ctx = torch.amp.autocast("cuda", torch.float16) if self.device.type == "cuda" else torch.no_grad()
            with ctx:
                age1, std1, probs1 = self.model.predict_age(t1)

        if use_tta:
            # Flipped prediction
            t2 = self.tta_transform(image_rgb).unsqueeze(0).to(self.device)
            with torch.no_grad():
                with ctx:
                    age2, std2, probs2 = self.model.predict_age(t2)

            # Average probability distributions (better than averaging ages)
            avg_probs = (probs1 + probs2) / 2.0
            indices = self.model.age_indices
            predicted_age = float(torch.sum(avg_probs * indices, dim=-1).cpu().item())
            variance = float(torch.sum(avg_probs * (indices - predicted_age) ** 2, dim=-1).cpu().item())
            pred_std = math.sqrt(max(variance, 1e-4))
        else:
            predicted_age = float(age1.cpu().item())
            pred_std = float(std1.cpu().item())

        predicted_age = float(np.clip(predicted_age, 1.0, 100.0))
        age_group = get_age_group(predicted_age)
        uncertainty = max(pred_std, 1.0)
        lower = max(1, int(round(predicted_age - 1.96 * uncertainty)))
        upper = min(100, int(round(predicted_age + 1.96 * uncertainty)))

        rel = uncertainty / max(predicted_age, 10.0)
        conf_label = "High" if rel < 0.15 else ("Medium" if rel < 0.35 else "Low")
        conf_pct = float(np.clip(100.0 * math.exp(-rel), 20.0, 99.0))
        
        # Gender prediction
        predicted_gender = "Unknown"
        gender_conf = 0.0
        if self.gender_pipe:
            try:
                # The pipeline expects a PIL image
                pil_face = Image.fromarray(image_rgb)
                g_res = self.gender_pipe(pil_face)
                # Returns list of dicts like [{'label': 'male', 'score': 0.99}, ...]
                best_gender = g_res[0]
                predicted_gender = best_gender["label"].capitalize()
                gender_conf = float(best_gender["score"] * 100)
            except Exception as e:
                print(f"Gender prediction failed: {e}")

        return {
            "image_path": str(image_input) if isinstance(image_input, (str, Path)) else "input",
            "predicted_age": round(predicted_age, 1),
            "predicted_age_group": age_group,
            "likely_age_range": f"{lower}-{upper} years",
            "confidence_level": f"Model-derived: {conf_label} ({conf_pct:.1f}%)",
            "prediction_std": round(pred_std, 2),
            "predicted_gender": predicted_gender,
            "gender_confidence": f"{gender_conf:.1f}%",
        }

def main():
    parser = argparse.ArgumentParser(description="V3 Age Estimation Inference")
    parser.add_argument("--model", default="models/best_age_model.pt")
    parser.add_argument("--image", type=str)
    parser.add_argument("--image-dir", type=str)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", type=str)
    parser.add_argument("--no-tta", action="store_true")
    args = parser.parse_args()

    predictor = AgePredictor(model_path=args.model, device=args.device)
    use_tta = not args.no_tta

    if args.image:
        res = predictor.predict(args.image, use_tta=use_tta)
        print("\n" + "=" * 50)
        print("V3 AGE PREDICTION (TTA" + (" ON" if use_tta else " OFF") + ")")
        print("=" * 50)
        for k, v in res.items():
            print(f"  {k}: {v}")
        print("=" * 50)
        if args.output:
            with open(args.output, "w") as f:
                json.dump(res, f, indent=4)

    elif args.image_dir:
        paths = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG"):
            paths.extend(Path(args.image_dir).glob(ext))
        results = []
        for p in paths:
            try:
                r = predictor.predict(str(p), use_tta=use_tta)
                results.append(r)
                print(f"  {p.name}: {r['predicted_age']} yrs ({r['predicted_age_group']})")
            except Exception as e:
                print(f"  FAIL {p.name}: {e}")
        if args.output and results:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()
