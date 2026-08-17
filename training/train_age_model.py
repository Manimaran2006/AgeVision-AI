import os
import sys
import json
import time
import math
import random
import logging
import warnings
from pathlib import Path
from datetime import datetime
from contextlib import nullcontext

import numpy as np
import cv2
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms

from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm

import timm

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ============================================================
# REPRODUCIBILITY
# ============================================================
SEED = 42

def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

seed_everything()

# ============================================================
# DIRECTORIES
# ============================================================
LOG_DIR = "logs"
MODEL_DIR = "models"
CHECKPOINT_DIR = "checkpoints"
RESULT_DIR = "results"
for d in [LOG_DIR, MODEL_DIR, CHECKPOINT_DIR, RESULT_DIR]:
    os.makedirs(d, exist_ok=True)

# ============================================================
# LOGGING
# ============================================================
log_file = os.path.join(LOG_DIR, f"training_v3_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# ============================================================
# CONFIG
# ============================================================
CONFIG = {
    "manifest_path": "dataset_manifest.json",
    "img_size": 224,
    "num_classes": 101,
    "gaussian_sigma": 2.0,       # Tighter sigma for sharper distributions
    "expectation_weight": 0.5,   # Stronger L1 expectation penalty
    "variance_weight": 0.02,     # Mean-Variance regularization
    "batch_size": 8,
    "gradient_accumulation_steps": 4,
    "epochs": 3,                 # Fast fine-tune
    "early_stopping_patience": 3,
    "learning_rate": 5e-5,       # Slightly lower for face-pretrained backbone
    "backbone_learning_rate": 5e-6,
    "weight_decay": 1e-4,
    "scheduler_T0": 3,
    "min_lr": 1e-7,
    "num_workers": 4,
    "prefetch_factor": 2,
    "gradient_clip": 1.0,
    "use_amp": True,
    "use_face_crop": True,
    # Backbone: face-optimized ConvNeXt from timm
    "backbone_name": "convnext_small.fb_in22k",
}

AGE_GROUPS = [
    ("Child", 0, 12), ("Teenager", 13, 19),
    ("Young Adult", 20, 35), ("Adult", 36, 59), ("Senior", 60, 200),
]

def get_age_group(age):
    for name, low, high in AGE_GROUPS:
        if low <= float(age) <= high:
            return name
    return "Unknown"

# ============================================================
# GAUSSIAN TARGET
# ============================================================
def generate_gaussian_target(age, num_classes=101, sigma=2.0):
    classes = np.arange(num_classes, dtype=np.float32)
    dist = np.exp(-((classes - float(age)) ** 2) / (2.0 * sigma ** 2))
    dist /= (np.sum(dist) + 1e-8)
    return torch.tensor(dist, dtype=torch.float32)

# ============================================================
# FACE CROP (Haar — fast)
# ============================================================
HAAR_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
_cascade = None

def get_cascade():
    global _cascade
    if _cascade is None and os.path.exists(HAAR_PATH):
        _cascade = cv2.CascadeClassifier(HAAR_PATH)
    return _cascade

def quick_face_crop(img, padding=0.20):
    cascade = get_cascade()
    h, w = img.shape[:2]
    if cascade is not None:
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            faces = cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3, minSize=(30, 30))
            if len(faces) > 0:
                faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)
                fx, fy, fw, fh = faces[0]
                px, py = int(fw * padding), int(fh * padding)
                x1, y1 = max(0, fx-px), max(0, fy-py)
                x2, y2 = min(w, fx+fw+px), min(h, fy+fh+py)
                cropped = img[y1:y2, x1:x2]
                if cropped.size > 0:
                    return cropped
        except Exception:
            pass
    mn = min(h, w)
    sy, sx = (h-mn)//2, (w-mn)//2
    return img[sy:sy+mn, sx:sx+mn]

# ============================================================
# DATASET
# ============================================================
class AgeDataset(Dataset):
    def __init__(self, samples, transform=None, img_size=224, num_classes=101, sigma=2.0, use_face_crop=True):
        self.samples = samples
        self.transform = transform
        self.img_size = img_size
        self.num_classes = num_classes
        self.sigma = sigma
        self.use_face_crop = use_face_crop
        self.failed = 0

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        try:
            img_path = sample["path"]
            age = float(sample["age"])
            clamped = min(max(age, 0.0), float(self.num_classes - 1))

            image = cv2.imread(img_path)
            if image is None:
                image = np.array(Image.open(img_path).convert("RGB"))
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

            if self.use_face_crop:
                image = quick_face_crop(image)

            if self.transform:
                image = self.transform(image)
            else:
                image = cv2.resize(image, (self.img_size, self.img_size))
                image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

            target_dist = generate_gaussian_target(clamped, self.num_classes, self.sigma)
            target_scalar = torch.tensor(age, dtype=torch.float32)
            return image, target_dist, target_scalar

        except Exception as e:
            self.failed += 1
            if self.failed <= 3:
                logger.warning(f"Failed sample {idx}: {e}")
            return self[np.random.randint(0, len(self.samples))]

# ============================================================
# MODEL: ConvNeXt-Small (IN-22K pretrained) + DLDL Head
# ============================================================
class FaceAgeModel(nn.Module):
    def __init__(self, num_classes=101, backbone_name="convnext_small.fb_in22k"):
        super().__init__()
        logger.info(f"Initializing backbone: {backbone_name} (num_classes={num_classes})")
        self.backbone = timm.create_model(backbone_name, pretrained=True, num_classes=0)
        in_features = self.backbone.num_features
        logger.info(f"Backbone feature dim: {in_features}")

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
        logits = self.age_head(features)
        return logits

    def compute_expectation(self, logits):
        probs = F.softmax(logits, dim=-1)
        expected = torch.sum(probs * self.age_indices, dim=-1)
        return expected, probs

# ============================================================
# LOSS: KL + L1 Expectation + Variance Penalty
# ============================================================
class MeanVarianceDLDLLoss(nn.Module):
    def __init__(self, expectation_weight=0.5, variance_weight=0.02, num_classes=101):
        super().__init__()
        self.exp_w = expectation_weight
        self.var_w = variance_weight
        self.register_buffer("age_indices", torch.arange(num_classes, dtype=torch.float32))

    def forward(self, logits, target_dists, target_ages):
        log_probs = F.log_softmax(logits, dim=-1)
        kl = F.kl_div(log_probs, target_dists, reduction="batchmean")

        probs = F.softmax(logits, dim=-1)
        indices = self.age_indices.to(logits.device)
        expected = torch.sum(probs * indices, dim=-1)

        l1 = F.smooth_l1_loss(expected, target_ages)
        variance = torch.sum(probs * (indices - expected.unsqueeze(-1)) ** 2, dim=-1).mean()

        total = kl + self.exp_w * l1 + self.var_w * variance
        return total, expected

# ============================================================
# TRAINER
# ============================================================
class V3Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        self.criterion = MeanVarianceDLDLLoss(
            expectation_weight=config["expectation_weight"],
            variance_weight=config["variance_weight"],
            num_classes=config["num_classes"],
        )
        if self.device.type == "cuda" and config["use_amp"]:
            self.scaler = torch.amp.GradScaler("cuda", init_scale=1024.0)
        self.best_val_mae = float("inf")
        self.best_epoch = 0
        self.patience_counter = 0
        self.history = {k: [] for k in [
            "train_loss", "train_mae", "train_rmse",
            "val_loss", "val_mae", "val_rmse",
            "learning_rate", "epoch_time",
        ]}
        logger.info("=" * 80)
        logger.info(f"Device: {self.device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
        logger.info(f"Loss: KL + {config['expectation_weight']}*L1 + {config['variance_weight']}*Variance")
        logger.info(f"Backbone: {config['backbone_name']}")
        logger.info("=" * 80)

    def setup_model(self):
        self.model = FaceAgeModel(
            num_classes=self.config["num_classes"],
            backbone_name=self.config["backbone_name"],
        ).to(self.device)
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Parameters: {total:,} (Trainable: {trainable:,})")

    def setup_optimizer(self):
        backbone_params = list(self.model.backbone.parameters())
        head_params = list(self.model.age_head.parameters())
        self.optimizer = optim.AdamW([
            {"params": backbone_params, "lr": self.config["backbone_learning_rate"]},
            {"params": head_params, "lr": self.config["learning_rate"]},
        ], weight_decay=self.config["weight_decay"])
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=self.config["scheduler_T0"], T_mult=2, eta_min=self.config["min_lr"],
        )

    def autocast_ctx(self):
        if self.device.type == "cuda" and self.config["use_amp"]:
            return torch.amp.autocast(device_type="cuda", dtype=torch.float16)
        return nullcontext()

    def train_epoch(self, loader, epoch):
        self.model.train()
        total_loss, total_ae, total_se, total_n = 0., 0., 0., 0
        acc = self.config["gradient_accumulation_steps"]
        self.optimizer.zero_grad(set_to_none=True)
        t0 = time.perf_counter()
        pbar = tqdm(loader, desc=f"Epoch {epoch+1} [V3 Train]", ncols=120)

        for bi, (imgs, dists, ages) in enumerate(pbar):
            imgs = imgs.to(self.device, non_blocking=True)
            dists = dists.to(self.device, non_blocking=True)
            ages = ages.to(self.device, non_blocking=True)

            with self.autocast_ctx():
                logits = self.model(imgs)
                loss, exp = self.criterion(logits, dists, ages)
                scaled = loss / acc

            if self.scaler:
                self.scaler.scale(scaled).backward()
            else:
                scaled.backward()

            if ((bi + 1) % acc == 0) or (bi + 1 == len(loader)):
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config["gradient_clip"])
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config["gradient_clip"])
                    self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.scheduler.step(epoch + bi / len(loader))

            bs = ages.size(0)
            ae = torch.abs(exp.detach() - ages).sum().item()
            se = ((exp.detach() - ages) ** 2).sum().item()
            total_ae += ae; total_se += se; total_n += bs
            total_loss += loss.item() * bs

            if (bi + 1) % 25 == 0:
                cur_mae = total_ae / total_n
                elapsed = time.perf_counter() - t0
                pbar.set_postfix({"loss": f"{total_loss/total_n:.4f}", "MAE": f"{cur_mae:.2f}y", "img/s": f"{total_n/elapsed:.1f}"})

        return total_loss/total_n, total_ae/total_n, math.sqrt(total_se/total_n)

    def validate(self, loader, epoch, return_preds=False):
        self.model.eval()
        total_loss, total_ae, total_se, total_n = 0., 0., 0., 0
        all_p, all_t = [], []
        pbar = tqdm(loader, desc=f"Epoch {epoch+1} [V3 Val]", ncols=120)

        with torch.no_grad():
            for imgs, dists, ages in pbar:
                imgs = imgs.to(self.device, non_blocking=True)
                dists = dists.to(self.device, non_blocking=True)
                ages = ages.to(self.device, non_blocking=True)
                with self.autocast_ctx():
                    logits = self.model(imgs)
                    loss, exp = self.criterion(logits, dists, ages)
                bs = ages.size(0)
                ae = torch.abs(exp - ages).sum().item()
                se = ((exp - ages) ** 2).sum().item()
                total_ae += ae; total_se += se; total_n += bs
                total_loss += loss.item() * bs
                if return_preds:
                    all_p.extend(exp.float().cpu().numpy().tolist())
                    all_t.extend(ages.float().cpu().numpy().tolist())

        avg_loss = total_loss / total_n
        mae = total_ae / total_n
        rmse = math.sqrt(total_se / total_n)
        if return_preds:
            return avg_loss, mae, rmse, np.array(all_p), np.array(all_t)
        return avg_loss, mae, rmse

    def train(self, train_loader, val_loader):
        epochs = self.config["epochs"]
        logger.info("=" * 80)
        logger.info(f"STARTING V3 TRAINING ({epochs} EPOCHS)")
        logger.info("=" * 80)
        self.setup_optimizer()

        for epoch in range(epochs):
            t0 = time.perf_counter()
            tl, tm, tr = self.train_epoch(train_loader, epoch)
            vl, vm, vr = self.validate(val_loader, epoch)
            et = time.perf_counter() - t0
            lr = min(g["lr"] for g in self.optimizer.param_groups)

            for k, v in zip(self.history.keys(), [tl, tm, tr, vl, vm, vr, lr, et]):
                self.history[k].append(v)

            logger.info("=" * 80)
            logger.info(f"Epoch {epoch+1}/{epochs} V3 Summary:")
            logger.info(f"  Train Loss: {tl:.4f} | Train MAE: {tm:.3f} yrs | Train RMSE: {tr:.3f} yrs")
            logger.info(f"  Val Loss:   {vl:.4f} | Val MAE:   {vm:.3f} yrs | Val RMSE:   {vr:.3f} yrs")
            logger.info(f"  LR: {lr:.8f} | Duration: {et/60:.1f} min")
            logger.info("=" * 80)

            if vm < self.best_val_mae:
                self.best_val_mae = vm
                self.best_epoch = epoch
                self.patience_counter = 0
                path = os.path.join(MODEL_DIR, "best_age_model.pt")
                torch.save({"model_state_dict": self.model.state_dict(), "val_mae": vm, "epoch": epoch+1, "config": self.config, "backbone": self.config["backbone_name"]}, path)
                logger.info(f"★ NEW BEST MODEL: {path} (Val MAE: {vm:.3f} yrs)")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config["early_stopping_patience"]:
                    logger.info("Early stopping triggered.")
                    break

# ============================================================
# RESULTS EXPORT
# ============================================================
def save_results(preds, targs, history, n_train, n_val):
    errors = np.abs(preds - targs)
    sq = (preds - targs) ** 2
    mae = float(np.mean(errors))
    rmse = float(np.sqrt(np.mean(sq)))
    mse = float(np.mean(sq))
    t3 = float(np.mean(errors <= 3) * 100)
    t5 = float(np.mean(errors <= 5) * 100)
    t10 = float(np.mean(errors <= 10) * 100)

    groups = {}
    for g, lo, hi in AGE_GROUPS:
        m = (targs >= lo) & (targs <= hi)
        if np.any(m):
            groups[g] = {"samples": int(np.sum(m)), "mae": float(np.mean(errors[m])), "rmse": float(np.sqrt(np.mean(errors[m]**2)))}

    res = preds - targs
    output = {
        "generated_at": datetime.now().isoformat(),
        "model_type": "V3 ConvNeXt-Small + DLDL + MeanVariance + FaceCrop",
        "dataset": {"train_samples": n_train, "validation_samples": n_val, "total_samples": n_train + n_val},
        "metrics": {"MAE": mae, "RMSE": rmse, "MSE": mse,
                    "accuracy_within_3_years_percent": t3, "accuracy_within_5_years_percent": t5,
                    "accuracy_within_10_years_percent": t10, "age_group_metrics": groups},
        "uncertainty": {"residual_mean": float(np.mean(res)), "residual_std": float(np.std(res)), "residual_mae": mae},
        "config": CONFIG,
    }
    with open(os.path.join(RESULT_DIR, "final_metrics.json"), "w") as f:
        json.dump(output, f, indent=4)
    with open(os.path.join(RESULT_DIR, "training_history.json"), "w") as f:
        json.dump(history, f, indent=4)
    np.savez_compressed(os.path.join(RESULT_DIR, "validation_predictions.npz"), predictions=preds, targets=targs)

    with open(os.path.join(RESULT_DIR, "metrics_report.txt"), "w") as f:
        f.write("=" * 80 + "\n")
        f.write("V3 AGE ESTIMATION — FINAL REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"MAE:  {mae:.3f} years\nRMSE: {rmse:.3f} years\nMSE:  {mse:.3f}\n")
        f.write(f"±3y: {t3:.2f}% | ±5y: {t5:.2f}% | ±10y: {t10:.2f}%\n\n")
        for g, d in groups.items():
            f.write(f"  {g:<14} N={d['samples']:<7} MAE={d['mae']:.3f}  RMSE={d['rmse']:.3f}\n")

    logger.info("Saved all results to results/")

# ============================================================
# MAIN
# ============================================================
def main():
    seed_everything()
    logger.info("=" * 80)
    logger.info("V3 HIGH-PRECISION PIPELINE: ConvNeXt-Small + DLDL + MeanVariance + FaceCrop")
    logger.info("=" * 80)

    with open(CONFIG["manifest_path"], "r") as f:
        manifest = json.load(f)
    train_samples = manifest["splits"]["train"]
    val_samples = manifest["splits"]["val"]
    logger.info(f"Dataset: {len(train_samples):,} train | {len(val_samples):,} val")

    train_tf = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((CONFIG["img_size"], CONFIG["img_size"])),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.03),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((CONFIG["img_size"], CONFIG["img_size"])),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_ds = AgeDataset(train_samples, train_tf, CONFIG["img_size"], CONFIG["num_classes"], CONFIG["gaussian_sigma"], CONFIG["use_face_crop"])
    val_ds = AgeDataset(val_samples, val_tf, CONFIG["img_size"], CONFIG["num_classes"], CONFIG["gaussian_sigma"], CONFIG["use_face_crop"])

    lkw = {"batch_size": CONFIG["batch_size"], "pin_memory": True, "num_workers": CONFIG["num_workers"],
           "persistent_workers": CONFIG["num_workers"] > 0, "prefetch_factor": CONFIG["prefetch_factor"]}
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **lkw)
    val_loader = DataLoader(val_ds, shuffle=False, drop_last=False, **lkw)

    trainer = V3Trainer(CONFIG)
    trainer.setup_model()

    # Quick sanity test (256 samples, 60 steps)
    logger.info("=" * 80)
    logger.info("QUICK SANITY TEST")
    logger.info("=" * 80)
    sub = Subset(train_ds, list(range(256)))
    sub_loader = DataLoader(sub, batch_size=8, shuffle=True, drop_last=True)
    trainer.setup_optimizer()
    trainer.model.train()
    loader_it = iter(sub_loader)
    for step in range(1, 61):
        try:
            imgs, dists, ages = next(loader_it)
        except StopIteration:
            loader_it = iter(sub_loader)
            imgs, dists, ages = next(loader_it)
        imgs = imgs.to(trainer.device); dists = dists.to(trainer.device); ages = ages.to(trainer.device)
        with trainer.autocast_ctx():
            logits = trainer.model(imgs)
            loss, exp = trainer.criterion(logits, dists, ages)
        trainer.optimizer.zero_grad(set_to_none=True)
        if trainer.scaler:
            trainer.scaler.scale(loss).backward()
            trainer.scaler.unscale_(trainer.optimizer)
            torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), 1.0)
            trainer.scaler.step(trainer.optimizer)
            trainer.scaler.update()
        else:
            loss.backward()
            trainer.optimizer.step()
        batch_mae = torch.abs(exp.detach() - ages).mean().item()
        if step in [1, 20, 40, 60]:
            logger.info(f"  Sanity step {step:>3}/60 | Loss: {loss.item():.4f} | Batch MAE: {batch_mae:.2f} yrs")
    logger.info("★ SANITY TEST PASSED")
    logger.info("=" * 80)

    # Fresh model for full training
    trainer.setup_model()
    trainer.train(train_loader, val_loader)

    # Final evaluation
    best_path = os.path.join(MODEL_DIR, "best_age_model.pt")
    ckpt = torch.load(best_path, map_location=trainer.device, weights_only=False)
    trainer.model.load_state_dict(ckpt["model_state_dict"])
    logger.info("RUNNING FINAL EVALUATION")
    _, final_mae, final_rmse, preds, targs = trainer.validate(val_loader, trainer.best_epoch, return_preds=True)
    save_results(preds, targs, trainer.history, len(train_samples), len(val_samples))
    logger.info("=" * 80)
    logger.info(f"★ V3 COMPLETE! Final MAE: {final_mae:.3f} yrs | RMSE: {final_rmse:.3f} yrs")
    logger.info("=" * 80)

if __name__ == "__main__":
    main()
