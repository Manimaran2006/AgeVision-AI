import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN

class ImageValidator:
    def __init__(self, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # keep_all=True allows us to see if there are multiple faces
        self.mtcnn = MTCNN(keep_all=True, device=self.device)
        
        # Thresholds
        self.min_confidence = 0.90
        self.min_face_size = 80  # pixels
        self.min_blur_variance = 100

    def check_blur(self, image_bgr):
        """Calculates the Laplacian variance to measure image sharpness."""
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        return variance

    def validate(self, image_bgr):
        """
        Validates the image for age prediction.
        Returns: (is_valid, error_message, best_face_box)
        """
        # Convert BGR to RGB for MTCNN
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        
        # 1. Detect Faces
        boxes, probs, landmarks = self.mtcnn.detect(image_rgb, landmarks=True)
        
        if boxes is None or len(boxes) == 0:
            return False, "No human face detected. Please upload a clear photo of a person.", None
            
        # 2. Check for Multiple Faces
        if len(boxes) > 1:
            # We could pick the largest face, but for a web app it's often safer to reject 
            # to avoid predicting the wrong person.
            # Let's find the most prominent face (highest confidence + largest area)
            best_idx = -1
            best_score = -1
            for i, (box, prob) in enumerate(zip(boxes, probs)):
                if prob is None: continue
                area = (box[2] - box[0]) * (box[3] - box[1])
                score = prob * area
                if score > best_score:
                    best_score = score
                    best_idx = i
            
            if best_idx == -1:
                return False, "Failed to identify a primary face.", None
                
            box = boxes[best_idx]
            prob = probs[best_idx]
            landmark = landmarks[best_idx]
        else:
            box = boxes[0]
            prob = probs[0]
            landmark = landmarks[0]

        # 3. Confidence Check (Is it really a human?)
        if prob < self.min_confidence:
            return False, f"Face detected with low confidence ({prob:.0%}). Are you sure this is a human face?", None

        # 4. Size Check
        width = box[2] - box[0]
        height = box[3] - box[1]
        if width < self.min_face_size or height < self.min_face_size:
            return False, f"Face is too small ({int(width)}x{int(height)}px). Please upload a closer portrait.", None

        # 5. Blur Check (Check the cropped face area)
        x1, y1 = max(0, int(box[0])), max(0, int(box[1]))
        x2, y2 = min(image_bgr.shape[1], int(box[2])), min(image_bgr.shape[0], int(box[3]))
        face_crop = image_bgr[y1:y2, x1:x2]
        
        if face_crop.size == 0:
            return False, "Invalid face crop dimensions.", None
            
        blur_score = self.check_blur(face_crop)
        if blur_score < self.min_blur_variance:
            return False, "Image is too blurry. Please upload a sharper photo.", None
            
        # 6. Eye Geometry Check (Extreme Angles)
        if landmark is not None and len(landmark) >= 2:
            left_eye = landmark[0]
            right_eye = landmark[1]
            eye_distance = np.linalg.norm(left_eye - right_eye)
            face_width = box[2] - box[0]
            
            # If the distance between eyes is less than ~30% of the face width, 
            # the person is likely looking sideways (profile).
            if eye_distance / face_width < 0.25:
                 return False, "Extreme profile angle detected. Please look directly at the camera.", None

        return True, "Valid", [x1, y1, x2, y2]
