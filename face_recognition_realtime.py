"""
Face Recognition + Anti-Spoofing - PRODUCTION READY VERSION
Fixed thresholds, better logic, liveness detection
"""

import cv2
import torch
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
from pathlib import Path
import json
import argparse
from collections import deque
import mediapipe as mp

from model import Face3DFusionModel
from config import config


class LivenessDetector:
    """Simple blink-based liveness detection"""
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.blink_history = deque(maxlen=30)  # 30 frames history
        self.is_blinking = False
        self.blink_count = 0
        self.last_blink_time = 0
    
    def detect_blink(self, frame):
        """Detect eye blink using Eye Aspect Ratio (EAR)"""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        
        if not results.multi_face_landmarks:
            return False, 0
        
        landmarks = results.multi_face_landmarks[0]
        
        # Left eye landmarks
        left_eye = [362, 385, 387, 263, 373, 380]
        # Right eye landmarks
        right_eye = [33, 160, 158, 133, 153, 144]
        
        def eye_aspect_ratio(eye_points):
            # Vertical distances
            v1 = np.linalg.norm(np.array([
                landmarks.landmark[eye_points[1]].x,
                landmarks.landmark[eye_points[1]].y
            ]) - np.array([
                landmarks.landmark[eye_points[5]].x,
                landmarks.landmark[eye_points[5]].y
            ]))
            
            v2 = np.linalg.norm(np.array([
                landmarks.landmark[eye_points[2]].x,
                landmarks.landmark[eye_points[2]].y
            ]) - np.array([
                landmarks.landmark[eye_points[4]].x,
                landmarks.landmark[eye_points[4]].y
            ]))
            
            # Horizontal distance
            h = np.linalg.norm(np.array([
                landmarks.landmark[eye_points[0]].x,
                landmarks.landmark[eye_points[0]].y
            ]) - np.array([
                landmarks.landmark[eye_points[3]].x,
                landmarks.landmark[eye_points[3]].y
            ]))
            
            ear = (v1 + v2) / (2.0 * h)
            return ear
        
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        avg_ear = (left_ear + right_ear) / 2.0
        
        # EAR threshold for blink
        is_blink = avg_ear < 0.2
        
        # Track blinks
        self.blink_history.append(is_blink)
        
        # Detect blink event (transition from open to close to open)
        if len(self.blink_history) >= 3:
            if not self.is_blinking and is_blink:
                self.is_blinking = True
            elif self.is_blinking and not is_blink:
                self.is_blinking = False
                self.blink_count += 1
        
        # Calculate liveness score (blink rate in last 30 frames)
        blink_rate = sum(self.blink_history) / len(self.blink_history)
        
        return is_blink, self.blink_count


# Setup
mp_face = mp.solutions.face_detection
detector = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.6)

rgb_tf = transforms.Compose([
    transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225])
])


def load_model(ckpt_path):
    print(f"Loading model from: {ckpt_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")

    try:
        with torch.serialization.safe_globals([np.core.multiarray.scalar]):
            ckpt = torch.load(ckpt_path, map_location=device)
    except:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    num_classes = ckpt.get("num_classes", config.NUM_CLASSES)
    
    model = Face3DFusionModel(num_classes=num_classes, config=config)
    
    state = ckpt.get("model", ckpt.get("model_state_dict", ckpt))
    model.load_state_dict(state, strict=False)
    
    model.to(device)
    model.eval()
    return model, device


def load_registered_embeddings(path, device):
    path = Path(path)
    emb_db = {}
    
    if path.is_file():
        files = [path]
    elif path.is_dir():
        files = list(path.rglob("embeddings.json"))
    else:
        return {}

    for json_file in files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            for name, emb_list in data.items():
                emb_db[name] = torch.tensor(emb_list, dtype=torch.float32).unsqueeze(0).to(device)
                print(f"   • Loaded: {name}")
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
            
    return emb_db


def extract_features(model, face_tensor, device):
    with torch.no_grad():
        inputs = {
            'vis': face_tensor.to(device),
            'depth': None,
            'normals': None,
            'mesh': None
        }
        
        out = model(inputs)
        
        # Embedding
        emb = out.get("embeddings", out) if isinstance(out, dict) else out
        emb = F.normalize(emb, p=2, dim=1)
        
        # Spoof Score
        spoof_prob = 0.5  # Default neutral
        
        if isinstance(out, dict) and "spoof_score" in out:
            spoof_tensor = out["spoof_score"]
            if spoof_tensor is not None:
                spoof_prob = torch.sigmoid(spoof_tensor).item()
        
        return emb, spoof_prob


def nothing(x):
    pass


def run_realtime(model_path, emb_path, camera_id=0):
    # Load resources
    model, device = load_model(model_path)
    emb_db = load_registered_embeddings(emb_path, device)
    
    if not emb_db:
        print("⚠ No embeddings found! Run extract_embeddings.py first.")
        return
    
    # Setup
    cap = cv2.VideoCapture(camera_id)
    window_name = "Attend3D Face Recognition - FIXED"
    cv2.namedWindow(window_name)
    
    # BETTER THRESHOLDS
    cv2.createTrackbar("ID Threshold", window_name, 75, 100, nothing)  # 0.75
    cv2.createTrackbar("Spoof Threshold", window_name, 50, 100, nothing)  # 0.50
    
    # Liveness detector
    liveness = LivenessDetector()
    
    # Smoothing buffers
    spoof_buffer = deque(maxlen=10)
    similarity_buffer = deque(maxlen=5)
    
    print("\n SYSTEM READY")
    print("Thresholds:")
    print("  - ID: 0.75 (recommended range: 0.70-0.80)")
    print("  - Spoof: 0.50 (recommended range: 0.45-0.55)")
    print("\nPress 'q' to quit\n")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Get thresholds
        id_thresh = cv2.getTrackbarPos("ID Threshold", window_name) / 100.0
        spoof_thresh = cv2.getTrackbarPos("Spoof Threshold", window_name) / 100.0
        
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = detector.process(rgb)
        
        display_frame = frame.copy()
        
        # Liveness check
        is_blink, blink_count = liveness.detect_blink(frame)
        has_liveness = blink_count >= 2  # At least 2 blinks
        
        if results.detections:
            for det in results.detections:
                bbox = det.location_data.relative_bounding_box
                x1 = int(bbox.xmin * w)
                y1 = int(bbox.ymin * h)
                x2 = int((bbox.xmin + bbox.width) * w)
                y2 = int((bbox.ymin + bbox.height) * h)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                face_img = frame[y1:y2, x1:x2]
                if face_img.size == 0:
                    continue
                
                # Extract features
                face_pil = transforms.ToPILImage()(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
                face_tensor = rgb_tf(face_pil).unsqueeze(0)
                
                emb, spoof_prob = extract_features(model, face_tensor, device)
                
                # Smooth spoof score
                spoof_buffer.append(spoof_prob)
                avg_spoof = np.mean(spoof_buffer)
                
                # Lower spoof score = more real
                # spoof_prob < 0.5 → Real
                # spoof_prob > 0.5 → Fake
                is_real_model = avg_spoof < spoof_thresh
                
                # COMBINED: Model + Liveness
                is_real = is_real_model and (has_liveness or frame_count < 60)
                
                # Identity matching
                best_name = "Unknown"
                best_sim = 0.0
                
                if is_real and len(emb_db) > 0:
                    for name, db_emb in emb_db.items():
                        sim = F.cosine_similarity(emb, db_emb).item()
                        if sim > best_sim:
                            best_sim = sim
                            best_name = name
                    
                    # Smooth similarity
                    similarity_buffer.append(best_sim)
                    avg_sim = np.mean(similarity_buffer)
                    
                    if avg_sim < id_thresh:
                        best_name = "Unknown"
                
                # ========== VISUALIZATION ==========
                if not is_real:
                    color = (0, 0, 255)  # RED - FAKE
                    label_top = "⚠ FAKE DETECTED"
                    label_bottom = f"Spoof: {avg_spoof:.2f} (>{spoof_thresh:.2f})"
                    
                    if not has_liveness:
                        label_bottom += " | No blink"
                else:
                    if best_name == "Unknown":
                        color = (0, 165, 255)  # ORANGE - UNKNOWN
                        label_top = "❓ Unknown Person"
                    else:
                        color = (0, 255, 0)  # GREEN - RECOGNIZED
                        label_top = f"✓ {best_name}"
                    
                    label_bottom = f"Spoof: {avg_spoof:.2f} | ID: {best_sim:.2f}"
                
                # Draw
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                
                # Top label
                cv2.rectangle(display_frame, (x1, y1-30), (x2, y1), color, -1)
                cv2.putText(display_frame, label_top, (x1+5, y1-8),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                
                # Bottom label
                cv2.rectangle(display_frame, (x1, y2), (x2, y2+30), color, -1)
                cv2.putText(display_frame, label_bottom, (x1+5, y2+20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        
        # ========== INFO PANEL ==========
        panel_h = 120
        cv2.rectangle(display_frame, (5, 5), (400, panel_h), (0, 0, 0), -1)
        
        y_offset = 25
        cv2.putText(display_frame, f"ID Threshold: {id_thresh:.2f}",
                   (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        
        y_offset += 25
        cv2.putText(display_frame, f"Spoof Threshold: {spoof_thresh:.2f}",
                   (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        
        y_offset += 25
        liveness_color = (0, 255, 0) if has_liveness else (0, 0, 255)
        cv2.putText(display_frame, f"Blink Count: {blink_count}",
                   (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, liveness_color, 1)
        
        y_offset += 25
        cv2.putText(display_frame, "Press 'q' to quit",
                   (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        cv2.imshow(window_name, display_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True,
                       help='Path to model checkpoint (.pth)')
    parser.add_argument('--embeddings', type=str, required=True,
                       help='Path to embeddings folder/file')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera ID (default: 0)')
    args = parser.parse_args()
    
    run_realtime(args.model, args.embeddings, args.camera)