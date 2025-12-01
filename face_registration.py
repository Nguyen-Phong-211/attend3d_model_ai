"""
Face Registration System V2.0 - Face ID Style
Thu thập video 5 giây liên tục cho mỗi góc độ với UX mượt mà
"""

import cv2
import numpy as np
import mediapipe as mp
import os
import json
from datetime import datetime
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from collections import deque
import time
import random

class FaceRegistrationV2:
    def __init__(self, output_dir="registered_faces"):
        """
        Hệ thống đăng ký khuôn mặt phong cách Face ID
        Thu thập video 5s liên tục cho mỗi góc độ
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # MediaPipe setup
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            min_detection_confidence=0.7
        )
        
        # Cấu hình góc độ - giống Face ID
        self.capture_sequence = [
            {
                'name': 'center',
                'display_name': 'Nhìn thẳng',
                'yaw_range': (-15, 15),
                'pitch_range': (-15, 15),
                'duration': 5.0,  # giây
                'instruction': 'Giữ đầu thẳng và nhìn vào camera'
            },
            {
                'name': 'left_slow',
                'display_name': 'Quay trái từ từ',
                'yaw_range': (15, 60),
                'pitch_range': (-15, 15),
                'duration': 5.0,
                'instruction': 'Từ từ quay đầu sang trái'
            },
            {
                'name': 'right_slow',
                'display_name': 'Quay phải từ từ',
                'yaw_range': (-60, -15),
                'pitch_range': (-15, 15),
                'duration': 5.0,
                'instruction': 'Từ từ quay đầu sang phải'
            },
            {
                'name': 'up_slow',
                'display_name': 'Ngửa đầu từ từ',
                'yaw_range': (-15, 15),
                'pitch_range': (15, 45),
                'duration': 5.0,
                'instruction': 'Từ từ ngửa đầu lên'
            },
            {
                'name': 'down_slow',
                'display_name': 'Cúi đầu từ từ',
                'yaw_range': (-15, 15),
                'pitch_range': (-45, -15),
                'duration': 5.0,
                'instruction': 'Từ từ cúi đầu xuống'
            },
            {
                'name': 'circle_motion',
                'display_name': 'Xoay 360°',
                'yaw_range': (-90, 90),
                'pitch_range': (-30, 30),
                'duration': 8.0,
                'instruction': 'Từ từ xoay đầu theo vòng tròn'
            }
        ]
        
        self.current_step = 0
        self.session_data = {
            'videos': [],
            'landmarks': [],
            'quality_scores': []
        }
        
        # Smoothing
        self.angle_buffer = deque(maxlen=5)
        self.quality_buffer = deque(maxlen=10)
        
        # Liveness detection
        self.blink_counter = 0
        self.blink_threshold = 2  # Cần ít nhất 2 cái nháy mắt
        
    def estimate_head_pose(self, face_landmarks, image_shape):
        """Ước tính góc quay đầu với độ chính xác cao"""
        img_h, img_w = image_shape[:2]
        
        # Sử dụng nhiều điểm landmark hơn để tăng độ chính xác
        key_points = [1, 33, 263, 61, 291, 199, 168, 6, 10, 152, 234, 454]
        
        face_2d = []
        face_3d = []
        
        for idx in key_points:
            lm = face_landmarks.landmark[idx]
            x, y = int(lm.x * img_w), int(lm.y * img_h)
            face_2d.append([x, y])
            face_3d.append([x, y, lm.z * img_w])
        
        face_2d = np.array(face_2d, dtype=np.float64)
        face_3d = np.array(face_3d, dtype=np.float64)
        
        focal_length = 1.0 * img_w
        cam_matrix = np.array([
            [focal_length, 0, img_w / 2],
            [0, focal_length, img_h / 2],
            [0, 0, 1]
        ])
        dist_matrix = np.zeros((4, 1), dtype=np.float64)
        
        success, rot_vec, trans_vec = cv2.solvePnP(
            face_3d, face_2d, cam_matrix, dist_matrix,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        rmat, _ = cv2.Rodrigues(rot_vec)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
        
        yaw = angles[1] * 360
        pitch = angles[0] * 360
        roll = angles[2] * 360
        
        # Smooth angles
        self.angle_buffer.append((yaw, pitch, roll))
        if len(self.angle_buffer) > 0:
            yaw = np.mean([a[0] for a in self.angle_buffer])
            pitch = np.mean([a[1] for a in self.angle_buffer])
            roll = np.mean([a[2] for a in self.angle_buffer])
        
        return yaw, pitch, roll
    
    def detect_blink(self, face_landmarks):
        """
        Phát hiện nháy mắt để xác nhận liveness
        Sử dụng Eye Aspect Ratio (EAR)
        """
        # Left eye landmarks
        left_eye = [362, 385, 387, 263, 373, 380]
        # Right eye landmarks  
        right_eye = [33, 160, 158, 133, 153, 144]
        
        def eye_aspect_ratio(eye_points):
            # Vertical distances
            v1 = np.linalg.norm(np.array([face_landmarks.landmark[eye_points[1]].x, 
                                          face_landmarks.landmark[eye_points[1]].y]) - 
                               np.array([face_landmarks.landmark[eye_points[5]].x,
                                        face_landmarks.landmark[eye_points[5]].y]))
            v2 = np.linalg.norm(np.array([face_landmarks.landmark[eye_points[2]].x,
                                          face_landmarks.landmark[eye_points[2]].y]) - 
                               np.array([face_landmarks.landmark[eye_points[4]].x,
                                        face_landmarks.landmark[eye_points[4]].y]))
            # Horizontal distance
            h = np.linalg.norm(np.array([face_landmarks.landmark[eye_points[0]].x,
                                        face_landmarks.landmark[eye_points[0]].y]) - 
                              np.array([face_landmarks.landmark[eye_points[3]].x,
                                       face_landmarks.landmark[eye_points[3]].y]))
            
            ear = (v1 + v2) / (2.0 * h)
            return ear
        
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        avg_ear = (left_ear + right_ear) / 2.0
        
        # EAR threshold for blink detection
        return avg_ear < 0.2
    
    def calculate_quality_score(self, frame, face_landmarks, yaw, pitch, roll):
        """
        Tính điểm chất lượng của frame (0-100)
        Dựa trên: độ sáng, blur, kích thước khuôn mặt, góc quay
        """
        score = 100.0
        
        # 1. Kiểm tra độ sáng
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        if brightness < 60 or brightness > 200:
            score -= 20
        
        # 2. Kiểm tra blur (Laplacian variance)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if laplacian_var < 100:
            score -= 30
        
        # 3. Kiểm tra kích thước khuôn mặt
        h, w = frame.shape[:2]
        face_width = 0
        face_height = 0
        
        for lm in face_landmarks.landmark:
            x = lm.x * w
            y = lm.y * h
            face_width = max(face_width, x)
            face_height = max(face_height, y)
        
        face_area = face_width * face_height / (w * h)
        if face_area < 0.15:
            score -= 25
        
        # 4. Kiểm tra góc quay (roll không được quá lệch)
        if abs(roll) > 20:
            score -= 15
        
        return max(0, min(100, score))
    
    def draw_modern_ui(self, frame, progress, step_info, yaw, pitch, roll, quality_score):
        """
        Vẽ UI hiện đại giống Face ID
        """
        h, w = frame.shape[:2]
        overlay = frame.copy()
        
        # 1. Vẽ vòng tròn hướng dẫn (giống Face ID)
        center = (w // 2, h // 2)
        radius = 180
        
        # Vẽ vòng tròn nền
        cv2.circle(overlay, center, radius + 10, (50, 50, 50), 8)
        
        # Vẽ vòng tròn progress
        angle = int(360 * progress)
        if progress < 1.0:
            color = (0, 165, 255)  # Màu cam
        else:
            color = (0, 255, 0)  # Màu xanh lá
        
        # Vẽ arc
        cv2.ellipse(overlay, center, (radius + 10, radius + 10), 
                   -90, 0, angle, color, 12)
        
        # 2. Vẽ chấm ở giữa (theo dõi khuôn mặt)
        face_center = (int(w // 2 + yaw * 2), int(h // 2 + pitch * 2))
        cv2.circle(overlay, face_center, 15, (255, 255, 255), -1)
        cv2.circle(overlay, face_center, 15, color, 3)
        
        # 3. Progress bar ở trên
        bar_y = 40
        bar_height = 12
        bar_width = w - 100
        bar_x = 50
        
        cv2.rectangle(overlay, (bar_x, bar_y), 
                     (bar_x + bar_width, bar_y + bar_height),
                     (80, 80, 80), -1)
        
        progress_width = int(bar_width * (self.current_step + progress) / len(self.capture_sequence))
        cv2.rectangle(overlay, (bar_x, bar_y),
                     (bar_x + progress_width, bar_y + bar_height),
                     (0, 255, 0), -1)
        
        # 4. Text hướng dẫn
        alpha = 0.7
        frame_with_overlay = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        
        # Instruction text
        instruction = step_info['instruction']
        frame_with_overlay = self.put_text_vietnamese(
            frame_with_overlay, instruction,
            (w // 2 - 200, h - 80),
            font_size=28,
            color=(255, 255, 255)
        )
        
        # Step indicator
        step_text = f"Bước {self.current_step + 1}/{len(self.capture_sequence)}"
        frame_with_overlay = self.put_text_vietnamese(
            frame_with_overlay, step_text,
            (w // 2 - 80, 80),
            font_size=24,
            color=(255, 255, 255)
        )
        
        # Quality indicator
        quality_color = (0, 255, 0) if quality_score > 70 else (0, 165, 255) if quality_score > 50 else (0, 0, 255)
        cv2.putText(frame_with_overlay, f"Quality: {int(quality_score)}%",
                   (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, quality_color, 2)
        
        return frame_with_overlay
    
    def put_text_vietnamese(self, img, text, position, font_size=30, color=(255, 255, 255)):
        """Vẽ text tiếng Việt bằng PIL"""
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
            except:
                font = ImageFont.load_default()
        
        draw.text(position, text, font=font, fill=color)
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    
    def capture_step_video(self, cap, step_info, video_writer):
        """
        Thu thập video 5 giây cho một góc độ
        """
        duration = step_info['duration']
        start_time = time.time()
        
        frame_data = []
        last_blink_state = False
        
        print(f"\n▶ Bắt đầu: {step_info['display_name']}")
        print(f"  {step_info['instruction']}")
        
        while time.time() - start_time < duration:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect face
            face_mesh_result = self.face_mesh.process(rgb_frame)
            
            if face_mesh_result.multi_face_landmarks:
                face_landmarks = face_mesh_result.multi_face_landmarks[0]
                yaw, pitch, roll = self.estimate_head_pose(face_landmarks, frame.shape)
                
                # Detect blink
                is_blinking = self.detect_blink(face_landmarks)
                if is_blinking and not last_blink_state:
                    self.blink_counter += 1
                last_blink_state = is_blinking
                
                # Calculate quality
                quality_score = self.calculate_quality_score(frame, face_landmarks, yaw, pitch, roll)
                
                # Save frame data
                landmarks_data = [{'x': lm.x, 'y': lm.y, 'z': lm.z} 
                                 for lm in face_landmarks.landmark]
                
                frame_data.append({
                    'timestamp': time.time() - start_time,
                    'landmarks': landmarks_data,
                    'angles': {'yaw': yaw, 'pitch': pitch, 'roll': roll},
                    'quality_score': quality_score
                })
                
                # Draw UI
                progress = (time.time() - start_time) / duration
                frame_display = self.draw_modern_ui(
                    frame, progress, step_info, yaw, pitch, roll, quality_score
                )
                
            else:
                # No face detected
                progress = (time.time() - start_time) / duration
                frame_display = self.draw_modern_ui(
                    frame, progress, step_info, 0, 0, 0, 0
                )
                frame_display = self.put_text_vietnamese(
                    frame_display, "⚠ Không phát hiện khuôn mặt",
                    (frame.shape[1] // 2 - 150, frame.shape[0] // 2),
                    font_size=24,
                    color=(0, 0, 255)
                )
            
            video_writer.write(frame_display)
            cv2.imshow('Face Registration V2', frame_display)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                return None
        
        # Tính average quality score
        avg_quality = np.mean([f['quality_score'] for f in frame_data]) if frame_data else 0
        
        return {
            'step_name': step_info['name'],
            'frames': frame_data,
            'duration': duration,
            'avg_quality': avg_quality,
            'blink_count': self.blink_counter
        }
    
    def register_user(self, user_id):
        """
        Bắt đầu quy trình đăng ký
        """
        random_number = random.randint(1000000, 9999999)
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = self.output_dir / user_id / f"{user_id}_{session_id}_{random_number}"
        session_dir.mkdir(parents=True, exist_ok=True)
        
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Setup video writer
        video_path = session_dir / f"{session_dir.name}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(str(video_path), fourcc, 30, (1280, 720))
        
        print(f"\n{'='*60}")
        print(f"ĐĂNG KÝ KHUÔN MẶT - PHONG CÁCH FACE ID")
        print(f"User: {user_id}")
        print(f"{'='*60}\n")
        print("Hướng dẫn:")
        print("  • Giữ khuôn mặt trong vòng tròn")
        print("  • Làm theo hướng dẫn trên màn hình")
        print("  • Mỗi động tác kéo dài 5-8 giây")
        print("  • Nhấn 'q' để hủy\n")
        
        all_step_data = []
        
        # Countdown 3s
        for i in range(3, 0, -1):
            ret, frame = cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                frame = self.put_text_vietnamese(
                    frame, f"Bắt đầu sau {i}...",
                    (frame.shape[1] // 2 - 100, frame.shape[0] // 2),
                    font_size=48,
                    color=(255, 255, 255)
                )
                cv2.imshow('Face Registration V2', frame)
                cv2.waitKey(1000)
        
        # Capture từng bước
        for step_idx, step_info in enumerate(self.capture_sequence):
            self.current_step = step_idx
            self.blink_counter = 0
            
            step_data = self.capture_step_video(cap, step_info, video_writer)
            
            if step_data is None:
                print("\nĐã hủy đăng ký")
                cap.release()
                video_writer.release()
                cv2.destroyAllWindows()
                return False
            
            all_step_data.append(step_data)
            print(f"  ✓ Hoàn thành: Quality {step_data['avg_quality']:.1f}%, Blinks: {step_data['blink_count']}")
        
        cap.release()
        video_writer.release()
        cv2.destroyAllWindows()
        
        # Save metadata
        metadata = {
            'user_id': user_id,
            'session_id': session_id,
            'timestamp': datetime.now().isoformat(),
            'steps': all_step_data,
            'video_path': str(video_path)
        }
        
        with open(session_dir / 'metadata.json', 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        # Save landmarks cho từng step
        for step_data in all_step_data:
            step_name = step_data['step_name']
            landmarks_file = session_dir / f"landmarks_{step_name}.json"
            
            with open(landmarks_file, 'w', encoding='utf-8') as f:
                json.dump(step_data['frames'], f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"ĐĂNG KÝ THÀNH CÔNG!")
        print(f"Dữ liệu đã lưu tại: {session_dir}")
        print(f"{'='*60}\n")
        
        return True


def main():
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python face_registration_v2.py <user_id>")
        print("\nExample:")
        print("  python face_registration_v2.py john_doe")
        sys.exit(1)
    
    user_id = sys.argv[1]
    system = FaceRegistrationV2()
    success = system.register_user(user_id)
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()