"""
Utility functions for face recognition system
"""

import json
from pathlib import Path
import cv2
import numpy as np
from datetime import datetime


class EmbeddingsManager:
    """Quản lý embeddings database"""
    
    @staticmethod
    def merge_embeddings(input_paths, output_path):
        """
        Gộp nhiều embeddings.json thành 1 file
        
        Args:
            input_paths: List các đường dẫn embeddings.json
            output_path: Đường dẫn output
        """
        merged = {}
        
        for path in input_paths:
            with open(path, 'r') as f:
                data = json.load(f)
            
            for user_id, embedding in data.items():
                if user_id in merged:
                    print(f"  Duplicate: {user_id} (skipping)")
                    continue
                merged[user_id] = embedding
        
        with open(output_path, 'w') as f:
            json.dump(merged, f, indent=2)
        
        print(f" Merged {len(merged)} identities → {output_path}")
    
    @staticmethod
    def list_identities(embeddings_path):
        """List tất cả identities trong database"""
        embeddings_path = Path(embeddings_path)
        
        if embeddings_path.is_file():
            with open(embeddings_path, 'r') as f:
                data = json.load(f)
            
            print(f"\n Identities in {embeddings_path.name}:")
            for i, user_id in enumerate(data.keys(), 1):
                print(f"  {i}. {user_id}")
            print(f"\nTotal: {len(data)}")
        
        elif embeddings_path.is_dir():
            json_files = list(embeddings_path.rglob("embeddings.json"))
            
            all_identities = set()
            for json_file in json_files:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                all_identities.update(data.keys())
            
            print(f"\n Identities in {embeddings_path}:")
            for i, user_id in enumerate(sorted(all_identities), 1):
                print(f"  {i}. {user_id}")
            print(f"\nTotal: {len(all_identities)}")
    
    @staticmethod
    def remove_identity(embeddings_path, user_id):
        """Xóa 1 identity khỏi database"""
        with open(embeddings_path, 'r') as f:
            data = json.load(f)
        
        if user_id in data:
            del data[user_id]
            
            with open(embeddings_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            print(f" Removed: {user_id}")
        else:
            print(f" Not found: {user_id}")


class VideoAnalyzer:
    """Phân tích video đăng ký"""
    
    @staticmethod
    def analyze_registration_video(video_path):
        """
        Phân tích video đăng ký
        In thông tin: fps, resolution, duration, frame count
        """
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            print(f" Cannot open video: {video_path}")
            return
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        
        print(f"\n Video Analysis: {Path(video_path).name}")
        print(f"{'='*50}")
        print(f"  Resolution: {width}x{height}")
        print(f"  FPS: {fps:.2f}")
        print(f"  Frame count: {frame_count}")
        print(f"  Duration: {duration:.2f}s")
        print(f"{'='*50}\n")
        
        cap.release()
    
    @staticmethod
    def extract_frames(video_path, output_dir, interval=1.0):
        """
        Extract frames từ video
        
        Args:
            video_path: Đường dẫn video
            output_dir: Thư mục lưu frames
            interval: Khoảng cách giữa các frame (giây)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            print(f" Cannot open video: {video_path}")
            return
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps * interval)
        
        frame_idx = 0
        saved_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % frame_interval == 0:
                output_path = output_dir / f"frame_{saved_count:04d}.jpg"
                cv2.imwrite(str(output_path), frame)
                saved_count += 1
            
            frame_idx += 1
        
        cap.release()
        print(f" Extracted {saved_count} frames → {output_dir}")


class QualityChecker:
    """Kiểm tra chất lượng ảnh"""
    
    @staticmethod
    def check_brightness(image):
        """Kiểm tra độ sáng"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        
        if brightness < 60:
            return "Too dark", brightness
        elif brightness > 200:
            return "Too bright", brightness
        else:
            return "Good", brightness
    
    @staticmethod
    def check_blur(image):
        """Kiểm tra blur (Laplacian variance)"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        if laplacian_var < 100:
            return "Blurry", laplacian_var
        else:
            return "Sharp", laplacian_var
    
    @staticmethod
    def analyze_image(image_path):
        """Phân tích chất lượng ảnh"""
        image = cv2.imread(str(image_path))
        
        if image is None:
            print(f" Cannot read image: {image_path}")
            return
        
        brightness_status, brightness_val = QualityChecker.check_brightness(image)
        blur_status, blur_val = QualityChecker.check_blur(image)
        
        print(f"\n Image Quality: {Path(image_path).name}")
        print(f"{'='*50}")
        print(f"  Brightness: {brightness_status} ({brightness_val:.1f})")
        print(f"  Sharpness: {blur_status} ({blur_val:.1f})")
        print(f"{'='*50}\n")


def create_demo_structure():
    """Tạo cấu trúc thư mục demo"""
    structure = """
    project/
    │
    ├── checkpoints/
    │   └── best_model.pth
    │
    ├── registered_faces/
    │   ├── john_doe/
    │   │   └── john_doe_20231129_083823_1786228/
    │   │       ├── john_doe_20231129_083823_1786228.mp4
    │   │       ├── metadata.json
    │   │       ├── landmarks_*.json
    │   │       └── embeddings.json  (← generated by extract_embeddings.py)
    │   │
    │   └── alice/
    │       └── alice_20231129_084523_2847123/
    │           ├── alice_20231129_084523_2847123.mp4
    │           ├── metadata.json
    │           ├── landmarks_*.json
    │           └── embeddings.json
    │
    ├── face_registration.py
    ├── extract_embeddings.py
    ├── face_recognition_realtime_fixed.py
    ├── utils.py
    ├── model.py
    ├── config.py
    └── ...
    """
    print(structure)


def print_workflow():
    """In workflow sử dụng"""
    workflow = """
    ╔═══════════════════════════════════════════════════════════════╗
    ║          WORKFLOW: 3D FACE RECOGNITION SYSTEM                 ║
    ╚═══════════════════════════════════════════════════════════════╝
    
    STEP 1: Đăng ký khuôn mặt (Face Registration)
    ─────────────────────────────────────────────
    $ python face_registration.py john_doe
    
    → Output: 
      registered_faces/john_doe/john_doe_YYYYMMDD_HHMMSS_XXXXXX/
        ├── john_doe_YYYYMMDD_HHMMSS_XXXXXX.mp4
        ├── metadata.json
        └── landmarks_*.json
    
    
    STEP 2: Extract embeddings từ video
    ────────────────────────────────────
    $ python extract_embeddings.py \
        --model checkpoints/best_model.pth \
        --input registered_faces \
        --device cpu
    
    → Output:
      registered_faces/john_doe/john_doe_YYYYMMDD_HHMMSS_XXXXXX/
        └── embeddings.json  ← NEW!
    
    
    STEP 3: Chạy realtime recognition
    ──────────────────────────────────
    $ python face_recognition_realtime_fixed.py \
        --model checkpoints/best_model.pth \
        --embeddings registered_faces \
        --device cpu \
        --threshold 0.55
    
    → Mở camera và nhận diện realtime
    
    
    ═══════════════════════════════════════════════════════════════
    UTILITIES
    ═══════════════════════════════════════════════════════════════
    
    • List identities:
      from utils import EmbeddingsManager
      EmbeddingsManager.list_identities('registered_faces')
    
    • Merge embeddings:
      EmbeddingsManager.merge_embeddings(
          ['path1/embeddings.json', 'path2/embeddings.json'],
          'merged_embeddings.json'
      )
    
    • Analyze video:
      from utils import VideoAnalyzer
      VideoAnalyzer.analyze_registration_video('video.mp4')
    
    • Check image quality:
      from utils import QualityChecker
      QualityChecker.analyze_image('image.jpg')
    """
    print(workflow)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("\n📚 USAGE:")
        print("  python utils.py workflow    # Show workflow")
        print("  python utils.py structure   # Show folder structure")
        print("  python utils.py list <path> # List identities")
        sys.exit(0)
    
    command = sys.argv[1]
    
    if command == "workflow":
        print_workflow()
    
    elif command == "structure":
        create_demo_structure()
    
    elif command == "list":
        if len(sys.argv) < 3:
            print(" Usage: python utils.py list <embeddings_path>")
            sys.exit(1)
        
        EmbeddingsManager.list_identities(sys.argv[2])
    
    else:
        print(f" Unknown command: {command}")