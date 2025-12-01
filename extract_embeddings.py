"""
Extract embeddings from registered face videos - MULTI-ANGLE VERSION
Lấy embeddings từ NHIỀU góc độ để tăng robustness
"""

import cv2
import torch
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

from model import Face3DFusionModel
from config import config


class EmbeddingExtractorFixed:
    def __init__(self, model_path, device='cpu'):
        self.device = torch.device(device)
        print(f"DEVICE: {self.device}")
        
        # Load model
        print(f"LOADING MODEL FROM: {model_path}")
        self.model = self._load_model(model_path)
        self.model.eval()
        
        # Transform
        self.rgb_transform = transforms.Compose([
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        print("MODEL LOADED SUCCESSFULLY\n")
    
    def _load_model(self, checkpoint_path):
        try:
            with torch.serialization.safe_globals([np.core.multiarray.scalar]):
                ckpt = torch.load(checkpoint_path, map_location=self.device)
        except:
            ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        num_classes = ckpt.get("num_classes", config.NUM_CLASSES)
        
        model = Face3DFusionModel(num_classes=num_classes, config=config)
        
        state_dict = ckpt.get("model", ckpt.get("model_state_dict", ckpt))
        model.load_state_dict(state_dict, strict=False)
        
        model.to(self.device)
        return model
    
    def extract_from_frame(self, frame):
        """Extract embedding from 1 frame"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)
        
        frame_tensor = self.rgb_transform(frame_pil).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            inputs = {
                'vis': frame_tensor,
                'depth': None,
                'normals': None,
                'mesh': None
            }
            
            outputs = self.model(inputs)
            embeddings = outputs['embeddings']
            embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings.cpu()
    
    def extract_from_video_multiangle(self, video_path, sample_every=5):
        """
        Extract embeddings from MULTIPLE angles
        - Lấy frames từ đầu đến cuối video
        - Sample mỗi 5 frames để tránh redundancy
        - Chọn top K frames có quality cao nhất
        """
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError(f"CANNOT OPEN VIDEO: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"VIDEO: {Path(video_path).name}")
        print(f"TOTAL FRAMES: {total_frames}")
        
        embeddings_list = []
        frame_qualities = []
        frame_idx = 0
        
        pbar = tqdm(total=total_frames // sample_every, desc="Extracting")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Sample frames
            if frame_idx % sample_every == 0:
                # Calculate quality score
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
                brightness = np.mean(gray)
                
                # Penalize too bright/dark
                brightness_penalty = 1 - abs(brightness - 128) / 128
                quality = laplacian_var * brightness_penalty
                
                # Extract embedding
                embedding = self.extract_from_frame(frame)
                embeddings_list.append(embedding)
                frame_qualities.append(quality)
                pbar.update(1)
            
            frame_idx += 1
        
        cap.release()
        pbar.close()
        
        if len(embeddings_list) == 0:
            raise ValueError("NO EMBEDDINGS EXTRACTED")
        
        # Select top 30% highest quality frames
        quality_array = np.array(frame_qualities)
        top_k = max(10, len(embeddings_list) // 3)  # At least 10 frames
        top_indices = np.argsort(quality_array)[-top_k:]
        
        selected_embeddings = [embeddings_list[i] for i in top_indices]
        
        print(f"SELECTED TOP {top_k}/{len(embeddings_list)} FRAMES")
        print(f"QUALITY RANGE: {quality_array[top_indices].min():.1f} - {quality_array[top_indices].max():.1f}")
        
        # Stack and average
        all_embeddings = torch.cat(selected_embeddings, dim=0)  # (N, 512)
        mean_embedding = all_embeddings.mean(dim=0, keepdim=True)  # (1, 512)
        
        # Re-normalize after averaging
        mean_embedding = F.normalize(mean_embedding, p=2, dim=1)
        
        print(f"FINAL EMBEDDING SHAPE: {mean_embedding.shape}\n")
        
        return mean_embedding, all_embeddings
    
    def process_registration_folder(self, registration_folder, output_name="embeddings.json"):
        """Process single registration folder"""
        folder_path = Path(registration_folder)
        
        if not folder_path.exists():
            raise ValueError(f"INVALID FOLDER: {folder_path}")
        
        video_files = list(folder_path.glob("*.mp4"))
        
        if len(video_files) == 0:
            raise ValueError(f"NO VIDEO FILES FOUND: {folder_path}")
        
        video_path = video_files[0]
        
        # Use multi-angle extraction
        mean_emb, all_embs = self.extract_from_video_multiangle(video_path)
        
        # Get user ID
        metadata_path = folder_path / "metadata.json"
        user_id = folder_path.parent.name
        
        if metadata_path.exists():
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
                user_id = metadata.get('user_id', user_id)
        
        # Save embeddings
        output_path = folder_path / output_name
        
        embeddings_data = {
            user_id: mean_emb.squeeze(0).tolist()
        }
        
        with open(output_path, 'w') as f:
            json.dump(embeddings_data, f, indent=2)
        
        print(f"SAVING EMBEDDINGS TO: {output_path}")
        print(f"USER ID: {user_id}")
        
        return output_path
    
    def process_multiple_users(self, registered_faces_root):
        """Process multiple users"""
        root_path = Path(registered_faces_root)
        
        if not root_path.exists():
            raise ValueError(f"INVALID FOLDER: {root_path}")
        
        print(f"PROCESSING FOLDER: {root_path}\n")
        
        session_folders = []
        
        for user_folder in root_path.iterdir():
            if not user_folder.is_dir():
                continue
            
            for session_folder in user_folder.iterdir():
                if not session_folder.is_dir():
                    continue
                
                if any(session_folder.glob("*.mp4")):
                    session_folders.append(session_folder)
        
        if len(session_folders) == 0:
            print("NO REGISTRATION SESSIONS FOUND")
            return []
        
        print(f"FOUND {len(session_folders)} REGISTRATION SESSIONS\n")
        
        results = []
        
        for session_folder in session_folders:
            print(f"{'='*60}")
            try:
                output_path = self.process_registration_folder(session_folder)
                results.append(output_path)
            except Exception as e:
                print(f"ERROR PROCESSING {session_folder}: {e}")
        
        print(f"\n{'='*60}")
        print(f"PROCESSED {len(results)}/{len(session_folders)} SESSIONS")
        
        return results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Extract embeddings - MULTI-ANGLE VERSION'
    )
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda', 'mps'])
    
    args = parser.parse_args()
    
    extractor = EmbeddingExtractorFixed(args.model, device=args.device)
    
    input_path = Path(args.input)
    
    if any(input_path.glob("*.mp4")):
        print("PROCESSING SINGLE SESSION...\n")
        extractor.process_registration_folder(input_path)
    else:
        print("PROCESSING MULTIPLE SESSIONS...\n")
        extractor.process_multiple_users(input_path)


if __name__ == "__main__":
    main()