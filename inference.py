# inference.py - Production inference tool
import torch
import cv2
import numpy as np
import os
from PIL import Image
from torchvision import transforms
from model import create_model
from config import config
from utils import load_mesh_vertices
import json

class Face3DInference:
    """
    Inference engine cho Face Recognition + Anti-Spoofing
    Tương tự Face ID của iPhone và app ngân hàng
    """
    
    def __init__(self, checkpoint_path, device=None):
        self.config = config
        self.device = device or torch.device(config.DEVICE)

        self.idx_to_class = {}
        checkpoint_dir = os.path.dirname(checkpoint_path)
        map_path = os.path.join(checkpoint_dir, 'label_map.json')

        if os.path.exists(map_path):
            try:
                with open(map_path, 'r') as f:
                    data = json.load(f)
                    self.idx_to_class = {int(k): v for k, v in data.items()}
                print(f"Loaded label map with {len(self.idx_to_class)} identities")
            except Exception as e:
                print(f"Warning: Could not load label_map.json: {e}")
        else:
            print("Warning: label_map.json not found. Predictions will return IDs only.")
        
        # Load model
        print(f"Loading model from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Lấy num_classes từ checkpoint nếu có
        if 'num_classes' in checkpoint:
            num_classes = checkpoint['num_classes']
        else:
            # Ước lượng từ model state
            num_classes = checkpoint['model']['arcface.weight'].shape[0]
        
        self.model = create_model(num_classes, config)
        self.model.load_state_dict(checkpoint['model'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded on {self.device}")
        print(f"Num identities: {num_classes}")
        
        # Transforms
        self.rgb_tf = transforms.Compose([
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])
        
        self.norm_tf = transforms.Compose([
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
        ])
        
        self.depth_tf = transforms.Compose([
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.ToTensor(),
        ])
    
    def load_sample(self, folder_path):
        """Load sample từ thư mục"""
        folder_name = os.path.basename(folder_path)
        dataset_path = os.path.dirname(folder_path)
        
        inputs = {}
        
        # Load RGB
        vis_path = os.path.join(dataset_path, f"{folder_name}_vis.jpg")
        if not os.path.exists(vis_path):
            vis_path = os.path.join(dataset_path, f"{folder_name}_vis_original_size.jpg")
        
        if os.path.exists(vis_path):
            vis_img = Image.open(vis_path).convert('RGB')
            inputs['vis'] = self.rgb_tf(vis_img).unsqueeze(0)
        else:
            raise FileNotFoundError(f"Cannot find RGB image at {vis_path}")
        
        # Load depth
        depth_path = os.path.join(folder_path, f"{folder_name}_depth.jpg")
        if os.path.exists(depth_path):
            depth_img = Image.open(depth_path).convert('L')
            inputs['depth'] = self.depth_tf(depth_img).unsqueeze(0)
        
        # Load normals
        normal_path = os.path.join(folder_path, f"{folder_name}_normals.png")
        if os.path.exists(normal_path):
            normal_img = Image.open(normal_path).convert('RGB')
            inputs['normals'] = self.norm_tf(normal_img).unsqueeze(0)
        
        obj_path = os.path.join(folder_path, f"{folder_name}.obj")
        if os.path.exists(obj_path) and self.config.USE_MESH:
            mesh = load_mesh_vertices(obj_path, max_vertices=self.config.MESH_MAX_VERTICES)
            if mesh is not None:
                inputs['mesh'] = mesh.unsqueeze(0)
        
        return inputs
    
    @torch.no_grad()
    def predict(self, folder_path, return_embedding=False):
        """
        Dự đoán identity và phát hiện spoofing
        
        Returns:
            dict: {
                'identity': int,
                'confidence': float,
                'is_real': bool,
                'spoofing_score': float,  # 0-1, càng cao càng fake
                'embedding': tensor (nếu return_embedding=True)
            }
        """
        inputs = self.load_sample(folder_path)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(inputs)
        
        # Classification
        logits = outputs['logits']
        probabilities = torch.softmax(logits, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        
        # Anti-spoofing
        spoof_logit = outputs.get('spoof_score', None)
        spoofing_score = None
        is_real = True
        
        if spoof_logit is not None:
            spoof_prob = torch.sigmoid(spoof_logit).cpu().item()
            spoofing_score = spoof_prob
            is_real = (spoof_prob < self.config.SPOOF_THRESHOLD)

        pred_idx = int(predicted.item())
        pred_name = str(pred_idx)
        if self.idx_to_class and pred_idx in self.idx_to_class:
            pred_name = self.idx_to_class[pred_idx]
        
        result = {
            'identity_id': pred_idx,      
            'identity': pred_name,
            'confidence': float(confidence.item()),
            'is_real': is_real,
            'spoofing_score': spoofing_score
        }
        
        if return_embedding:
            result['embedding'] = outputs['embeddings'].cpu()
        
        return result
    
    def batch_predict(self, folder_paths):
        """Predict batch các samples"""
        results = []
        for folder_path in folder_paths:
            try:
                result = self.predict(folder_path)
                results.append(result)
            except Exception as e:
                print(f"Error predicting {folder_path}: {e}")
                results.append(None)
        return results
    
    def verify_face(self, folder_path1, folder_path2, threshold=0.5):
        """
        So sánh 2 khuôn mặt có phải cùng người không
        (Face verification)
        
        Returns:
            dict: {
                'is_same_person': bool,
                'similarity': float,  # cosine similarity
                'person1_is_real': bool,
                'person2_is_real': bool
            }
        """
        result1 = self.predict(folder_path1, return_embedding=True)
        result2 = self.predict(folder_path2, return_embedding=True)
        
        # Cosine similarity
        emb1 = result1['embedding']
        emb2 = result2['embedding']
        similarity = torch.cosine_similarity(emb1, emb2).item()
        
        return {
            'is_same_person': similarity >= threshold,
            'similarity': similarity,
            'person1_is_real': result1['is_real'],
            'person2_is_real': result2['is_real'],
            'person1_spoof_score': result1['spoofing_score'],
            'person2_spoof_score': result2['spoofing_score']
        }


def main():
    """Demo inference"""
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python inference.py <checkpoint_path> <folder_path>")
        print("\nExample:")
        print("  python inference.py checkpoints/best_acc.pth /path/to/frame_000001/")
        sys.exit(1)
    
    checkpoint_path = sys.argv[1]
    folder_path = sys.argv[2]
    
    # Initialize inference engine
    engine = Face3DInference(checkpoint_path)
    
    # Single prediction
    print(f"\nPredicting: {folder_path}")
    result = engine.predict(folder_path)
    
    print("\n" + "="*60)
    print("RESULT:")
    print(f"  Identity: {result['identity']}")
    print(f"  Confidence: {result['confidence']*100:.2f}%")
    print(f"  Is Real: {'✓ YES' if result['is_real'] else '✗ NO (FAKE)'}")
    print(f"  Spoofing Score: {result['spoofing_score']*100:.1f}% (0=real, 100=fake)")
    print("="*60)
    
    # Verification example (nếu có 2 folders)
    if len(sys.argv) >= 4:
        folder_path2 = sys.argv[3]
        print(f"\nVerifying: {folder_path} vs {folder_path2}")
        
        verify_result = engine.verify_face(folder_path, folder_path2)
        
        print("\n" + "="*60)
        print("VERIFICATION:")
        print(f"  Same Person: {'✓ YES' if verify_result['is_same_person'] else '✗ NO'}")
        print(f"  Similarity: {verify_result['similarity']*100:.1f}%")
        print(f"  Person 1 Real: {'✓' if verify_result['person1_is_real'] else '✗'}")
        print(f"  Person 2 Real: {'✓' if verify_result['person2_is_real'] else '✗'}")
        print("="*60)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python inference.py <checkpoint_path> <folder_path>")
        sys.exit(1)
    
    checkpoint_path = sys.argv[1]
    folder_path = sys.argv[2]
    
    engine = Face3DInference(checkpoint_path)
    result = engine.predict(folder_path)
    
    print("\n" + "="*60)
    print("RESULT:")
    print(f"  Identity: {result['identity']}")
    print(f"  Confidence: {result['confidence']*100:.2f}%")
    print(f"  Is Real: {'✓ YES' if result['is_real'] else '✗ NO (FAKE)'}")
    print(f"  Spoofing Score: {result['spoofing_score']*100:.1f}%")
    print("="*60)