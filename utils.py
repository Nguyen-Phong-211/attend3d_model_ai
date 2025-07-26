import torch
import cv2
import numpy as np
import os

def load_single_sample(folder_path):
    """Load a single sample for inference"""
    folder_name = os.path.basename(folder_path)
    
    inputs = {}
    
    # Load depth
    depth_path = os.path.join(folder_path, f"{folder_name}_depth.jpg")
    if os.path.exists(depth_path):
        depth_img = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
        if depth_img is not None:
            depth_img = cv2.resize(depth_img, (224, 224))
            depth_img = depth_img.astype(np.float32) / 255.0
            inputs['depth'] = torch.from_numpy(depth_img).unsqueeze(0).unsqueeze(0)  # (1, 1, 224, 224)
    
    # Load normals
    normal_path = os.path.join(folder_path, f"{folder_name}_normals.png")
    if os.path.exists(normal_path):
        normal_img = cv2.imread(normal_path)
        if normal_img is not None:
            normal_img = cv2.cvtColor(normal_img, cv2.COLOR_BGR2RGB)
            normal_img = cv2.resize(normal_img, (224, 224))
            normal_img = normal_img.astype(np.float32) / 255.0
            inputs['normals'] = torch.from_numpy(normal_img).permute(2, 0, 1).unsqueeze(0)  # (1, 3, 224, 224)
    
    # Load mesh
    obj_path = os.path.join(folder_path, f"{folder_name}.obj")
    if os.path.exists(obj_path):
        mesh_vertices = load_mesh_vertices(obj_path)
        if mesh_vertices is not None:
            inputs['mesh'] = mesh_vertices.unsqueeze(0)  # (1, 3, N)
    
    return inputs

def load_mesh_vertices(obj_path):
    """Load vertices from .obj file"""
    if not os.path.exists(obj_path):
        return None
        
    try:
        vertices = []
        with open(obj_path, 'r') as f:
            for line in f:
                if line.startswith('v '):
                    coords = line.strip().split()[1:]
                    if len(coords) >= 3:
                        vertices.append([float(coord) for coord in coords[:3]])
        
        if len(vertices) > 0:
            # Limit to 1024 vertices
            vertices = np.array(vertices[:1024], dtype=np.float32)
            vertices = torch.from_numpy(vertices).transpose(0, 1)  # (3, N)
            return vertices
    except Exception as e:
        print(f"Error loading mesh: {e}")
    
    return None

def predict_single(model, folder_path, device='cpu'):
    """Predict single sample"""
    model.eval()
    model.to(device)
    
    inputs = load_single_sample(folder_path)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(inputs)
        probabilities = torch.softmax(outputs['logits'], dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        
        # Anti-spoofing check
        is_real = True
        if outputs['spoofing_score'] is not None:
            is_real = outputs['spoofing_score'].item() > 0.5
        
        return {
            'predicted_class': predicted.item(),
            'confidence': confidence.item(),
            'is_real': is_real,
            'spoofing_score': outputs['spoofing_score'].item() if outputs['spoofing_score'] is not None else None
        }

# Test function
 def test_data_loading(data_root="/Volumes/WD 500GB EL/data"):
    """Test data loading"""
    from dataset import Face3DDataset
    
    print("Testing data loading...")
    dataset = Face3DDataset(data_root)
    
    if len(dataset) > 0:
        print(f"Successfully loaded {len(dataset)} samples")
        
        # Test first sample
        inputs, label = dataset[0]
        print(f"First sample label: {label}")
        print(f"First sample inputs: {list(inputs.keys())}")
        for key, value in inputs.items():
            print(f"  {key}: {value.shape}")
    else:
        print("No data found!")

if __name__ == "__main__":
    test_data_loading()