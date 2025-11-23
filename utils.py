import torch
import cv2
import numpy as np
import os
from PIL import Image
from torchvision import transforms

# ============================================================================
# DATA LOADING UTILITIES
# ============================================================================

def load_single_sample(folder_path, config=None):
    """
    Load a single sample for inference
    
    Args:
        folder_path: path to folder containing sample data
        config: optional config object for image size, etc.
        
    Returns:
        dict: {'vis': tensor, 'depth': tensor, 'normals': tensor, 'mesh': tensor}
    
    Cấu trúc:
        folder_path/
            {base_name}_depth.jpg
            {base_name}_normals.png
            {base_name}.obj
        parent_folder/
            {base_name}_vis.jpg  (hoặc _vis_original_size.jpg)
    """
    if config is None:
        # Default config
        class DefaultConfig:
            IMAGE_SIZE = 224
            USE_MESH = True
            MESH_MAX_VERTICES = 1024
        config = DefaultConfig()
    
    # Infer base name from folder name
    folder_name = os.path.basename(folder_path)
    dataset_path = os.path.dirname(folder_path)
    
    # Find base name by checking depth file
    files_in_folder = os.listdir(folder_path)
    depth_candidates = [f for f in files_in_folder if f.endswith('_depth.jpg')]
    
    if len(depth_candidates) > 0:
        base_name = depth_candidates[0].replace('_depth.jpg', '')
    else:
        # Fallback: use folder name
        base_name = folder_name
    
    # Setup transforms
    rgb_transform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    norm_transform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    depth_transform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
    ])
    
    inputs = {}
    
    # Load RGB (vis file ở parent folder)
    vis_path = os.path.join(dataset_path, f"{base_name}_vis.jpg")
    if not os.path.exists(vis_path):
        vis_path = os.path.join(dataset_path, f"{base_name}_vis_original_size.jpg")
    if not os.path.exists(vis_path):
        # Find .png in folder
        vis_path = os.path.join(folder_path, f"{base_name}.png")
    
    if os.path.exists(vis_path):
        try:
            vis_img = Image.open(vis_path).convert('RGB')
            inputs['vis'] = rgb_transform(vis_img).unsqueeze(0)  # (1, 3, H, W)
        except Exception as e:
            print(f"Warning: Cannot load RGB image: {e}")
    
    # Load depth
    depth_path = os.path.join(folder_path, f"{base_name}_depth.jpg")
    if os.path.exists(depth_path):
        try:
            depth_img = Image.open(depth_path).convert('L')
            inputs['depth'] = depth_transform(depth_img).unsqueeze(0)  # (1, 1, H, W)
        except Exception as e:
            print(f"Warning: Cannot load depth: {e}")
    
    # Load normals
    normal_path = os.path.join(folder_path, f"{base_name}_normals.png")
    if os.path.exists(normal_path):
        try:
            normal_img = Image.open(normal_path).convert('RGB')
            inputs['normals'] = norm_transform(normal_img).unsqueeze(0)  # (1, 3, H, W)
        except Exception as e:
            print(f"Warning: Cannot load normals: {e}")
    
    # Load mesh
    if config.USE_MESH:
        obj_path = os.path.join(folder_path, f"{base_name}.obj")
        if os.path.exists(obj_path):
            mesh_vertices = load_mesh_vertices(obj_path, config.MESH_MAX_VERTICES)
            if mesh_vertices is not None:
                inputs['mesh'] = mesh_vertices.unsqueeze(0)  # (1, M, 3)
    
    return inputs


def load_mesh_vertices(obj_path, max_vertices=1024):
    """
    Load vertices from .obj file with normalization
    
    Args:
        obj_path: path to .obj file
        max_vertices: maximum number of vertices
        
    Returns:
        torch.Tensor: (M, 3) where M <= max_vertices
    """
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
        
        if len(vertices) == 0:
            return None
        
        vertices = np.array(vertices, dtype=np.float32)
        
        # Normalize mesh
        centroid = np.mean(vertices, axis=0)
        vertices = vertices - centroid
        max_dist = np.max(np.sqrt(np.sum(vertices ** 2, axis=1)))
        if max_dist < 1e-8:
            max_dist = 1.0
        vertices = vertices / max_dist
        
        # Sample to max_vertices
        if vertices.shape[0] > max_vertices:
            indices = np.random.choice(vertices.shape[0], max_vertices, replace=False)
            vertices = vertices[indices]
        elif vertices.shape[0] < max_vertices:
            # Pad with duplicates
            pad_indices = np.random.choice(vertices.shape[0], 
                                          max_vertices - vertices.shape[0], 
                                          replace=True)
            pad_verts = vertices[pad_indices]
            vertices = np.vstack([vertices, pad_verts])
        
        return torch.from_numpy(vertices).float()  # (M, 3)
        
    except Exception as e:
        print(f"Error loading mesh from {obj_path}: {e}")
        return None


# ============================================================================
# INFERENCE UTILITIES
# ============================================================================

def predict_single(model, folder_path, device='cpu', config=None):
    """
    Predict identity and spoofing for a single sample
    
    Args:
        model: trained Face3DFusionModel
        folder_path: path to sample folder
        device: 'cpu', 'cuda', or 'mps'
        config: config object
        
    Returns:
        dict: {
            'predicted_class': int,
            'confidence': float,
            'is_real': bool,
            'spoofing_score': float,
            'embeddings': tensor (optional)
        }
    """
    model.eval()
    model.to(device)
    
    # Load sample
    inputs = load_single_sample(folder_path, config)
    
    if len(inputs) == 0:
        raise RuntimeError(f"Cannot load any data from {folder_path}")
    
    # Move to device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(inputs)
        
        # Classification
        logits = outputs.get('logits', None)
        if logits is None:
            raise RuntimeError('Model did not return logits')
        
        probabilities = torch.softmax(logits, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        
        # Anti-spoofing
        spoof_logit = outputs.get('spoof_score', None)
        spoofing_score = None
        is_real = True
        
        if spoof_logit is not None:
            spoof_prob = torch.sigmoid(spoof_logit).detach().cpu().squeeze()
            spoofing_score = float(spoof_prob.item() if spoof_prob.dim() == 0 else spoof_prob[0].item())
            
            # Threshold
            threshold = getattr(config, 'SPOOF_THRESHOLD', 0.5) if config else 0.5
            is_real = spoofing_score < threshold
        
        # Embeddings
        embeddings = outputs.get('embeddings', None)
        
        result = {
            'predicted_class': int(predicted.item()),
            'confidence': float(confidence.item()),
            'is_real': bool(is_real),
            'spoofing_score': spoofing_score
        }
        
        if embeddings is not None:
            result['embeddings'] = embeddings.cpu()
        
        return result


def predict_batch(model, folder_paths, device='cpu', config=None, batch_size=8):
    """
    Predict for multiple samples in batches
    
    Args:
        model: trained model
        folder_paths: list of folder paths
        device: device to use
        config: config object
        batch_size: batch size for inference
        
    Returns:
        list of prediction dicts
    """
    model.eval()
    model.to(device)
    
    results = []
    
    for i in range(0, len(folder_paths), batch_size):
        batch_paths = folder_paths[i:i+batch_size]
        batch_results = []
        
        for folder_path in batch_paths:
            try:
                result = predict_single(model, folder_path, device, config)
                batch_results.append(result)
            except Exception as e:
                print(f"Error predicting {folder_path}: {e}")
                batch_results.append(None)
        
        results.extend(batch_results)
    
    return results


def verify_faces(model, folder_path1, folder_path2, device='cpu', config=None, threshold=0.5):
    """
    Verify if two faces belong to the same person
    
    Args:
        model: trained model
        folder_path1, folder_path2: paths to two samples
        device: device
        config: config
        threshold: cosine similarity threshold
        
    Returns:
        dict: {
            'is_same_person': bool,
            'similarity': float,
            'person1_is_real': bool,
            'person2_is_real': bool,
            'person1_spoof_score': float,
            'person2_spoof_score': float
        }
    """
    # Get predictions with embeddings
    result1 = predict_single(model, folder_path1, device, config)
    result2 = predict_single(model, folder_path2, device, config)
    
    if 'embeddings' not in result1 or 'embeddings' not in result2:
        raise RuntimeError("Model must return embeddings for verification")
    
    # Compute cosine similarity
    emb1 = result1['embeddings']
    emb2 = result2['embeddings']
    similarity = torch.cosine_similarity(emb1, emb2).item()
    
    return {
        'is_same_person': similarity >= threshold,
        'similarity': similarity,
        'person1_is_real': result1['is_real'],
        'person2_is_real': result2['is_real'],
        'person1_spoof_score': result1['spoofing_score'],
        'person2_spoof_score': result2['spoofing_score']
    }


# ============================================================================
# MODEL UTILITIES
# ============================================================================

def load_model(checkpoint_path, num_classes, config, device='cpu'):
    """
    Load model from checkpoint
    
    Args:
        checkpoint_path: path to checkpoint file
        num_classes: number of identities
        config: config object
        device: device to load model on
        
    Returns:
        model: loaded model in eval mode
    """
    from model import create_model
    
    # Create model
    model = create_model(num_classes, config)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    print(f"✓ Model loaded from {checkpoint_path}")
    if 'epoch' in checkpoint:
        print(f"  Epoch: {checkpoint['epoch'] + 1}")
    if 'metrics' in checkpoint:
        metrics = checkpoint['metrics']
        if 'cls_acc' in metrics:
            print(f"  Accuracy: {metrics['cls_acc']:.2f}%")
        if 'spoof_metrics' in metrics and metrics['spoof_metrics']:
            print(f"  Spoof AUC: {metrics['spoof_metrics']['auc']:.4f}")
    
    return model


def save_model(model, save_path, epoch=None, metrics=None, config=None):
    """
    Save model checkpoint
    
    Args:
        model: model to save
        save_path: path to save checkpoint
        epoch: current epoch
        metrics: training metrics
        config: config object
    """
    checkpoint = {
        'model': model.state_dict(),
        'epoch': epoch,
        'metrics': metrics
    }
    
    if config is not None:
        checkpoint['config'] = vars(config)
    
    torch.save(checkpoint, save_path)
    print(f"✓ Model saved to {save_path}")


# ============================================================================
# VISUALIZATION UTILITIES
# ============================================================================

def visualize_sample(folder_path, save_path=None):
    """
    Visualize a sample with all modalities
    
    Args:
        folder_path: path to sample folder
        save_path: optional path to save visualization
    """
    import matplotlib.pyplot as plt
    
    folder_name = os.path.basename(folder_path)
    dataset_path = os.path.dirname(folder_path)
    
    # Find base name
    files = os.listdir(folder_path)
    depth_files = [f for f in files if f.endswith('_depth.jpg')]
    base_name = depth_files[0].replace('_depth.jpg', '') if depth_files else folder_name
    
    # Load images
    vis_path = os.path.join(dataset_path, f"{base_name}_vis.jpg")
    if not os.path.exists(vis_path):
        vis_path = os.path.join(dataset_path, f"{base_name}_vis_original_size.jpg")
    
    depth_path = os.path.join(folder_path, f"{base_name}_depth.jpg")
    normals_path = os.path.join(folder_path, f"{base_name}_normals.png")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # RGB
    if os.path.exists(vis_path):
        rgb_img = cv2.imread(vis_path)
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        axes[0].imshow(rgb_img)
        axes[0].set_title('RGB')
        axes[0].axis('off')
    
    # Depth
    if os.path.exists(depth_path):
        depth_img = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
        axes[1].imshow(depth_img, cmap='gray')
        axes[1].set_title('Depth')
        axes[1].axis('off')
    
    # Normals
    if os.path.exists(normals_path):
        normals_img = cv2.imread(normals_path)
        normals_img = cv2.cvtColor(normals_img, cv2.COLOR_BGR2RGB)
        axes[2].imshow(normals_img)
        axes[2].set_title('Normals')
        axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Visualization saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


# ============================================================================
# TESTING UTILITIES
# ============================================================================

def test_data_loading(data_root):
    """Test data loading from a directory"""
    from dataset import Face3DDataset
    from config import config
    
    print("="*60)
    print("Testing data loading...")
    print(f"Data root: {data_root}")
    
    try:
        dataset = Face3DDataset(data_root, config, mode='train')
        
        if len(dataset) > 0:
            print(f"✓ Successfully loaded {len(dataset)} samples")
            print(f"  Identities: {len(dataset.label_map)}")
            
            # Test first sample
            sample, label, is_spoof = dataset[0]
            print(f"\nFirst sample:")
            print(f"  Label: {label}")
            print(f"  Is spoof: {is_spoof.item()}")
            print(f"  Modalities: {list(sample.keys())}")
            for key, value in sample.items():
                print(f"    {key}: {value.shape}")
        else:
            print("✗ No data found!")
            
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("="*60)


def test_model_inference(checkpoint_path, folder_path, config):
    """Test model inference on a single sample"""
    print("="*60)
    print("Testing model inference...")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Sample: {folder_path}")
    
    try:
        # Load model
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        num_classes = checkpoint['model']['arcface.weight'].shape[0]
        
        model = load_model(checkpoint_path, num_classes, config, device='cpu')
        
        # Predict
        result = predict_single(model, folder_path, device='cpu', config=config)
        
        print(f"\nPrediction:")
        print(f"  Identity: {result['predicted_class']}")
        print(f"  Confidence: {result['confidence']*100:.2f}%")
        print(f"  Is Real: {'✓ YES' if result['is_real'] else '✗ NO (FAKE)'}")
        print(f"  Spoofing Score: {result['spoofing_score']*100:.1f}%")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("="*60)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python utils.py test_data <data_root>")
        print("  python utils.py test_inference <checkpoint> <folder_path>")
        print("  python utils.py visualize <folder_path> [save_path]")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == 'test_data':
        data_root = sys.argv[2] if len(sys.argv) > 2 else "/Volumes/WD 500GB EL/DATA_ROOT/REAL"
        test_data_loading(data_root)
    
    elif command == 'test_inference':
        if len(sys.argv) < 4:
            print("Usage: python utils.py test_inference <checkpoint> <folder_path>")
            sys.exit(1)
        
        from config import config
        checkpoint_path = sys.argv[2]
        folder_path = sys.argv[3]
        test_model_inference(checkpoint_path, folder_path, config)
    
    elif command == 'visualize':
        if len(sys.argv) < 3:
            print("Usage: python utils.py visualize <folder_path> [save_path]")
            sys.exit(1)
        
        folder_path = sys.argv[2]
        save_path = sys.argv[3] if len(sys.argv) > 3 else None
        visualize_sample(folder_path, save_path)
    
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)