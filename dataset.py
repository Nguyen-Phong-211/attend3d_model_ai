import os
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import glob
from torchvision import transforms

class Face3DDataset(Dataset):
    def __init__(self, data_root, transform=None, mode='train'):
        self.data_root = data_root
        # self.transform = transforms
        self.transform = transform or transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
        ])
        self.mode = mode
        self.samples = []
        self.labels = []
        
        # Load all data
        self._load_data()
        
    def _load_data(self):
        """Load all samples from dataset folders"""
        print("Loading dataset...")
        
        # Các thư mục dataset
        dataset_folders = ['AFW', 'HELEN', 'IBUG', 'LFPW']
        
        # Collect all subfolders
        all_folders = []
        for dataset_name in dataset_folders:
            dataset_path = os.path.join(self.data_root, dataset_name)
            if os.path.exists(dataset_path):
                subfolders = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) 
                             if os.path.isdir(os.path.join(dataset_path, f))]
                all_folders.extend(subfolders)
        
        # Create label mapping
        # unique_folders = list(set([os.path.basename(f) for f in all_folders]))
        # self.label_map = {folder_name: idx for idx, folder_name in enumerate(unique_folders)}
        # Gán nhãn theo người (ví dụ dùng thư mục cha làm identity)
        person_ids = set()

        for folder_path in all_folders:
            parent_dir = os.path.basename(os.path.dirname(folder_path))
            person_id = parent_dir + "_" + folder_path.split("_")[1]
            person_ids.add(person_id)

        person_ids = sorted(list(person_ids))
        self.label_map = {pid: idx for idx, pid in enumerate(person_ids)}
        
        # Create samples
        for folder_path in tqdm(all_folders, desc="Loading samples"):
            folder_name = os.path.basename(folder_path)
            parent_dir = os.path.basename(os.path.dirname(folder_path))
            person_id = parent_dir + "_" + folder_name.split("_")[1]
                
            # if folder_name in self.label_map:
            #     label = self.label_map[folder_name]
            if person_id in self.label_map:
                label = self.label_map[person_id]
                # Check if required files exist
                depth_path = os.path.join(folder_path, f"{folder_name}_depth.jpg")
                normal_path = os.path.join(folder_path, f"{folder_name}_normals.png")
                obj_path = os.path.join(folder_path, f"{folder_name}.obj")
                
                if os.path.exists(depth_path) and os.path.exists(normal_path):
                    self.samples.append({
                        'folder_path': folder_path,
                        'folder_name': folder_name,
                        'depth_path': depth_path,
                        'normal_path': normal_path,
                        'obj_path': obj_path
                    })
                    self.labels.append(label)
        
        print(f"Loaded {len(self.samples)} samples with {len(self.label_map)} unique classes")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        label = self.labels[idx]
        
        # Load depth image
        depth_img = None
        if os.path.exists(sample['depth_path']):
            depth_img = cv2.imread(sample['depth_path'], cv2.IMREAD_GRAYSCALE)
            if depth_img is not None:
                depth_img = cv2.resize(depth_img, (224, 224))
                # depth_img = depth_img.astype(np.float32) / 255.0
                # depth_img = torch.from_numpy(depth_img).unsqueeze(0)  # (1, 224, 224)
                depth_img = depth_img.astype(np.uint8)  # transform cần kiểu uint8
                depth_img = self.transform(depth_img)  # (1, 224, 224)
        
        # Load normal map
        normal_img = None
        if os.path.exists(sample['normal_path']):
            normal_img = cv2.imread(sample['normal_path'])
            if normal_img is not None:
                normal_img = cv2.cvtColor(normal_img, cv2.COLOR_BGR2RGB)
                normal_img = cv2.resize(normal_img, (224, 224))
                # normal_img = normal_img.astype(np.float32) / 255.0
                # normal_img = torch.from_numpy(normal_img).permute(2, 0, 1)  # (3, 224, 224)
                normal_img = normal_img.astype(np.uint8)
                normal_img = self.transform(normal_img)
        
        # Load mesh vertices (optional)
        mesh_vertices = self._load_mesh_vertices(sample['obj_path'])
        
        # Create input dictionary
        inputs = {}
        if depth_img is not None:
            inputs['depth'] = depth_img
        if normal_img is not None:
            inputs['normals'] = normal_img
        if mesh_vertices is not None:
            inputs['mesh'] = mesh_vertices
            
        return inputs, label
    
    def _load_mesh_vertices(self, obj_path):
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
                # Limit to 1024 vertices and convert to tensor
                vertices = np.array(vertices[:1024], dtype=np.float32)
                vertices = torch.from_numpy(vertices).transpose(0, 1)  # (3, N)
                return vertices
        except Exception as e:
            print(f"Error loading mesh {obj_path}: {e}")
        
        return None

def get_dataloaders(config):
    """Create train and validation dataloaders"""
    dataset = Face3DDataset(config.DATA_ROOT)
    
    # Split into train/val
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=False
    )
    
    return train_loader, val_loader, len(dataset.label_map)