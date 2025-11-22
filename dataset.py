# dataset.py - FIXED VERSION (Handle mismatched folder/file names)
import os
import glob
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import random
from PIL import Image, UnidentifiedImageError
from collections import defaultdict

if not hasattr(transforms, "Identity"):
    class Identity(object):
        def __call__(self, x):
            return x
    transforms.Identity = Identity

# Farthest Point Sampling cho mesh
def farthest_point_sampling(points, num_samples):
    """
    Lấy mẫu các điểm xa nhất để giữ lại thông tin hình học tốt nhất
    """
    N = points.shape[0]
    if N <= num_samples:
        return np.arange(N)
    
    centroids = np.zeros(num_samples, dtype=np.int32)
    distance = np.ones(N) * 1e10
    farthest = np.random.randint(0, N)
    
    for i in range(num_samples):
        centroids[i] = farthest
        centroid = points[farthest, :]
        dist = np.sum((points - centroid) ** 2, axis=1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance)
    
    return centroids

class Face3DDataset(Dataset):
    """
    Dataset cho nhận diện khuôn mặt 3D + Anti-spoofing
    
    Cấu trúc DATA REAL:
    DATA_ROOT/REAL/
      AFW/
        AFW_134212_1_0/
            AFW_134212_1_0_depth.jpg, 
            AFW_134212_1_0_normals.png, 
            AFW_134212_1_0.obj, 
            AFW_134212_1_0.png
        AFW_134212_1_0_vis.jpg (ngoài folder)

    Cấu trúc DATA FAKE:
    DATA_ROOT/FAKE_RENDER/
        easy_1_1110/
            easy_1_1110_depth.jpg, 
            easy_1_1110_normals.png, 
            easy_1_1110.obj
        easy_1_1110_vis.jpg (ngoài folder)

    DATA_ROOT/render_3d/ (TRƯỜNG HỢP ĐẶC BIỆT!)
        original_000/                    ← Folder name
            frame_000001_depth.jpg       ← File name KHÁC folder
            frame_000001_normals.png
            frame_000001.obj
        frame_000001_vis.jpg (ngoài folder, dùng tên file không phải tên folder!)
    """
    def __init__(self, data_root, config, samples=None, label_map=None, 
                 mode='train', is_fake=False):
        super().__init__()
        self.data_root = data_root
        self.config = config
        self.mode = mode
        self.is_fake = is_fake

        # Transforms với augmentation
        if mode == 'train' and hasattr(config, 'USE_AUGMENTATION') and config.USE_AUGMENTATION:
            self.rgb_tf = transforms.Compose([
                transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
                # transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(getattr(config, 'AUG_ROTATION', 15)),
                transforms.ColorJitter(
                    brightness=getattr(config, 'AUG_COLOR_JITTER', 0.2),
                    contrast=getattr(config, 'AUG_COLOR_JITTER', 0.2),
                    saturation=getattr(config, 'AUG_COLOR_JITTER', 0.2)
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
            ])
        else:
            self.rgb_tf = transforms.Compose([
                transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
                # transforms.RandomHorizontalFlip() if mode=='train' else transforms.Identity(),
                transforms.Identity(),
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

        if samples is not None and label_map is not None:
            self.samples = samples
            self.label_map = label_map
            print(f"[{mode}] Reuse index: {len(self.samples)} samples, {len(self.label_map)} identities")
        else:
            self.samples = []
            self.label_map = {}
            self._prepare_index()

    def _prepare_index(self):
        """Quét và index tất cả samples"""
        candidate_folders = []
        
        # Quét tất cả các thư mục
        for ds in os.listdir(self.data_root):
            ds_path = os.path.join(self.data_root, ds)
            if not os.path.isdir(ds_path): 
                continue
            
            for folder in os.listdir(ds_path):
                folder_path = os.path.join(ds_path, folder)
                if os.path.isdir(folder_path):
                    candidate_folders.append((ds_path, folder_path, folder))
        
        print(f"Scanning {len(candidate_folders)} folders in {self.data_root}...")
        
        # Xử lý từng folder
        for dataset_path, folder_path, folder_name in candidate_folders:
            # BƯỚC 1: Tìm tên file base thực tế BÊN TRONG folder
            # Vì render_3d có folder name khác file name!
            files_in_folder = os.listdir(folder_path)
            
            # Tìm file depth để xác định base name
            depth_candidates = [f for f in files_in_folder if f.endswith('_depth.jpg')]
            
            if len(depth_candidates) == 0:
                # Không có depth file, thử tìm obj file
                obj_candidates = [f for f in files_in_folder if f.endswith('.obj') and not f.endswith('_detail.obj')]
                if len(obj_candidates) == 0:
                    continue
                base_name_from_file = obj_candidates[0].replace('.obj', '')
            else:
                base_name_from_file = depth_candidates[0].replace('_depth.jpg', '')
            
            # BƯỚC 2: Xác định base name để tìm files
            # Ưu tiên dùng tên file thực tế thay vì tên folder
            base = base_name_from_file
            
            # Files TRONG thư mục (dùng base từ file name)
            depth_file = os.path.join(folder_path, f"{base}_depth.jpg")
            normals_file = os.path.join(folder_path, f"{base}_normals.png")
            obj_file = os.path.join(folder_path, f"{base}.obj")
            png_file = os.path.join(folder_path, f"{base}.png")
            
            # File VIS ở NGOÀI thư mục (cũng dùng base từ file name)
            vis_file = os.path.join(dataset_path, f"{base}_vis.jpg")
            
            # Backup: tìm _vis_original_size.jpg
            if not os.path.exists(vis_file):
                vis_file_alt = os.path.join(dataset_path, f"{base}_vis_original_size.jpg")
                if os.path.exists(vis_file_alt):
                    vis_file = vis_file_alt
                else:
                    # Thử tìm .png trong folder
                    if os.path.exists(png_file):
                        vis_file = png_file
                    else:
                        # Cuối cùng, thử dùng folder_name làm base
                        vis_fallback = os.path.join(dataset_path, f"{folder_name}_vis.jpg")
                        if os.path.exists(vis_fallback):
                            vis_file = vis_fallback
                        else:
                            continue  # Skip nếu không có RGB
            
            # Cần ít nhất 1 trong 3: depth, normals, mesh
            has_data = (os.path.exists(depth_file) or 
                       os.path.exists(normals_file) or 
                       os.path.exists(obj_file))
            
            if not has_data:
                continue
            
            # Tạo person_id từ base name (không phải folder name)
            parent = os.path.basename(dataset_path)
            base_id_parts = base.split('_')
            person_id_num = base_id_parts[1] if len(base_id_parts) > 1 else base
            person_id = f"{parent}__{person_id_num}"
            
            if person_id not in self.label_map:
                self.label_map[person_id] = len(self.label_map)
            
            # Xác định spoof
            is_spoof_folder = 'spoof' in base.lower() or 'spoof' in folder_name.lower()
            is_spoof = 1.0 if (self.is_fake or is_spoof_folder) else 0.0
            
            self.samples.append({
                'folder': folder_path,
                'base': base,  # Dùng base từ file name
                'vis': vis_file,
                'depth': depth_file if os.path.exists(depth_file) else None,
                'normals': normals_file if os.path.exists(normals_file) else None,
                'obj': obj_file if os.path.exists(obj_file) else None,
                'label': self.label_map[person_id],
                'is_spoof': is_spoof
            })
        
        print(f"Found {len(self.samples)} samples, {len(self.label_map)} identities")

    def __len__(self):
        return len(self.samples)

    def _load_obj_vertices(self, obj_path):
        """Load và normalize mesh vertices"""
        if obj_path is None or not os.path.exists(obj_path): 
            return None
        
        try:
            verts = []
            with open(obj_path, 'r') as f:
                for line in f:
                    if line.startswith('v '):
                        parts = line.strip().split()[1:]
                        if len(parts) >= 3:
                            verts.append([float(parts[0]), float(parts[1]), float(parts[2])])
            
            if len(verts) == 0: 
                return None
            
            verts = np.array(verts, dtype=np.float32)
            
            # Normalize
            centroid = np.mean(verts, axis=0)
            verts = verts - centroid
            dist = np.max(np.sqrt(np.sum(verts ** 2, axis=1)))
            if dist < 1e-8:
                dist = 1.0
            verts = verts / dist
            
            # Sampling
            M = self.config.MESH_MAX_VERTICES
            if verts.shape[0] > M:
                if hasattr(self.config, 'USE_FPS') and self.config.USE_FPS:
                    indices = farthest_point_sampling(verts, M)
                    verts = verts[indices]
                else:
                    indices = np.random.choice(verts.shape[0], M, replace=False)
                    verts = verts[indices]
            elif verts.shape[0] < M:
                pad_indices = np.random.choice(verts.shape[0], M - verts.shape[0], replace=True)
                pad_verts = verts[pad_indices]
                verts = np.vstack([verts, pad_verts])
            
            return torch.from_numpy(verts)
        
        except Exception as e:
            print(f"Error loading mesh {obj_path}: {e}")
            return None

    def __getitem__(self, idx):
        s = self.samples[idx]
        
        # Load RGB
        try:
            vis_img = Image.open(s['vis']).convert('RGB')
            vis = self.rgb_tf(vis_img)
        except (UnidentifiedImageError, OSError) as e:
            print(f"[WARN] Error loading image: {s['vis']} — skipped. Error: {e}")
            return self.__getitem__((idx + 1) % len(self.samples))

        # Load depth
        depth = None
        if s['depth'] is not None:
            try:
                depth = self.depth_tf(Image.open(s['depth']).convert('L'))
            except (UnidentifiedImageError, OSError):
                pass

        # Load normals
        normals = None
        if s['normals'] is not None:
            try:
                normals = self.norm_tf(Image.open(s['normals']).convert('RGB'))
            except (UnidentifiedImageError, OSError):
                pass
                
        # Load mesh
        mesh = None
        if self.config.USE_MESH and s['obj'] is not None:
            verts = self._load_obj_vertices(s['obj'])
            if verts is not None:
                mesh = verts.float()

        # Đóng gói
        sample = {'vis': vis}
        if depth is not None:
            sample['depth'] = depth
        if normals is not None:
            sample['normals'] = normals
        if mesh is not None:
            sample['mesh'] = mesh
            
        label = s['label']
        is_spoof = torch.tensor(s['is_spoof'], dtype=torch.float32)

        return sample, label, is_spoof


def get_dataloaders(config, split=0.8):
    """
    Tạo dataloaders cho REAL và FAKE data
    """
    print("=" * 60)
    print("Loading datasets...")
    
    if hasattr(config, 'DATA_PATH_REAL') and hasattr(config, 'DATA_PATHS_FAKE'):
        # Chế độ mới: tách REAL và FAKE
        print(f"Loading REAL dataset from: {config.DATA_PATH_REAL}")
        real_dataset = Face3DDataset(config.DATA_PATH_REAL, config, 
                                     mode='train', is_fake=False)
        
        print("\nLoading FAKE datasets...")
        fake_samples_all = []
        fake_label_map = {}
        
        for fake_path in config.DATA_PATHS_FAKE:
            if not os.path.exists(fake_path):
                print(f"Warning: {fake_path} does not exist, skipping...")
                continue
            
            print(f"Loading from {fake_path}...")
            fake_ds = Face3DDataset(fake_path, config, mode='train', is_fake=True)
            
            # Re-map labels để không trùng với REAL
            for sample in fake_ds.samples:
                old_label = sample['label']
                old_id = [k for k, v in fake_ds.label_map.items() if v == old_label][0]
                
                if old_id not in fake_label_map:
                    fake_label_map[old_id] = len(real_dataset.label_map) + len(fake_label_map)
                
                sample['label'] = fake_label_map[old_id]
                fake_samples_all.append(sample)
        
        print(f"Total FAKE samples: {len(fake_samples_all)}")
        
        # Merge REAL + FAKE
        all_samples = real_dataset.samples + fake_samples_all
        all_label_map = {**real_dataset.label_map, **fake_label_map}
        
        print(f"\nTotal: {len(all_samples)} samples, {len(all_label_map)} identities")
        
    else:
        # Chế độ cũ: chỉ có DATA_ROOT
        print(f"Loading from DATA_ROOT: {config.DATA_ROOT}")
        base = Face3DDataset(config.DATA_ROOT, config, mode='train')
        all_samples = base.samples
        all_label_map = base.label_map
    
    print("=" * 60)
    
    # Split theo identity
    identity_map = defaultdict(list)
    for idx, sample in enumerate(all_samples):
        identity_map[sample['label']].append(idx)

    rng = random.Random(config.SEED)
    train_idx, val_idx = [], []

    for ident, idxs in identity_map.items():
        rng.shuffle(idxs)
        cut = int(split * len(idxs))
        cut = max(1, min(cut, len(idxs)-1)) if len(idxs) >= 2 else len(idxs)
        train_idx.extend(idxs[:cut])
        val_idx.extend(idxs[cut:])

    print(f"Split: {len(train_idx)} train samples, {len(val_idx)} val samples")

    # Tạo datasets
    train_dataset = Face3DDataset(
        config.DATA_PATH_REAL if hasattr(config, 'DATA_PATH_REAL') else config.DATA_ROOT, 
        config, mode='train',
        samples=all_samples, label_map=all_label_map
    )
    val_dataset = Face3DDataset(
        config.DATA_PATH_REAL if hasattr(config, 'DATA_PATH_REAL') else config.DATA_ROOT,
        config, mode='val',
        samples=all_samples, label_map=all_label_map
    )

    from torch.utils.data import Subset
    train_ds = Subset(train_dataset, train_idx)
    val_ds = Subset(val_dataset, val_idx)

    use_pin = (config.DEVICE == "cuda")
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True,
                              num_workers=config.NUM_WORKERS, pin_memory=use_pin, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False,
                           num_workers=config.NUM_WORKERS, pin_memory=use_pin, drop_last=False)

    return train_loader, val_loader, len(all_label_map)