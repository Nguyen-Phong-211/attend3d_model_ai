import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import random
from PIL import Image, ImageFilter
from collections import defaultdict

# CUSTOM AUGMENTATION CLASSES
class GaussianNoise:
    """Add Gaussian noise"""
    def __init__(self, std=0.02):
        self.std = std
    
    def __call__(self, tensor):
        if random.random() > 0.5:
            noise = torch.randn_like(tensor) * self.std
            return tensor + noise
        return tensor

class MotionBlur:
    """Motion blur to images using PIL ImageFilter.GaussianBlur"""
    def __init__(self, kernel_size=5):
        self.kernel_size = kernel_size
    
    def __call__(self, img):
        if random.random() > 0.5:
            return img.filter(ImageFilter.GaussianBlur(radius=random.randint(1, self.kernel_size)))
        return img

if not hasattr(transforms, "Identity"):
    class Identity(object):
        def __call__(self, x):
            return x
    transforms.Identity = Identity

def farthest_point_sampling(points, num_samples):
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
        distance = np.minimum(distance, dist)
        farthest = np.argmax(distance)

    return centroids

class Face3DDataset(Dataset):
    def __init__(self, data_root, config, samples=None, label_map=None, mode='train', is_fake=False):
        super().__init__()
        self.data_root = data_root
        self.config = config
        self.mode = mode
        self.is_fake = is_fake

        # IMPROVED RGB TRANSFORMS WITH AGGRESSIVE AUGMENTATION
        if mode == 'train' and getattr(config, 'USE_AUGMENTATION', False):
            aug_list = [
                transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            ]
            
            # Motion blur
            if getattr(config, 'AUG_MOTION_BLUR', False):
                aug_list.append(MotionBlur(kernel_size=5))
            
            aug_list.extend([
                transforms.RandomRotation(getattr(config, 'AUG_ROTATION', 15)),
                transforms.ColorJitter(
                    brightness=getattr(config, 'AUG_RANDOM_BRIGHTNESS', 0.3),
                    contrast=getattr(config, 'AUG_RANDOM_CONTRAST', 0.3),
                    saturation=getattr(config, 'AUG_COLOR_JITTER', 0.2),
                    hue=0.05
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
            ])
            
            # Gaussian noise
            if getattr(config, 'AUG_GAUSSIAN_NOISE', 0) > 0:
                aug_list.append(GaussianNoise(std=config.AUG_GAUSSIAN_NOISE))
            
            # Random erasing (cutout)
            if getattr(config, 'AUG_CUTOUT_PROB', 0) > 0:
                aug_list.append(
                    transforms.RandomErasing(
                        p=config.AUG_CUTOUT_PROB,
                        scale=(0.02, getattr(config, 'AUG_CUTOUT_SIZE', 0.15)),
                        ratio=(0.3, 3.3),
                        value='random'
                    )
                )
            
            aug_list.append(
                transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
            )
            
            self.rgb_tf = transforms.Compose(aug_list)
            
            print(f"[{mode}] Augmentation enabled:")
            print(f"  - Rotation: ±{config.AUG_ROTATION}°")
            print(f"  - Color jitter: {config.AUG_COLOR_JITTER}")
            print(f"  - Motion blur: {getattr(config, 'AUG_MOTION_BLUR', False)}")
            print(f"  - Gaussian noise: {getattr(config, 'AUG_GAUSSIAN_NOISE', 0)}")
            print(f"  - Random erasing: {getattr(config, 'AUG_CUTOUT_PROB', 0)}")
        else:
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
            transforms.ToTensor()
        ])

        if samples is not None and label_map is not None:
            self.samples = samples
            self.label_map = label_map
            print(f"[{mode}] Using provided samples: {len(samples)} items, {len(label_map)} identities")
            return

        self.samples = []
        self.label_map = {}
        self._prepare_index()

    def _prepare_index(self):
        if not self.data_root:
            print("No data_root provided, skipping folder scan.")
            return

        candidate_folders = []
        
        items = [item for item in os.listdir(self.data_root) 
            if not item.startswith('.') and not item.startswith('_')]

        first_item_path = os.path.join(self.data_root, items[0]) if items else None
        
        if first_item_path and os.path.isdir(first_item_path):
            sub_items = os.listdir(first_item_path)
            has_subdirs = any(os.path.isdir(os.path.join(first_item_path, s)) for s in sub_items)
            
            if has_subdirs:
                print(f"Detected 3-level structure in {self.data_root}")
                for ds in os.listdir(self.data_root):
                    ds_path = os.path.join(self.data_root, ds)
                    if not os.path.isdir(ds_path):
                        continue
                    for folder in os.listdir(ds_path):
                        folder_path = os.path.join(ds_path, folder)
                        if os.path.isdir(folder_path):
                            candidate_folders.append((ds_path, folder_path, folder, ds))
            else:
                print(f"Detected 2-level structure in {self.data_root}")
                dataset_name = os.path.basename(self.data_root)
                for folder in os.listdir(self.data_root):
                    folder_path = os.path.join(self.data_root, folder)
                    if os.path.isdir(folder_path):
                        candidate_folders.append((self.data_root, folder_path, folder, dataset_name))

        print(f"Scanning {len(candidate_folders)} folders in {self.data_root}...")

        for dataset_path, folder_path, folder_name, dataset_name in candidate_folders:
            files = os.listdir(folder_path)
            depth_file = next((f for f in files if f.endswith('_depth.jpg')), None)
            obj_file = next((f for f in files if f.endswith('.obj') and not f.endswith('_detail.obj')), None)

            base_name = None
            if depth_file:
                base_name = depth_file.replace('_depth.jpg','')
            elif obj_file:
                base_name = obj_file.replace('.obj','')
            else:
                continue

            base = base_name
            depth_path = os.path.join(folder_path, f"{base}_depth.jpg")
            normals_path = os.path.join(folder_path, f"{base}_normals.png")
            obj_path = os.path.join(folder_path, f"{base}.obj")
            png_path = os.path.join(folder_path, f"{base}.png")
            
            # 🔧 IMPROVED RGB LOADING LOGIC
            vis_candidates = [
                # 1. Standard vis.jpg outside folder (REAL dataset style)
                os.path.join(dataset_path, f"{base}_vis.jpg"),
                os.path.join(dataset_path, f"{base}_vis_original_size.jpg"),
                
                # 2. PNG inside folder (FAKE_RENDER style)
                png_path,
                
                # 3. Try with folder name (for render_3d mismatch)
                os.path.join(dataset_path, f"{folder_name}_vis.jpg"),
                os.path.join(dataset_path, f"{folder_name}_vis_original_size.jpg"),
                
                # 4. Try to find ANY _vis.jpg in parent folder
                *[os.path.join(dataset_path, f) for f in os.listdir(dataset_path) 
                if f.endswith('_vis.jpg') or f.endswith('_vis_original_size.jpg')],
            ]
            
            vis_path = None
            for candidate in vis_candidates:
                if candidate and os.path.exists(candidate):
                    vis_path = candidate
                    break
            
            if vis_path is None:
                print(f"[WARN] Cannot find RGB for {base} (folder: {folder_name})")
                print(f"  Tried:")
                for c in vis_candidates[:5]:  # Print first 5 attempts
                    print(f"    - {c}")
                continue

            if not any([os.path.exists(p) for p in [depth_path, normals_path, obj_path]]):
                continue

            # Identity extraction logic
            parts = base.split('_')
            if parts[0] in ['easy','hard']:
                person_key = f"{parts[0]}_{parts[1]}"
            elif 'spoof' in parts:
                idx = parts.index('spoof')
                person_key = f"spoof_{parts[idx+1]}" if idx+1<len(parts) else 'spoof_unknown'
            elif parts[0] == 'frame':  # Handle render_3d naming
                person_key = folder_name  # Use folder name as identity
            else:
                person_key = parts[1] if len(parts)>1 else base
            
            person_id = f"{dataset_name}__{person_key}"

            if person_id not in self.label_map:
                self.label_map[person_id] = len(self.label_map)

            is_spoof_flag = 1.0 if self.is_fake or 'spoof' in base.lower() or 'spoof' in folder_name.lower() else 0.0

            self.samples.append({
                'folder': folder_path,
                'base': base,
                'vis': vis_path,
                'depth': depth_path if os.path.exists(depth_path) else None,
                'normals': normals_path if os.path.exists(normals_path) else None,
                'obj': obj_path if os.path.exists(obj_path) else None,
                'label': self.label_map[person_id],
                'subject_id': person_id,
                'is_spoof': is_spoof_flag
            })

        print(f"Found {len(self.samples)} samples, {len(self.label_map)} identities")

    def _load_obj_vertices(self, obj_path):
        """ Load and preprocess vertices from an OBJ file """
        if obj_path is None or not os.path.exists(obj_path):
            return torch.empty(0, 3, dtype=torch.float32)

        try:
            verts = []
            with open(obj_path, 'r') as f:
                for line in f:
                    if line.startswith('v '):
                        parts = line.strip().split()[1:4]
                        verts.append([float(p) for p in parts])

            if not verts:
                return torch.empty(0, 3, dtype=torch.float32)

            verts = np.array(verts, dtype=np.float32)
            verts -= np.mean(verts, axis=0)
            max_dist = np.max(np.sqrt(np.sum(verts**2, axis=1)))
            verts /= max_dist if max_dist > 1e-8 else 1.0

            M = self.config.MESH_MAX_VERTICES
            N = verts.shape[0]
            if N > M:
                if getattr(self.config, 'USE_FPS', False):
                    verts = verts[farthest_point_sampling(verts, M)]
                else:
                    verts = verts[np.random.choice(N, M, replace=False)]
            elif N < M:
                pad_idx = np.random.choice(N, M - N, replace=True)
                verts = np.vstack([verts, verts[pad_idx]])

            return torch.from_numpy(verts).float()

        except Exception as e:
            print(f"Error loading mesh {obj_path}: {e}")
            return torch.empty(0, 3, dtype=torch.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """ Load and return a sample from the dataset """
        s = self.samples[idx]
        try:
            vis_img = Image.open(s['vis']).convert('RGB')
            vis = self.rgb_tf(vis_img)
        except Exception as e:
            print(f"[WARN] Error loading image {s['vis']}: {e}")
            return self.__getitem__((idx+1) % len(self.samples))

        depth = None
        if s['depth'] is not None:
            try:
                depth = self.depth_tf(Image.open(s['depth']).convert('L'))
            except Exception as e:
                depth = None

        normals = None
        if s['normals'] is not None:
            try:
                normals = self.norm_tf(Image.open(s['normals']).convert('RGB'))
            except Exception as e:
                normals = None

        mesh = None
        if getattr(self.config,'USE_MESH', False) and s['obj'] is not None:
            verts = self._load_obj_vertices(s['obj'])
            if verts.numel() > 0:
                mesh = verts

        sample = {'vis': vis}
        if depth is not None:
            sample['depth'] = depth
        if normals is not None:
            sample['normals'] = normals
        if mesh is not None:
            sample['mesh'] = mesh

        return sample, s['label'], torch.tensor(s['is_spoof'], dtype=torch.float32)


def get_dataloaders(config, split=0.8):
    """
    FIXED: Stratified split by SAMPLE-LEVEL for fake data (not identity-level)
    This ensures fake samples appear in both train and validation sets.
    """
    print('='*60)
    print('Loading datasets...')

    real_ds = Face3DDataset(config.DATA_PATH_REAL, config, mode='train', is_fake=False)

    fake_samples, fake_label_map = [], {}
    for fp in config.DATA_PATHS_FAKE:
        fake_ds = Face3DDataset(fp, config, mode='train', is_fake=True)
        for s in fake_ds.samples:
            s['label'] = 0  # ALL fake samples share the same label
            fake_samples.append(s)

    all_samples = real_ds.samples + fake_samples
    # all_label_map = {**real_ds.label_map, **fake_label_map}
    all_label_map = real_ds.label_map  # Only real identities matter for classification

    print(f"Total samples = {len(all_samples)}, Total identities = {len(all_label_map)}")
    
    real_samples = [s for s in all_samples if s['is_spoof'] == 0.0]
    fake_samples_list = [s for s in all_samples if s['is_spoof'] == 1.0]
    
    print(f"  Real samples: {len(real_samples)}")
    print(f"  Fake samples: {len(fake_samples_list)}")
    
    if len(fake_samples_list) == 0:
        print("\n  WARNING: No fake samples found!")
    
    rng = random.Random(config.SEED)
    
    # === REAL: Stratified by identity (unchanged) ===
    real_id_map = defaultdict(list)
    for s in real_samples:
        real_id_map[s['label']].append(s)
    
    train_real, val_real = [], []
    for samples_per_id in real_id_map.values():
        rng.shuffle(samples_per_id)
        cut = max(1, int(split * len(samples_per_id)))
        train_real.extend(samples_per_id[:cut])
        val_real.extend(samples_per_id[cut:])
    
    # === FAKE: SIMPLE RANDOM SPLIT (not by identity!) ===
    # This is the KEY FIX: split fake samples directly, not by identity
    rng.shuffle(fake_samples_list)
    cut_fake = int(split * len(fake_samples_list))
    train_fake = fake_samples_list[:cut_fake]
    val_fake = fake_samples_list[cut_fake:]
    
    # Combine
    train_samples = train_real + train_fake
    val_samples = val_real + val_fake
    
    rng.shuffle(train_samples)
    rng.shuffle(val_samples)
    
    print(f"\nStratified Split (FIXED):")
    print(f"  Train: {len(train_samples)} ({len(train_real)} real + {len(train_fake)} fake)")
    print(f"  Val:   {len(val_samples)} ({len(val_real)} real + {len(val_fake)} fake)")
    
    # Sanity check
    if len(val_fake) == 0:
        print("\nWARNING: Still no fake samples in validation!")
        print("   Forcing at least 10% of fake samples to validation...")
        
        # Force at least 10% fake to validation
        min_val_fake = max(10, int(0.1 * len(fake_samples_list)))
        train_fake = fake_samples_list[:-min_val_fake]
        val_fake = fake_samples_list[-min_val_fake:]
        
        train_samples = train_real + train_fake
        val_samples = val_real + val_fake
        
        rng.shuffle(train_samples)
        rng.shuffle(val_samples)
        
        print(f"  NEW Train: {len(train_samples)} ({len(train_real)} real + {len(train_fake)} fake)")
        print(f"  NEW Val:   {len(val_samples)} ({len(val_real)} real + {len(val_fake)} fake)")
    
    train_ds = Face3DDataset('', config, mode='train', samples=train_samples, label_map=all_label_map)
    val_ds = Face3DDataset('', config, mode='val', samples=val_samples, label_map=all_label_map)

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True, 
                              num_workers=config.NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False, 
                           num_workers=config.NUM_WORKERS, pin_memory=True)

    return train_loader, val_loader, len(all_label_map)