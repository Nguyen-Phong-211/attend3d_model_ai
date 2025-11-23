import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import random
from PIL import Image, UnidentifiedImageError
from collections import defaultdict

# Define Identity transform if not present
if not hasattr(transforms, "Identity"):
    class Identity(object):
        def __call__(self, x):
            return x
    transforms.Identity = Identity

# Farthest Point Sampling (FPS) function for mesh
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

        # RGB transforms
        if mode == 'train' and getattr(config, 'USE_AUGMENTATION', False):
            self.rgb_tf = transforms.Compose([
                transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
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
                transforms.Identity(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
            ])

        # Normals transform
        self.norm_tf = transforms.Compose([
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
        ])

        # Depth transform
        self.depth_tf = transforms.Compose([
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.ToTensor()
        ])

        # Use provided samples if available
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
        for ds in os.listdir(self.data_root):
            ds_path = os.path.join(self.data_root, ds)
            if not os.path.isdir(ds_path):
                continue
            for folder in os.listdir(ds_path):
                folder_path = os.path.join(ds_path, folder)
                if os.path.isdir(folder_path):
                    candidate_folders.append((ds_path, folder_path, folder))

        print(f"Scanning {len(candidate_folders)} folders in {self.data_root}...")

        for dataset_path, folder_path, folder_name in candidate_folders:
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
            vis_path = os.path.join(dataset_path, f"{base}_vis.jpg")

            if not os.path.exists(vis_path):
                alt_vis = os.path.join(dataset_path, f"{base}_vis_original_size.jpg")
                if os.path.exists(alt_vis):
                    vis_path = alt_vis
                elif os.path.exists(png_path):
                    vis_path = png_path
                else:
                    fallback = os.path.join(dataset_path, f"{folder_name}_vis.jpg")
                    if os.path.exists(fallback):
                        vis_path = fallback
                    else:
                        continue

            # Check at least one data exists
            if not any([os.path.exists(p) for p in [depth_path, normals_path, obj_path]]):
                continue

            parent = os.path.basename(dataset_path)
            parts = base.split('_')
            if parts[0] in ['easy','hard']:
                person_key = f"{parts[0]}_{parts[1]}"
            elif 'spoof' in parts:
                idx = parts.index('spoof')
                person_key = f"spoof_{parts[idx+1]}" if idx+1<len(parts) else 'spoof_unknown'
            else:
                person_key = parts[1] if len(parts)>1 else base
            person_id = f"{parent}__{person_key}"

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

            # Sampling / padding
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

            # if verts.shape[0] == 0:
            #     return torch.empty(0, 3, dtype=torch.float32)

            return torch.from_numpy(verts).float()

        except Exception as e:
            print(f"Error loading mesh {obj_path}: {e}")
            return torch.empty(0, 3, dtype=torch.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
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
                print(f"[WARN] Error loading depth {s['depth']}: {e}")
                depth = None

        normals = None
        if s['normals'] is not None:
            try:
                normals = self.norm_tf(Image.open(s['normals']).convert('RGB'))
            except Exception as e:
                print(f"[WARN] Error loading normals {s['normals']}: {e}")
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
    print('='*60)
    print('Loading datasets...')

    real_ds = Face3DDataset(config.DATA_PATH_REAL, config, mode='train', is_fake=False)

    # Load all fake datasets
    fake_samples, fake_label_map = [], {}
    for fp in config.DATA_PATHS_FAKE:
        fake_ds = Face3DDataset(fp, config, mode='train', is_fake=True)
        for s in fake_ds.samples:
            sub_id = s['subject_id']
            if sub_id not in fake_label_map:
                fake_label_map[sub_id] = len(real_ds.label_map)+len(fake_label_map)
            s['label'] = fake_label_map[sub_id]
            fake_samples.append(s)

    all_samples = real_ds.samples + fake_samples
    all_label_map = {**real_ds.label_map, **fake_label_map}

    print(f"Total samples = {len(all_samples)}, Total identities = {len(all_label_map)}")

    # Identity-wise split
    id_map = defaultdict(list)
    for idx, s in enumerate(all_samples):
        id_map[s['label']].append(idx)

    rng = random.Random(config.SEED)
    train_samples, val_samples = [], []
    for idxs in id_map.values():
        rng.shuffle(idxs)
        cut = max(1, int(split*len(idxs))) if len(idxs)>=2 else len(idxs)
        train_samples += [all_samples[i] for i in idxs[:cut]]
        val_samples += [all_samples[i] for i in idxs[cut:]]

    print(f"Train samples = {len(train_samples)}, Val samples = {len(val_samples)}")

    train_ds = Face3DDataset('', config, mode='train', samples=train_samples, label_map=all_label_map)
    val_ds = Face3DDataset('', config, mode='val', samples=val_samples, label_map=all_label_map)

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)

    return train_loader, val_loader, len(all_label_map)