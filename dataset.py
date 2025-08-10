# dataset.py
import os
import glob
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import random

class Face3DDataset(Dataset):
    """
    Expects data layout:
    DATA_ROOT/
      AFW/
        AFW_134212_1_0/
          AFW_134212_1_0_depth.jpg
          AFW_134212_1_0_normals.png
          AFW_134212_1_0.obj
          AFW_134212_1_0.png
        ...
      HELEN/...
    Also allows sibling vis images: AFW_134212_1_0_vis.jpg located next to the folder.
    """
    def __init__(self, data_root, config, mode='train'):
        super().__init__()
        self.data_root = data_root
        self.config = config
        self.mode = mode

        # transforms
        self.rgb_tf = transforms.Compose([
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.RandomHorizontalFlip() if mode=='train' else transforms.Identity(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])
        # normals: treat as RGB
        self.norm_tf = transforms.Compose([
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
        ])
        # depth: single channel, normalize to [0,1]
        self.depth_tf = transforms.Compose([
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.ToTensor(),  # will be (1,H,W)
        ])

        self.samples = []  # list of dicts: {'folder':..., 'name':..., 'vis':..., 'depth':..., 'normals':..., 'obj':...}
        self.label_map = {}
        self._prepare_index()

    def _prepare_index(self):
        # find all sample folders under dataset subfolders
        candidate_folders = []
        for ds in os.listdir(self.data_root):
            ds_path = os.path.join(self.data_root, ds)
            if not os.path.isdir(ds_path): continue
            for folder in os.listdir(ds_path):
                folder_path = os.path.join(ds_path, folder)
                if os.path.isdir(folder_path):
                    candidate_folders.append(folder_path)

        # Build samples
        for folder_path in candidate_folders:
            base = os.path.basename(folder_path)
            depth = os.path.join(folder_path, f"{base}_depth.jpg")
            normals = os.path.join(folder_path, f"{base}_normals.png")
            obj = os.path.join(folder_path, f"{base}.obj")
            # sibling vis images might be next to folder
            vis_candidate1 = os.path.join(os.path.dirname(folder_path), f"{base}_vis.jpg")
            vis_candidate2 = os.path.join(folder_path, f"{base}.png")
            vis = vis_candidate1 if os.path.exists(vis_candidate1) else (vis_candidate2 if os.path.exists(vis_candidate2) else None)

            # require at least vis and depth or normals
            if vis is None:
                continue
            if not os.path.exists(depth) and not os.path.exists(normals) and not os.path.exists(obj):
                continue

            # infer person id: try to use prefix before first underscore (dataset dependent)
            # Better: use parent folder name + ID to make unique identities
            parent = os.path.basename(os.path.dirname(folder_path))
            # use base tokenization with dataset-specific logic if needed
            person_id = f"{parent}__{base.split('_')[1] if '_' in base else base}"

            if person_id not in self.label_map:
                self.label_map[person_id] = len(self.label_map)

            self.samples.append({
                'folder': folder_path,
                'base': base,
                'vis': vis,
                'depth': depth if os.path.exists(depth) else None,
                'normals': normals if os.path.exists(normals) else None,
                'obj': obj if os.path.exists(obj) else None,
                'label': self.label_map[person_id]
            })

        print(f"Found {len(self.samples)} samples, {len(self.label_map)} identities")

    def __len__(self):
        return len(self.samples)

    def _load_obj_vertices(self, obj_path):
        # read v lines, return (N,3) numpy float32
        if obj_path is None or not os.path.exists(obj_path): return None
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
        # sample or pad to M x 3
        M = self.config.MESH_MAX_VERTICES
        if verts.shape[0] >= M:
            idx = np.linspace(0, verts.shape[0]-1, num=M).astype(int)
            verts = verts[idx]
        else:
            pad = np.zeros((M - verts.shape[0], 3), dtype=np.float32)
            verts = np.vstack([verts, pad])
        return torch.from_numpy(verts)  # (M,3)

    def __getitem__(self, idx):
        s = self.samples[idx]
        # load vis (RGB)
        try:
            vis_img = Image.open(s['vis']).convert('RGB')
        except (UnidentifiedImageError, OSError):
            print(f"[WARN] Error loading image: {s['vis']} — skipped.")
            return self.__getitem__((index + 1) % len(self.samples))

        vis = self.rgb_tf(vis_img)

        depth = None
        if s['depth'] is not None and os.path.exists(s['depth']):
            d = Image.open(s['depth']).convert('L')  # single channel
            depth = self.depth_tf(d)  # (1,H,W)

        normals = None
        if s['normals'] is not None and os.path.exists(s['normals']):
            n = Image.open(s['normals']).convert('RGB')
            normals = self.norm_tf(n)

        mesh = None
        if self.config.USE_MESH and s['obj'] is not None and os.path.exists(s['obj']):
            verts = self._load_obj_vertices(s['obj'])
            if verts is not None:
                # return (3, M) for conv1d style
                mesh = verts.transpose(0,1)  # (M,3) -> (3,M)
                mesh = mesh.float()

        sample = {}
        sample['vis'] = vis
        if depth is not None:
            sample['depth'] = depth
        if normals is not None:
            sample['normals'] = normals
        if mesh is not None:
            sample['mesh'] = mesh  # (M,3) as tensor

        label = s['label']
        return sample, label

def get_dataloaders(config, split=0.8):
    dataset = Face3DDataset(config.DATA_ROOT, config, mode='train')
    n = len(dataset)
    if n == 0:
        raise RuntimeError("No data found. Check DATA_ROOT and file layout.")
    idxs = list(range(n))
    random.seed(config.SEED)
    random.shuffle(idxs)
    split_idx = int(split * n)
    train_idx = idxs[:split_idx]
    val_idx = idxs[split_idx:]
    from torch.utils.data import Subset
    train_ds = Subset(dataset, train_idx)
    val_ds = Subset(dataset, val_idx)

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=True)
    return train_loader, val_loader, len(dataset.label_map)