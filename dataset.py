# dataset.py
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
import numpy as np
from torchvision import transforms

if not hasattr(transforms, "Identity"):
    class Identity(object):
        def __call__(self, x):
            return x
    transforms.Identity = Identity

class Face3DDataset(Dataset):
    """
    Expects data layout:
    DATA_ROOT/
      AFW/
        AFW_134212_1_0/
          AFW_134212_1_0_depth.jpg
          ...
        # Example of a spoof sample
        AFW_134212_1_0_spoof_print/ 
          ...
      HELEN/...
    """
    def __init__(self, data_root, config, samples=None, label_map=None, mode='train'):
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
        self.norm_tf = transforms.Compose([
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
        ])
        self.depth_tf = transforms.Compose([
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.ToTensor(),
        ])

        # self.samples = []
        # self.label_map = {}
        # self._prepare_index()
        if samples is not None and label_map is not None:
            self.samples = samples
            self.label_map = label_map
            print(f"[Reuse index] Found {len(self.samples)} samples, {len(self.label_map)} identities")
        else:
            self.samples = []
            self.label_map = {}
            self._prepare_index()

    def _prepare_index(self):
        candidate_folders = []
        for ds in os.listdir(self.data_root):
            ds_path = os.path.join(self.data_root, ds)
            if not os.path.isdir(ds_path): continue
            for folder in os.listdir(ds_path):
                folder_path = os.path.join(ds_path, folder)
                if os.path.isdir(folder_path):
                    candidate_folders.append(folder_path)

        for folder_path in candidate_folders:
            base = os.path.basename(folder_path)
            depth = os.path.join(folder_path, f"{base}_depth.jpg")
            normals = os.path.join(folder_path, f"{base}_normals.png")
            obj = os.path.join(folder_path, f"{base}.obj")
            vis_candidate1 = os.path.join(os.path.dirname(folder_path), f"{base}_vis.jpg")
            vis_candidate2 = os.path.join(folder_path, f"{base}.png")
            vis = vis_candidate1 if os.path.exists(vis_candidate1) else (vis_candidate2 if os.path.exists(vis_candidate2) else None)
            
            if vis is None or (not os.path.exists(depth) and not os.path.exists(normals) and not os.path.exists(obj)):
                continue

            parent = os.path.basename(os.path.dirname(folder_path))
            # <--- CHANGE: Extract base_id without spoof info for consistent identity
            base_id_parts = base.split('_')
            person_id_num = base_id_parts[1] if len(base_id_parts) > 1 else base
            person_id = f"{parent}__{person_id_num}"
            
            if person_id not in self.label_map:
                self.label_map[person_id] = len(self.label_map)

            # <--- NEW: Determine if the sample is a spoof attack
            # We assume a sample is a spoof if its folder name contains 'spoof'
            is_spoof = 'spoof' in base.lower()

            self.samples.append({
                'folder': folder_path,
                'base': base,
                'vis': vis,
                'depth': depth if os.path.exists(depth) else None,
                'normals': normals if os.path.exists(normals) else None,
                'obj': obj if os.path.exists(obj) else None,
                'label': self.label_map[person_id],
                'is_spoof': float(is_spoof) # Use 0.0 for real, 1.0 for spoof
            })

        print(f"Found {len(self.samples)} samples, {len(self.label_map)} identities")

    def __len__(self):
        return len(self.samples)

    def _load_obj_vertices(self, obj_path):
        if obj_path is None or not os.path.exists(obj_path): return None
        verts = []
        with open(obj_path, 'r') as f:
            for line in f:
                if line.startswith('v '):
                    parts = line.strip().split()[1:]
                    if len(parts) >= 3:
                        verts.append([float(parts[0]), float(parts[1]), float(parts[2])])
        if len(verts) == 0: return None
        
        verts = np.array(verts, dtype=np.float32)

        # === THAY ĐỔI: CHUẨN HÓA MESH ===
        # 1. Dịch chuyển về gốc tọa độ
        centroid = np.mean(verts, axis=0)
        verts = verts - centroid
        # 2. Co giãn vào hình cầu đơn vị
        dist = np.max(np.sqrt(np.sum(verts ** 2, axis=1)))
        verts = verts / dist

        # === THAY ĐỔI: LẤY MẪU NGẪU NHIÊN VÀ ĐỆM ===
        M = self.config.MESH_MAX_VERTICES
        if verts.shape[0] > M:
            # Lấy mẫu ngẫu nhiên nếu số đỉnh nhiều hơn M
            indices = np.random.choice(verts.shape[0], M, replace=False)
            verts = verts[indices]
        elif verts.shape[0] < M:
            # Lặp lại các điểm ngẫu nhiên để đệm nếu số đỉnh ít hơn M
            pad_indices = np.random.choice(verts.shape[0], M - verts.shape[0], replace=True)
            pad_verts = verts[pad_indices]
            verts = np.vstack([verts, pad_verts])
            
        return torch.from_numpy(verts)

    def __getitem__(self, idx):
        s = self.samples[idx]
        try:
            vis_img = Image.open(s['vis']).convert('RGB')
        except (UnidentifiedImageError, OSError):
            print(f"[WARN] Error loading image: {s['vis']} — skipped.")
            return self.__getitem__((idx + 1) % len(self.samples))

        # 1. Tải tất cả dữ liệu đầu vào
        vis = self.rgb_tf(vis_img)

        depth = None
        if s['depth'] is not None:
            try:
                depth = self.depth_tf(Image.open(s['depth']).convert('L'))
            except (UnidentifiedImageError, OSError):
                print(f"[WARN] Error loading depth image: {s['depth']} — skipped.")
                depth = None

        normals = None
        if s['normals'] is not None:
            try:
                normals = self.norm_tf(Image.open(s['normals']).convert('RGB'))
            except (UnidentifiedImageError, OSError):
                print(f"[WARN] Error loading normals image: {s['normals']} — skipped.")
                normals = None
                
        mesh = None
        if self.config.USE_MESH and s['obj'] is not None:
            verts = self._load_obj_vertices(s['obj'])
            if verts is not None:
                # Chỉ cần gán trực tiếp, không transpose
                mesh = verts.float()

        # 2. Đóng gói tất cả vào một dictionary
        sample = {'vis': vis}
        if depth is not None:
            sample['depth'] = depth
        if normals is not None:
            sample['normals'] = normals
        if mesh is not None:
            sample['mesh'] = mesh
            
        # 3. Lấy nhãn
        label = s['label']
        is_spoof = torch.tensor(s['is_spoof'], dtype=torch.float32)

        # 4. Trả về kết quả
        return sample, label, is_spoof

# def get_dataloaders(config, split=0.8):
#     dataset = Face3DDataset(config.DATA_ROOT, config, mode='train')
#     if len(dataset) == 0:
#         raise RuntimeError("No data found.")
    
#     # <--- NEW: Identity-based split to prevent data leakage
#     identity_map = defaultdict(list)
#     for idx, sample in enumerate(dataset.samples):
#         identity_map[sample['label']].append(idx)
        
#     identities = list(identity_map.keys())
#     random.seed(config.SEED)
#     random.shuffle(identities)
    
#     split_idx = int(split * len(identities))
#     train_identities = identities[:split_idx]
#     val_identities = identities[split_idx:]
    
#     train_idx = []
#     for identity in train_identities:
#         train_idx.extend(identity_map[identity])
        
#     val_idx = []
#     for identity in val_identities:
#         val_idx.extend(identity_map[identity])

#     print(f"Splitting {len(identities)} identities: {len(train_identities)} train, {len(val_identities)} val.")
#     print(f"Resulting in {len(train_idx)} train samples, {len(val_idx)} val samples.")

#     from torch.utils.data import Subset
#     # Make sure the dataset instance for validation does not use training augmentations
#     val_dataset = Face3DDataset(config.DATA_ROOT, config, mode='val')
#     train_ds = Subset(dataset, train_idx)
#     val_ds = Subset(val_dataset, val_idx)

#     train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True,
#                           num_workers=config.NUM_WORKERS, pin_memory=True, drop_last=True)
#     val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False,
#                         num_workers=config.NUM_WORKERS, pin_memory=True, drop_last=True)

    
#     return train_loader, val_loader, len(dataset.label_map)
def get_dataloaders(config, split=0.8):
    # build 1 lần để có samples + label_map chuẩn
    base = Face3DDataset(config.DATA_ROOT, config, mode='train')

    from collections import defaultdict
    identity_map = defaultdict(list)
    for idx, sample in enumerate(base.samples):
        identity_map[sample['label']].append(idx)

    rng = random.Random(config.SEED)
    train_idx, val_idx = [], []

    # CHIA THEO ẢNH nhưng giữ nguyên ID giữa train/val
    for ident, idxs in identity_map.items():
        rng.shuffle(idxs)
        cut = int(split * len(idxs))
        # đảm bảo mỗi ID có ít nhất 1 ảnh cho val (nếu bạn muốn)
        cut = max(1, min(cut, len(idxs)-1)) if len(idxs) >= 2 else len(idxs)
        train_idx.extend(idxs[:cut])
        val_idx.extend(idxs[cut:])

    print(f"Image-level split: {len(train_idx)} train samples, {len(val_idx)} val samples over {len(identity_map)} identities.")

    # tạo 2 dataset dùng chung samples/label_map để mapping class đồng nhất
    train_dataset = Face3DDataset(config.DATA_ROOT, config, mode='train',
                                  samples=base.samples, label_map=base.label_map)
    val_dataset   = Face3DDataset(config.DATA_ROOT, config, mode='val',
                                  samples=base.samples, label_map=base.label_map)

    from torch.utils.data import Subset
    train_ds = Subset(train_dataset, train_idx)
    val_ds   = Subset(val_dataset,   val_idx)

    # Với CPU/MPS trên Mac: pin_memory không có tác dụng → đặt theo thiết bị
    use_pin = (config.DEVICE == "cuda")
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True,
                              num_workers=config.NUM_WORKERS, pin_memory=use_pin, drop_last=True)
    # Val có model.eval() nên BatchNorm dùng running stats → để drop_last=False là hợp lý
    val_loader   = DataLoader(val_ds,   batch_size=config.BATCH_SIZE, shuffle=False,
                              num_workers=config.NUM_WORKERS, pin_memory=use_pin, drop_last=False)

    return train_loader, val_loader, len(base.label_map)