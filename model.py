# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np

# ArcFace / AddMargin (simple implementation)
class ArcFaceHead(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.3):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.s = s
        self.m = m

        # Ensure m is a tensor
        # m_tensor = torch.tensor(m, dtype=torch.float32)

        # Register as buffers so they move with the model
        # self.register_buffer("cos_m", torch.cos(m_tensor))
        # self.register_buffer("sin_m", torch.sin(m_tensor))
        # self.register_buffer("th", torch.cos(torch.pi - m_tensor))
        # self.register_buffer("mm", torch.sin(torch.pi - m_tensor) * m_tensor)
        self.register_buffer("cos_m", torch.tensor(np.cos(m)))
        self.register_buffer("sin_m", torch.tensor(np.sin(m)))
        self.register_buffer("th", torch.tensor(np.cos(np.pi - m)))
        self.register_buffer("mm", torch.tensor(np.sin(np.pi - m) * m))

    def forward(self, embeddings, labels=None):
        normalized_emb = F.normalize(embeddings, dim=1)
        normalized_w = F.normalize(self.weight, dim=1)
        cos_theta = F.linear(normalized_emb, normalized_w)

        if labels is None:
            return cos_theta * self.s

        # === THAY ĐỔI: SỬ DỤNG CÔNG THỨC LƯỢNG GIÁC ===
        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)
        
        sin_theta = torch.sqrt(1.0 - torch.pow(cos_theta, 2) + 1e-7)
        # Công thức cos(theta + m) = cos(theta)cos(m) - sin(theta)sin(m)
        cos_theta_m = cos_theta * self.cos_m - sin_theta * self.sin_m
        
        # Xử lý trường hợp theta + m > pi
        cond = cos_theta > self.th
        cos_theta_m = torch.where(cond, cos_theta_m, cos_theta - self.mm)

        logits = (one_hot * cos_theta_m) + ((1.0 - one_hot) * cos_theta)
        logits *= self.s
        return logits

# Simple PointNet-like mesh branch
class MeshBranch(nn.Module):
    def __init__(self, in_channels=3, out_dim=512, hidden=[64,128,256,512]):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, hidden[0], 1)
        self.conv2 = nn.Conv1d(hidden[0], hidden[1], 1)
        self.conv3 = nn.Conv1d(hidden[1], hidden[2], 1)
        self.conv4 = nn.Conv1d(hidden[2], hidden[3], 1)
        self.bn1 = nn.BatchNorm1d(hidden[0])
        self.bn2 = nn.BatchNorm1d(hidden[1])
        self.bn3 = nn.BatchNorm1d(hidden[2])
        self.bn4 = nn.BatchNorm1d(hidden[3])
        self.relu = nn.ReLU()
        self.fc = nn.Linear(hidden[3], out_dim)

    def forward(self, x):
        # x: (B, M, 3) OR (M,3), we expect (B, M, 3) -> transpose
        if x.dim() == 3 and x.size(1) != 3:
            # assume (B, M, 3)
            x = x.transpose(1,2)  # to (B, 3, M)
        elif x.dim() == 2:
            # (M,3) -> (1,3,M)
            x = x.unsqueeze(0).transpose(1,2)
        elif x.dim() == 3 and x.size(1) == 3:
            # (B,3,M) ok
            pass
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.relu(self.bn3(self.conv3(out)))
        out = self.relu(self.bn4(self.conv4(out)))
        # global max pool
        out = torch.max(out, dim=2)[0]
        out = self.fc(out)
        return out

# RGB backbone (ResNet50)
class RGBBackbone(nn.Module):
    def __init__(self, out_dim=512):
        super().__init__()
        res = models.resnet50(pretrained=True)
        # remove fc
        self.backbone = nn.Sequential(*(list(res.children())[:-1]))  # output (B,2048,1,1)
        self.fc = nn.Linear(2048, out_dim)

    def forward(self, x):
        x = self.backbone(x)  # (B,2048,1,1)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Depth / Normals backbone (ResNet18 but adapt channels)
def make_resnet18_input_channels(channels, pretrained=True):
    res = models.resnet18(pretrained=pretrained)
    # replace conv1
    res.conv1 = nn.Conv2d(channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    return res

class SmallBackbone(nn.Module):
    def __init__(self, channels, out_dim=512):
        super().__init__()
        res = make_resnet18_input_channels(channels, pretrained=True)
        res.fc = nn.Linear(512, out_dim)
        self.model = res

    def forward(self, x):
        return self.model(x)

# Full fusion model
class Face3DFusionModel(nn.Module):
    def __init__(self, num_classes, config):
        super().__init__()
        self.config = config
        self.num_classes = num_classes
        emb = config.EMBEDDING_DIM
        # branches
        self.rgb = RGBBackbone(out_dim=emb)
        self.depth = SmallBackbone(channels=1, out_dim=emb)
        self.normals = SmallBackbone(channels=3, out_dim=emb)
        self.mesh = MeshBranch(in_channels=3, out_dim=emb) if config.USE_MESH else None

        # fusion -> final embedding
        fuse_dim = emb * (3 + (1 if self.mesh else 0))
        self.fusion = nn.Sequential(
            nn.Linear(fuse_dim, emb),
            nn.BatchNorm1d(emb),
            nn.ReLU(),
            nn.Dropout(0.4)
        )

        # ArcFace head
        self.arcface = ArcFaceHead(in_features=emb, out_features=num_classes, s=config.ARC_FACE_S, m=config.ARC_FACE_M)

        # anti-spoofing head (binary)
        self.anti_spf = nn.Sequential(
            nn.Linear(emb, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            # nn.Sigmoid()
        )

    def forward(self, inputs, labels=None):
        # inputs: dict may contain 'vis', 'depth', 'normals', 'mesh' tensors
        feats = []
        device = next(self.parameters()).device
        # RGB
        if 'vis' in inputs and inputs['vis'] is not None:
            rgb_f = self.rgb(inputs['vis'].to(device))
            feats.append(rgb_f)
        # depth
        if 'depth' in inputs and inputs['depth'] is not None:
            depth_f = self.depth(inputs['depth'].to(device))
            feats.append(depth_f)
        # normals
        if 'normals' in inputs and inputs['normals'] is not None:
            norm_f = self.normals(inputs['normals'].to(device))
            feats.append(norm_f)
        # mesh
        if self.mesh is not None and 'mesh' in inputs and inputs['mesh'] is not None:
            # mesh expected (B, M, 3) or (M,3)
            mesh_in = inputs['mesh']
            if mesh_in.dim() == 2:
                mesh_in = mesh_in.unsqueeze(0).to(device)
            else:
                mesh_in = mesh_in.to(device)
            mesh_f = self.mesh(mesh_in)
            feats.append(mesh_f)

        if len(feats) == 0:
            raise ValueError("No input modality provided")

        if len(feats) == 1:
            fused = feats[0]
        else:
            fused = torch.cat(feats, dim=1)
            fused = self.fusion(fused)

        embeddings = F.normalize(fused, dim=1)

        # logits: ArcFace head expects labels optionally
        logits = self.arcface(embeddings, labels) if labels is not None else self.arcface(embeddings, None)

        # anti-spoofing score
        spoof_score = self.anti_spf(embeddings)

        return {
            'embeddings': embeddings,
            'logits': logits,
            'spoof_score': spoof_score
        }

def create_model(num_classes, config):
    model = Face3DFusionModel(num_classes, config)
    return model