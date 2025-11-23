# model.py - IMPROVED VERSION
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
import math
from torchvision.models import resnet50

# ============================================================================
# ARCFACE HEAD - Improved
# ============================================================================
class ArcFaceHead(nn.Module):
    """
    ArcFace: Additive Angular Margin Loss
    Paper: https://arxiv.org/abs/1801.07698
    
    Improvements:
    - Easy margin for numerical stability
    - Proper handling of edge cases
    """
    def __init__(self, in_features, out_features, s=64.0, m=0.5, easy_margin=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.easy_margin = easy_margin
        
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        # Pre-compute trigonometric values
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)  # threshold for cos(theta)
        self.mm = math.sin(math.pi - m) * m  # margin for easy_margin

    def forward(self, embeddings, labels=None):
        # Normalize features and weights
        cosine = F.linear(F.normalize(embeddings), F.normalize(self.weight))
        
        if labels is None:
            # Inference mode
            return cosine * self.s
        
        # Training mode: add angular margin
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m  # cos(theta + m)
        
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        
        # One-hot encoding
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        
        # Combine
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        
        return output


# ============================================================================
# MESH BRANCH - Improved PointNet++
# ============================================================================
class MeshBranch(nn.Module):
    """
    Improved PointNet with:
    - Spatial dropout for regularization
    - Better feature extraction
    - Multi-scale features
    """
    def __init__(self, in_channels=3, out_dim=512, hidden=[64, 128, 256, 512]):
        super().__init__()
        
        # Point-wise MLPs with batch norm and dropout
        self.conv1 = nn.Conv1d(in_channels, hidden[0], 1)
        self.bn1 = nn.BatchNorm1d(hidden[0])
        self.drop1 = nn.Dropout(0.1)
        
        self.conv2 = nn.Conv1d(hidden[0], hidden[1], 1)
        self.bn2 = nn.BatchNorm1d(hidden[1])
        self.drop2 = nn.Dropout(0.1)
        
        self.conv3 = nn.Conv1d(hidden[1], hidden[2], 1)
        self.bn3 = nn.BatchNorm1d(hidden[2])
        self.drop3 = nn.Dropout(0.1)
        
        self.conv4 = nn.Conv1d(hidden[2], hidden[3], 1)
        self.bn4 = nn.BatchNorm1d(hidden[3])
        
        # Global feature aggregation
        self.fc1 = nn.Linear(hidden[3], out_dim)
        self.bn_fc = nn.BatchNorm1d(out_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Input: (B, M, 3) - batch of point clouds
        Output: (B, out_dim) - global features
        """
        # Handle different input shapes
        if x.dim() == 2:
            x = x.unsqueeze(0)  # (M, 3) -> (1, M, 3)
        
        # Transpose to (B, 3, M) for Conv1d
        if x.size(1) != 3:
            x = x.transpose(1, 2)
        
        # Extract features
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.drop1(x)
        
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.drop2(x)
        
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.drop3(x)
        
        x = self.relu(self.bn4(self.conv4(x)))
        
        # Global max pooling
        x = torch.max(x, dim=2)[0]  # (B, hidden[3])
        
        # Final projection
        x = self.relu(self.bn_fc(self.fc1(x)))
        
        return x

# ============================================================================ 
# SE BLOCK for channel attention
# ============================================================================
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels//reduction, 1),
            nn.ReLU(),
            nn.Conv2d(channels//reduction, channels, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return x * self.fc(x)

# ============================================================================ 
# RGB BACKBONE ENHANCED with SE attention
# ============================================================================
class RGBBackboneEnhanced(nn.Module):
    def __init__(self, out_dim=512, arch='resnet50', pretrained=True):
        super().__init__()
        base = resnet50(pretrained=pretrained)
        self.backbone = nn.Sequential(*list(base.children())[:-2])
        self.se = SEBlock(2048)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(2048, out_dim), nn.BatchNorm1d(out_dim), nn.ReLU())

    def forward(self, x):
        x = self.backbone(x)
        x = self.se(x)          # channel attention
        x = self.pool(x)        # global avg pooling
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# ============================================================================
# DEPTH/NORMALS BACKBONE
# ============================================================================
class SmallBackbone(nn.Module):
    def __init__(self, channels=1, out_dim=512, pretrained=True):
        super().__init__()
        base = models.resnet50(pretrained=pretrained)
        
        # Replace first conv to handle different channels
        if channels != 3 and pretrained:
            with torch.no_grad():
                # average RGB weights if channels=1
                w = base.conv1.weight.mean(dim=1, keepdim=True)
                base.conv1.weight = nn.Parameter(w, requires_grad=True)
        
        self.backbone = nn.Sequential(*list(base.children())[:-1])
        self.attn = SEBlock(2048)
        self.fc = nn.Sequential(nn.Linear(2048, out_dim), nn.BatchNorm1d(out_dim), nn.ReLU())

    def forward(self, x):
        x = self.backbone(x)
        x = self.attn(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# ============================================================================
# ATTENTION MODULE for feature fusion
# ============================================================================
class ModalityAttention(nn.Module):
    def __init__(self, num_modalities, embed_dim=512, num_heads=4):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim, 
                nhead=num_heads, 
                dim_feedforward=embed_dim*2,
                dropout=0.1,
                activation='relu'
            ),
            num_layers=2
        )

    def forward(self, features_list):
        """
        features_list: list of (B, embed_dim)
        """
        x = torch.stack(features_list, dim=0)  # (modalities, B, embed_dim)
        x = self.transformer(x)
        x = x.mean(dim=0)  # fused embedding (B, embed_dim)
        return x

# ============================================================================
# FULL FUSION MODEL
# ============================================================================
class Face3DFusionModel(nn.Module):
    """
    Multi-modal 3D face recognition with anti-spoofing
    
    Features:
    - Multi-modal fusion (RGB + Depth + Normals + Mesh)
    - ArcFace for identity learning
    - Anti-spoofing detection
    - Attention-based fusion
    """
    def __init__(self, num_classes, config):
        super().__init__()
        self.config = config
        self.num_classes = num_classes
        emb = config.EMBEDDING_DIM
        
        # === MODALITY BRANCHES ===
        self.rgb = RGBBackboneEnhanced(
            out_dim=emb, 
            arch=getattr(config, 'RGB_ARCH', 'resnet50'),
            pretrained=True
        )
        
        self.depth = SmallBackbone(
            channels=1, 
            out_dim=emb,
            # arch=getattr(config, 'DEPTH_ARCH', 'resnet18')
        )
        
        self.normals = SmallBackbone(
            channels=3, 
            out_dim=emb,
            # arch=getattr(config, 'NORMAL_ARCH', 'resnet18')
        )
        
        self.mesh = MeshBranch(
            in_channels=3, 
            out_dim=emb
        ) if config.USE_MESH else None
        
        # === FUSION STRATEGY ===
        self.use_attention = getattr(config, 'USE_ATTENTION_FUSION', False)
        num_modalities = 3 + (1 if self.mesh else 0)
        
        if self.use_attention:
            self.attention_fusion = ModalityAttention(num_modalities, emb)
            fuse_dim = emb
        else:
            fuse_dim = emb * num_modalities
        
        # Fusion network
        self.fusion = nn.Sequential(
            nn.Linear(fuse_dim, emb * 4),
            nn.BatchNorm1d(emb * 4),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(emb * 4, emb * 2),
            nn.BatchNorm1d(emb * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(emb * 2, emb),
            nn.BatchNorm1d(emb),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # === TASK HEADS ===
        # ArcFace for identity
        self.arcface = ArcFaceHead(
            in_features=emb, 
            out_features=num_classes, 
            s=config.ARC_FACE_S, 
            m=config.ARC_FACE_M,
            easy_margin=getattr(config, 'ARC_EASY_MARGIN', False)
        )
        
        # Anti-spoofing head (binary classification)
        self.anti_spoof = nn.Sequential(
            nn.Linear(emb, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward(self, inputs, labels=None):
        """
        inputs: dict with keys 'vis', 'depth', 'normals', 'mesh'
        labels: (B,) tensor of identity labels
        
        Returns:
            dict: {
                'embeddings': (B, embed_dim),
                'logits': (B, num_classes),
                'spoof_score': (B, 1)
            }
        """
        device = next(self.parameters()).device
        features = []
        
        # === EXTRACT MODALITY FEATURES ===
        if 'vis' in inputs and inputs['vis'] is not None:
            rgb_feat = self.rgb(inputs['vis'].to(device))
            features.append(rgb_feat)
        
        if 'depth' in inputs and inputs['depth'] is not None:
            depth_feat = self.depth(inputs['depth'].to(device))
            features.append(depth_feat)
        
        if 'normals' in inputs and inputs['normals'] is not None:
            normal_feat = self.normals(inputs['normals'].to(device))
            features.append(normal_feat)
        
        if self.mesh is not None and 'mesh' in inputs and inputs['mesh'] is not None:
            mesh_in = inputs['mesh'].to(device)
            if mesh_in.dim() == 2:
                mesh_in = mesh_in.unsqueeze(0)
            mesh_feat = self.mesh(mesh_in)
            features.append(mesh_feat)
        
        if len(features) == 0:
            raise ValueError("No input modality provided")
        
        # === FEATURE FUSION ===
        if len(features) == 1:
            fused = features[0]
        else:
            if self.use_attention:
                fused = self.attention_fusion(features)
            else:
                fused = torch.cat(features, dim=1)
            fused = self.fusion(fused)
        
        # L2 normalization
        embeddings = F.normalize(fused, p=2, dim=1)
        
        # === TASK OUTPUTS ===
        # Identity classification with ArcFace
        if labels is not None:
            logits = self.arcface(embeddings, labels)
        else:
            logits = self.arcface(embeddings, None)
        
        # Anti-spoofing prediction
        spoof_score = self.anti_spoof(embeddings)
        
        return {
            'embeddings': embeddings,
            'logits': logits,
            'spoof_score': spoof_score
        }


# ============================================================================
# MODEL FACTORY
# ============================================================================
def create_model(num_classes, config):
    """
    Create Face3DFusionModel with given config
    
    Args:
        num_classes: number of identities
        config: configuration object
    
    Returns:
        Face3DFusionModel instance
    """
    model = Face3DFusionModel(num_classes, config)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n{'='*60}")
    print(f"Model Architecture:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Embedding dimension: {config.EMBEDDING_DIM}")
    print(f"  Number of identities: {num_classes}")
    print(f"  ArcFace (s={config.ARC_FACE_S}, m={config.ARC_FACE_M})")
    print(f"  Use mesh: {config.USE_MESH}")
    print(f"  Use attention fusion: {getattr(config, 'USE_ATTENTION_FUSION', False)}")
    print(f"{'='*60}\n")
    
    return model


# ============================================================================
# UTILITIES
# ============================================================================
def load_checkpoint(model, checkpoint_path, device='cpu'):
    """Load model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    return model


def save_checkpoint(model, optimizer, scheduler, epoch, metrics, path):
    """Save model checkpoint"""
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'epoch': epoch,
        'metrics': metrics
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved: {path}")