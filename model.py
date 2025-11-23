import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
import math
from torch.utils.checkpoint import checkpoint

# ============================================================================
# ARCFACE HEAD
# ============================================================================
class ArcFaceHead(nn.Module):
    """
    ArcFace: Additive Angular Margin Loss
    Paper: https://arxiv.org/abs/1801.07698
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

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, embeddings, labels=None):
        cosine = F.linear(F.normalize(embeddings), F.normalize(self.weight))
        
        if labels is None:
            return cosine * self.s
        
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        
        return output


# ============================================================================
# ANTI-SPOOFING HEAD
# ============================================================================
class ImprovedAntiSpoofHead(nn.Module):
    """
    Enhanced anti-spoofing with three branches:
    1. Multi-scale features
    2. Depth auxiliary task
    3. Feature consistency check
    """
    def __init__(self, embed_dim=512):
        super().__init__()
        
        # Binary classification branch
        self.binary_branch = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(64, 1)
        )
        
        # Depth prediction branch (auxiliary task)
        self.depth_branch = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 32*32)
        )
        
        # Feature consistency branch
        self.consistency_branch = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, embed_dim)
        )
    
    def forward(self, embeddings):
        # Binary prediction
        spoof_score = self.binary_branch(embeddings)
        
        # Depth prediction (for supervision)
        depth_pred = self.depth_branch(embeddings).view(-1, 1, 32, 32)
        
        # Feature consistency
        consistency_feat = self.consistency_branch(embeddings)
        
        return {
            'spoof_score': spoof_score,
            'depth_pred': depth_pred,
            'consistency_feat': consistency_feat
        }


# ============================================================================
# CENTER LOSS (top-1 cluster embeddings better than center loss)
# ============================================================================
class CenterLoss(nn.Module):
    """
    Center Loss implementation for feature clustering around class centers.
    Paper: A Discriminative Feature Learning Approach for Deep Face Recognition
    """
    def __init__(self, num_classes, feat_dim, device):
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.device = device
        
        # Centers for each class
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim).to(device))
    
    def forward(self, embeddings, labels):
        batch_size = embeddings.size(0)
        
        # Calculate distance matrix between embeddings and centers
        distmat = torch.pow(embeddings, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(embeddings, self.centers.t(), beta=1, alpha=-2)
        
        # Select distances corresponding to the correct classes
        classes = torch.arange(self.num_classes).long().to(self.device)
        labels_expand = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels_expand.eq(classes.expand(batch_size, self.num_classes))
        
        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size
        
        return loss


# ============================================================================
# MESH BRANCH
# ============================================================================
class MeshBranch(nn.Module):
    def __init__(self, in_channels=3, out_dim=512, hidden=[64, 128, 256, 512]):
        super().__init__()
        
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
        
        self.fc1 = nn.Linear(hidden[3], out_dim)
        self.bn_fc = nn.BatchNorm1d(out_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(0)
        
        if x.size(1) != 3:
            x = x.transpose(1, 2)
        
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.drop1(x)
        
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.drop2(x)
        
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.drop3(x)
        
        x = self.relu(self.bn4(self.conv4(x)))
        
        x = torch.max(x, dim=2)[0]
        x = self.relu(self.bn_fc(self.fc1(x)))
        
        return x


# ============================================================================
# RGB BACKBONE
# ============================================================================
class RGBBackbone(nn.Module):
    def __init__(self, out_dim=512, arch='resnet50', pretrained=True, use_checkpoint=False):
        super().__init__()
        self.arch = arch
        self.use_checkpoint = use_checkpoint
        
        if arch == 'resnet50':
            base = models.resnet50(pretrained=pretrained)
            self.backbone = nn.Sequential(*list(base.children())[:-1])
            feat_dim = 2048
        elif arch == 'resnet101':
            base = models.resnet101(pretrained=pretrained)
            self.backbone = nn.Sequential(*list(base.children())[:-1])
            feat_dim = 2048
        elif arch == 'resnet34':
            base = models.resnet34(pretrained=pretrained)
            self.backbone = nn.Sequential(*list(base.children())[:-1])
            feat_dim = 512
        else:
            raise ValueError(f"Unknown architecture: {arch}")
        
        self.fc = nn.Sequential(
            nn.Linear(feat_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU()
        )

    def forward(self, x):
        if self.use_checkpoint and self.training:
            x = checkpoint(self.backbone, x, use_reentrant=False)
        else:
            x = self.backbone(x)
        
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# ============================================================================
# DEPTH/NORMALS BACKBONE
# ============================================================================
class SmallBackbone(nn.Module):
    def __init__(self, channels, out_dim=512, arch='resnet18', pretrained=True):
        super().__init__()
        
        if arch == 'resnet18':
            base = models.resnet18(pretrained=pretrained)
            feat_dim = 512
        elif arch == 'resnet34':
            base = models.resnet34(pretrained=pretrained)
            feat_dim = 512
        else:
            raise ValueError(f"Unknown architecture: {arch}")
        
        base.conv1 = nn.Conv2d(channels, 64, kernel_size=7, stride=2, 
                               padding=3, bias=False)
        
        if channels == 1 and pretrained:
            with torch.no_grad():
                orig_weight = models.resnet18(pretrained=True).conv1.weight
                base.conv1.weight[:, 0, :, :] = orig_weight.mean(dim=1)
        
        base.fc = nn.Sequential(
            nn.Linear(feat_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU()
        )
        
        self.model = base

    def forward(self, x):
        return self.model(x)


# ============================================================================
# ATTENTION MODULE
# ============================================================================
class ImprovedModalityAttention(nn.Module):
    """
    Enhanced attention with gating mechanism
    """
    def __init__(self, num_modalities, embed_dim=512):
        super().__init__()
        
        # Attention network
        self.attention = nn.Sequential(
            nn.Linear(embed_dim * num_modalities, num_modalities * 4),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(num_modalities * 4, num_modalities * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(num_modalities * 2, num_modalities),
            nn.Softmax(dim=1)
        )
        
        # Gating mechanism
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * num_modalities, num_modalities),
            nn.Sigmoid()
        )
    
    def forward(self, features_list):
        concat = torch.cat(features_list, dim=1)
        
        # Compute attention weights
        weights = self.attention(concat)
        
        # Compute gating values
        gates = self.gate(concat)
        
        # Apply attention + gating
        weighted_features = []
        for i, feat in enumerate(features_list):
            w = weights[:, i:i+1] * gates[:, i:i+1]
            weighted_features.append(feat * w)
        
        return torch.cat(weighted_features, dim=1), weights


# ============================================================================
# MAIN MODEL
# ============================================================================
class Face3DFusionModel(nn.Module):
    """
    Multi-modal 3D face recognition with improved anti-spoofing
    """
    def __init__(self, num_classes, config):
        super().__init__()
        self.config = config
        self.num_classes = num_classes
        emb = config.EMBEDDING_DIM
        
        use_checkpoint = getattr(config, 'USE_GRADIENT_CHECKPOINT', False)
        
        # === MODALITY BRANCHES ===
        self.rgb = RGBBackbone(
            out_dim=emb, 
            arch=getattr(config, 'RGB_ARCH', 'resnet50'),
            pretrained=True,
            use_checkpoint=use_checkpoint
        )
        
        self.depth = SmallBackbone(
            channels=1, 
            out_dim=emb,
            arch=getattr(config, 'DEPTH_ARCH', 'resnet18')
        )
        
        self.normals = SmallBackbone(
            channels=3, 
            out_dim=emb,
            arch=getattr(config, 'NORMAL_ARCH', 'resnet18')
        )
        
        self.mesh = MeshBranch(
            in_channels=3, 
            out_dim=emb
        ) if config.USE_MESH else None
        
        # === IMPROVED FUSION ===
        self.use_attention = getattr(config, 'USE_ATTENTION_FUSION', True)
        num_modalities = 3 + (1 if self.mesh else 0)
        
        if self.use_attention:
            self.attention_fusion = ImprovedModalityAttention(num_modalities, emb)
            fuse_dim = emb * num_modalities
        else:
            fuse_dim = emb * num_modalities
        
        # Enhanced fusion network
        self.fusion = nn.Sequential(
            nn.Linear(fuse_dim, emb * 2),
            nn.BatchNorm1d(emb * 2),
            nn.ReLU(),
            nn.Dropout(0.4),  # Increase dropout
            
            nn.Linear(emb * 2, emb),
            nn.BatchNorm1d(emb),
            nn.ReLU(),
            nn.Dropout(0.3),  # Increase dropout
            
            nn.Linear(emb, emb),
            nn.BatchNorm1d(emb)
        )
        
        # === TASK HEADS ===
        self.arcface = ArcFaceHead(
            in_features=emb, 
            out_features=num_classes, 
            s=config.ARC_FACE_S, 
            m=config.ARC_FACE_M,
            easy_margin=getattr(config, 'ARC_EASY_MARGIN', False)
        )
        
        # USE ANTI-SPOOF HEAD
        self.anti_spoof = ImprovedAntiSpoofHead(emb)

    def forward(self, inputs, labels=None):
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
                fused, attention_weights = self.attention_fusion(features)
            else:
                fused = torch.cat(features, dim=1)
            fused = self.fusion(fused)
        
        embeddings = F.normalize(fused, p=2, dim=1)
        
        # === TASK OUTPUTS ===
        if labels is not None:
            logits = self.arcface(embeddings, labels)
        else:
            logits = self.arcface(embeddings, None)
        
        # IMPROVED ANTI-SPOOF OUTPUT
        spoof_outputs = self.anti_spoof(embeddings)
        
        return {
            'embeddings': embeddings,
            'logits': logits,
            'spoof_score': spoof_outputs['spoof_score'],
            'depth_pred': spoof_outputs['depth_pred'],
            'consistency_feat': spoof_outputs['consistency_feat']
        }


# ============================================================================
# MODEL FACTORY
# ============================================================================
def create_model(num_classes, config):
    """Create Face3DFusionModel with config"""
    model = Face3DFusionModel(num_classes, config)
    
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
    print(f"  Improved anti-spoofing: YES")
    
    if getattr(config, 'USE_GRADIENT_CHECKPOINT', False):
        print(f"  Gradient checkpointing: ENABLED")
    
    print(f"{'='*60}\n")
    
    return model


def load_checkpoint(model, checkpoint_path, device='cpu'):
    """Load model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
    
    return model


def save_checkpoint(model, optimizer, scheduler, epoch, metrics, path):
    """Save model checkpoint"""
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict() if optimizer else None,
        'scheduler': scheduler.state_dict() if scheduler else None,
        'epoch': epoch,
        'metrics': metrics
    }
    torch.save(checkpoint, path)
    print(f"✓ Checkpoint saved: {path}")