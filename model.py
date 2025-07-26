import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet18_Weights

class DepthFeatureExtractor(nn.Module):
    def __init__(self, embedding_dim=512):
        super(DepthFeatureExtractor, self).__init__()
        self.backbone = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        # resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.backbone.fc = nn.Linear(512, embedding_dim)
        
    def forward(self, x):
        return self.backbone(x)

class NormalFeatureExtractor(nn.Module):
    def __init__(self, embedding_dim=512):
        super(NormalFeatureExtractor, self).__init__()
        self.backbone = models.resnet18(pretrained=True)
        self.backbone.fc = nn.Linear(512, embedding_dim)
        
    def forward(self, x):
        return self.backbone(x)

class MeshFeatureExtractor(nn.Module):
    def __init__(self, input_dim=3, embedding_dim=512):
        super(MeshFeatureExtractor, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.conv4 = nn.Conv1d(256, 512, 1)
        self.fc = nn.Linear(512, embedding_dim)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        # Global max pooling
        x = torch.max(x, 2)[0]
        x = self.dropout(x)
        x = self.fc(x)
        return x

class Face3DRecognitionModel(nn.Module):
    def __init__(self, num_classes, embedding_dim=512):
        super(Face3DRecognitionModel, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        
        # Feature extractors
        self.depth_extractor = DepthFeatureExtractor(embedding_dim)
        self.normal_extractor = NormalFeatureExtractor(embedding_dim)
        self.mesh_extractor = MeshFeatureExtractor(embedding_dim=embedding_dim)
        
        # Feature fusion
        # self.fusion_layer = nn.Linear(embedding_dim * 3, embedding_dim)
        self.fusion_layer = nn.Sequential(
            nn.Linear(embedding_dim * 3, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.dropout = nn.Dropout(0.5)
        self.bn = nn.BatchNorm1d(embedding_dim)
        
        # Classification head
        self.classifier = nn.Linear(embedding_dim, num_classes)
        
        # Anti-spoofing head
        self.anti_spoofing = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
    def forward(self, inputs):
        features = []
        
        # Extract features from different modalities
        if 'depth' in inputs:
            depth_features = self.depth_extractor(inputs['depth'])
            features.append(depth_features)
        else:
            depth_features = None
            
        if 'normals' in inputs:
            normal_features = self.normal_extractor(inputs['normals'])
            features.append(normal_features)
        else:
            normal_features = None
            
        if 'mesh' in inputs:
            mesh_features = self.mesh_extractor(inputs['mesh'])
            features.append(mesh_features)
        else:
            mesh_features = None
        
        # Fusion
        if len(features) > 0:
            if len(features) == 1:
                fused_features = features[0]
            else:
                concatenated = torch.cat(features, dim=1)
                fused_features = self.fusion_layer(concatenated)
                fused_features = self.bn(fused_features)
                fused_features = self.dropout(fused_features)
        else:
            # Fallback
            batch_size = list(inputs.values())[0].size(0)
            fused_features = torch.zeros(batch_size, self.embedding_dim, device=list(inputs.values())[0].device)
        
        # Classification
        logits = self.classifier(fused_features)
        
        # Anti-spoofing
        spoofing_score = None
        if depth_features is not None and normal_features is not None:
            # Combine depth and normal features for anti-spoofing
            anti_spoofing_input = (depth_features + normal_features) / 2
            spoofing_score = self.anti_spoofing(anti_spoofing_input)
        
        return {
            'embeddings': fused_features,
            'logits': logits,
            'spoofing_score': spoofing_score
        }
    
    def get_embedding(self, inputs):
        """Get feature embedding for inference"""
        with torch.no_grad():
            features = []
            
            if 'depth' in inputs:
                depth_features = self.depth_extractor(inputs['depth'])
                features.append(depth_features)
                
            if 'normals' in inputs:
                normal_features = self.normal_extractor(inputs['normals'])
                features.append(normal_features)
                
            if 'mesh' in inputs:
                mesh_features = self.mesh_extractor(inputs['mesh'])
                features.append(mesh_features)
            
            if len(features) > 0:
                if len(features) == 1:
                    fused_features = features[0]
                else:
                    concatenated = torch.cat(features, dim=1)
                    fused_features = self.fusion_layer(concatenated)
                    fused_features = self.bn(fused_features)
            else:
                batch_size = list(inputs.values())[0].size(0)
                fused_features = torch.zeros(batch_size, self.embedding_dim, device=list(inputs.values())[0].device)
                
            return fused_features

def create_model(num_classes, config):
    """Factory function to create model"""
    model = Face3DRecognitionModel(num_classes, config.EMBEDDING_DIM)
    return model