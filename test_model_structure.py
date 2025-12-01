import torch
from model import Face3DFusionModel
from config import config

ckpt = torch.load("/Users/nguyennguyenphong/Documents/study/Final/attend3d_ai/checkpoints/best_model1.pth", map_location="cpu", weights_only=False)

if "num_classes" in ckpt:
    num_classes = ckpt["num_classes"]
else:
    num_classes = config.NUM_CLASSES

model = Face3DFusionModel(num_classes=num_classes, config=config)

if "model" in ckpt:
    state_dict = ckpt["model"]
else:
    state_dict = ckpt

model.load_state_dict(state_dict, strict=False)

print("\n CHECKING MODEL STRUCTURE:")
print("="*60)

has_anti_spoof = hasattr(model, 'anti_spoof')
print(f"Has anti_spoof module: {has_anti_spoof}")

if has_anti_spoof:
    print(f"Anti-spoof type: {type(model.anti_spoof)}")
    print(f"Anti-spoof modules: {list(model.anti_spoof._modules.keys())}")
else:
    print(" MODEL KHÔNG CÓ ANTI-SPOOFING HEAD!")
    print("    Model này không thể phát hiện fake!")

print("\n TEST FORWARD PASS:")
dummy_input = {
    'vis': torch.randn(1, 3, 224, 224),
    'depth': None,
    'normals': None,
    'mesh': None
}

model.eval()

with torch.no_grad():
    output = model(dummy_input)

print(f"Output keys: {output.keys()}")
print(f"Has 'spoof_score': {'spoof_score' in output}")

if 'spoof_score' in output:
    print(f" Spoof score shape: {output['spoof_score'].shape}")
    print(f"   Spoof score value: {output['spoof_score'].item():.4f}")
else:
    print(" OUTPUT KHÔNG CÓ 'spoof_score'!")
    print("   → Model không return spoof predictions!")

print("="*60)

"""
CHECKING MODEL STRUCTURE:
============================================================
Has anti_spoof module: True
Anti-spoof type: <class 'model.ImprovedAntiSpoofHead'>
Anti-spoof modules: ['binary_branch', 'depth_branch', 'consistency_branch']

 TEST FORWARD PASS:
Output keys: dict_keys(['embeddings', 'logits', 'spoof_score', 'depth_pred', 'consistency_feat'])
Has 'spoof_score': True
 Spoof score shape: torch.Size([1, 1])
   Spoof score value: -4.2454
========================================================
"""