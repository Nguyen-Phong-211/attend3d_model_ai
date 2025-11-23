import os
import torch
import json
from config import config
from dataset import get_dataloaders
from model import create_model
from trainer import Trainer

def main():
    print("ATTEND 3D - Training")
    print("Device:", config.DEVICE)

    if hasattr(config, 'DATA_PATH_REAL'):
        if not os.path.exists(config.DATA_PATH_REAL):
             print(f"WARNING: DATA_PATH_REAL not found at {config.DATA_PATH_REAL}")
    elif hasattr(config, 'DATA_ROOT'):
        if not os.path.exists(config.DATA_ROOT):
            raise RuntimeError(f"DATA_ROOT not found: {config.DATA_ROOT}")

    train_loader, val_loader, num_classes = get_dataloaders(config)
    print("Num classes:", num_classes)

    try:
        full_dataset = train_loader.dataset.dataset 
        if hasattr(full_dataset, 'label_map'):
            label_map = full_dataset.label_map
            idx_to_class = {v: k for k, v in label_map.items()}
            
            os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
            save_path = os.path.join(config.CHECKPOINT_DIR, 'label_map.json')
            
            with open(save_path, 'w') as f:
                json.dump(idx_to_class, f, indent=4)
            print(f"✓ Saved label map to {save_path}")
    except Exception as e:
        print(f"Warning: Could not save label_map. Error: {e}")

    # Create model
    model = create_model(num_classes, config)

    # Trainer
    trainer = Trainer(model, config)
    trainer.train(train_loader, val_loader, num_classes)

if __name__ == "__main__":
    main()