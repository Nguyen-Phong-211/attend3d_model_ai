import os
import torch
import json
import argparse
from datetime import datetime
from config import config
from dataset import get_dataloaders
from model import create_model
from trainer import Trainer


def main():
    # Parse command line arguments for experiment name
    parser = argparse.ArgumentParser(description='3D Face Recognition Training')
    parser.add_argument('--exp-name', type=str, default=None,
                        help='Experiment name (default: auto-generated from timestamp)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    args = parser.parse_args()
    
    # Generate experiment name
    if args.exp_name is None:
        experiment_name = f"3d_face_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    else:
        experiment_name = args.exp_name
    
    print("="*70)
    print("ATTEND 3D - Face Recognition Training")
    print(f"Experiment: {experiment_name}")
    print(f"Device: {config.DEVICE}")
    print("="*70)

    # Verify data paths
    if hasattr(config, 'DATA_PATH_REAL'):
        if not os.path.exists(config.DATA_PATH_REAL):
            print(f"WARNING: DATA_PATH_REAL not found at {config.DATA_PATH_REAL}")
    elif hasattr(config, 'DATA_ROOT'):
        if not os.path.exists(config.DATA_ROOT):
            raise RuntimeError(f"DATA_ROOT not found: {config.DATA_ROOT}")

    # Load data
    print("\nLoading datasets...")
    train_loader, val_loader, num_classes = get_dataloaders(config)
    print(f"✓ Loaded {num_classes} identities")

    # Save label map
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
            
            # Also save to JSON log directory
            json_log_dir = os.path.join(config.JSON_LOG_DIR, experiment_name)
            os.makedirs(json_log_dir, exist_ok=True)
            label_map_json = os.path.join(json_log_dir, 'label_map.json')
            with open(label_map_json, 'w') as f:
                json.dump(idx_to_class, f, indent=4)
            print(f"✓ Saved label map to {label_map_json}")
            
    except Exception as e:
        print(f"Warning: Could not save label_map. Error: {e}")

    # Create model
    print("\nCreating model...")
    model = create_model(num_classes, config)

    # Create trainer with experiment name
    print("\nInitializing trainer...")
    trainer = Trainer(model, config, experiment_name=experiment_name)
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=config.DEVICE)
        model.load_state_dict(checkpoint['model'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"✓ Resumed from epoch {start_epoch}")
    
    # Train
    print("\n" + "="*70)
    print("Starting training...")
    print("="*70)
    trainer.train(train_loader, val_loader, num_classes)
    
    # Print final summary
    print("\n" + "="*70)
    print("Training completed successfully!")
    print(f"Experiment: {experiment_name}")
    print(f"Logs saved to: {os.path.join(config.JSON_LOG_DIR, experiment_name)}")
    print("="*70)


if __name__ == "__main__":
    main()