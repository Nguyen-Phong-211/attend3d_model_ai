import torch
from config import Config
from dataset import get_dataloaders
from model import create_model
from trainer import Trainer
import os

def main():
    # Configuration
    config = Config()
    
    print("Starting 3D Face Recognition Training")
    print("=" * 50)
    print(f"Data path: {config.DATA_ROOT}")
    print(f"Device: {config.DEVICE}")
    print(f"Batch size: {config.BATCH_SIZE}")
    
    # Check if data path exists
    if not os.path.exists(config.DATA_ROOT):
        print(f"Error: Data path {config.DATA_ROOT} does not exist!")
        return
    
    # Create data loaders
    print("\nLoading data...")
    train_loader, val_loader, num_classes = get_dataloaders(config)
    
    if num_classes == 0:
        print("Error: No data found!")
        return
    
    print(f"Number of classes: {num_classes}")
    
    # Create model
    print("\nCreating model...")
    model = create_model(num_classes, config)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create trainer
    trainer = Trainer(model, config)
    
    # Start training
    trainer.train(train_loader, val_loader, num_classes)
    
    print("\nTraining completed!")

if __name__ == "__main__":
    main()