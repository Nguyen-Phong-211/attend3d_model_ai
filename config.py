import torch

class Config:
    # Data paths
    DATA_ROOT = "/Volumes/WD 500GB EL/data"
    
    # Model parameters
    IMAGE_SIZE = 224
    DEPTH_CHANNELS = 1
    NORMAL_CHANNELS = 3
    EMBEDDING_DIM = 512
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    EPOCHS = 30
    NUM_WORKERS = 4
    SEED = 42
    
    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Checkpoint
    CHECKPOINT_DIR = "checkpoints"
    CHECKPOINT_PATH = "checkpoints/best_model.pth"
    
    # Logging
    LOG_INTERVAL = 10