# config.py
import torch
import os

class Config:
    # Data paths
    DATA_ROOT = "/Volumes/WD 500GB EL/data"

    # Model / data
    IMAGE_SIZE = 224
    DEPTH_CHANNELS = 1
    NORMAL_CHANNELS = 3
    EMBEDDING_DIM = 512
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    EPOCHS = 60
    NUM_WORKERS = 8
    SEED = 42
    USE_MESH = True
    MESH_MAX_VERTICES = 1024

    # ArcFace margin (for high-accuracy face recognition)
    ARC_FACE_S = 30.0
    ARC_FACE_M = 0.3

    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Checkpoint
    CHECKPOINT_DIR = "checkpoints"
    CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pth")

    # Logging
    LOG_INTERVAL = 20

    # Misc
    SAVE_EVERY = 1

config = Config()