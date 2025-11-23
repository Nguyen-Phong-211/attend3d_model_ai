# config.py
import torch
import os

class Config:
    
    # ===== DATA PATHS =====
    DATA_PATH_REAL = "D://DATA_ROOT//REAL"

    DATA_PATHS_FAKE = [
        "D://DATA_ROOT//render_3d",
        "D://DATA_ROOT//FAKE_RENDER"
    ]

    # ===== MODEL ARCHITECTURE =====
    IMAGE_SIZE = 224
    DEPTH_CHANNELS = 1
    NORMAL_CHANNELS = 3
    EMBEDDING_DIM = 512

    # === FUSION OPTIONS ===
    USE_ATTENTION_FUSION = True

    # === ARCFACE OPTIONS ===
    ARC_EASY_MARGIN = False  

     # === MODEL ARCHITECTURE OPTIONS ===
    RGB_ARCH = 'resnet50'
    DEPTH_ARCH = 'resnet18'
    NORMAL_ARCH = 'resnet18'

    # ===== TRAINING HYPERPARAMETERS =====
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    EPOCHS = 50
    NUM_WORKERS = 8
    SEED = 42

    # ===== 3D MESH SETTINGS =====
    USE_MESH = True
    MESH_MAX_VERTICES = 1024
    USE_FPS = True

    # ===== ARCFACE SETTINGS =====
    ARC_FACE_S = 64.0
    ARC_FACE_M = 0.5

    # ===== LOSS WEIGHTS =====
    SPOOF_LOSS_WEIGHT = 1.0

    # ===== AUGMENTATION (cho anti-spoofing) =====
    USE_AUGMENTATION = True
    AUG_ROTATION = 15  # degrees
    AUG_COLOR_JITTER = 0.2
    AUG_GAUSSIAN_BLUR = True

    # ===== DEVICE =====
    if torch.cuda.is_available():
        DEVICE = "cuda"
    elif torch.backends.mps.is_available():
        DEVICE = "mps"
    else:
        DEVICE = "cpu"

    # ===== CHECKPOINTING =====
    CHECKPOINT_DIR = "checkpoints"
    CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pth")

   # ===== LOGGING =====
    LOG_INTERVAL = 20
    SAVE_EVERY = 1

    # ===== VALIDATION =====
    TRAIN_SPLIT = 0.8
    VAL_SPLIT = 0.2
    
    # ===== ANTI-SPOOFING METRICS =====
    SPOOF_THRESHOLD = 0.5

config = Config()