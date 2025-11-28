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
    USE_ATTENTION_FUSION = True
    
    # Model architectures
    RGB_ARCH = 'resnet50' # -> resnet50
    DEPTH_ARCH = 'resnet34' # -> resnet34
    NORMAL_ARCH = 'resnet34' # -> resnet34
    
    # ===== TRAINING HYPERPARAMETERS - CRITICAL FIXES =====
    BATCH_SIZE = 48
    LEARNING_RATE = 5e-5
    WEIGHT_DECAY = 5e-5
    EPOCHS = 70
    NUM_WORKERS = 4
    SEED = 42
    
    # ===== OPTIMIZER & SCHEDULER =====
    SCHEDULER = 'plateau'
    WARMUP_EPOCHS = 10
    MIN_LR = 1e-7
    PATIENCE_SCHEDULER = 5
    
    # ===== MIXED PRECISION & GRADIENT =====
    USE_AMP = True
    MAX_GRAD_NORM = 1.0
    USE_GRADIENT_CHECKPOINT = False
    
    # ===== EARLY STOPPING =====
    PATIENCE = 35
    
    # ===== 3D MESH SETTINGS =====
    USE_MESH = True
    MESH_MAX_VERTICES = 1024
    USE_FPS = True
    
    # ===== ARCFACE SETTINGS - MORE RELAXED =====
    ARC_FACE_S = 70
    ARC_FACE_M = 0.35
    ARC_EASY_MARGIN = False
    
    # ===== LOSS WEIGHTS - REBALANCED =====
    SPOOF_LOSS_WEIGHT = 0.05
    
    # CENTER LOSS
    USE_CENTER_LOSS = True
    CENTER_LOSS_WEIGHT = 5e-5
    
    # DEPTH AUXILIARY
    DEPTH_AUX_WEIGHT = 0.03
    
    # ===== AUGMENTATION - LIGHTER =====
    USE_AUGMENTATION = True
    AUG_ROTATION = 5
    AUG_COLOR_JITTER = 0.1
    AUG_RANDOM_BRIGHTNESS = 0.1
    AUG_RANDOM_CONTRAST = 0.1
    AUG_GAUSSIAN_NOISE = 0.005
    AUG_MOTION_BLUR = False
    AUG_CUTOUT_PROB = 0.15
    AUG_CUTOUT_SIZE = 0.12
    
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
    LOG_DIR = "runs/fixed_experiment_v3"
    JSON_LOG_DIR = "logs/experiments"
    LOG_INTERVAL = 20
    SAVE_EVERY = 5
    
    # ===== VALIDATION =====
    TRAIN_SPLIT = 0.8
    VAL_SPLIT = 0.2
    
    # ===== ANTI-SPOOFING METRICS =====
    SPOOF_THRESHOLD = 0.5
    
    # ===== LABEL SMOOTHING =====
    LABEL_SMOOTHING = 0.02

config = Config()