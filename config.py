import torch
import os

class Config:
    
    # ===== DATA PATHS =====
    DATA_PATH_REAL = "/Volumes/WD 500GB EL/DATA_ROOT/REAL"
    DATA_PATHS_FAKE = [
        "/Volumes/WD 500GB EL/DATA_ROOT/FAKE_RENDER",
        "/Volumes/WD 500GB EL/DATA_ROOT/render_3d"
    ]
    
    # ===== MODEL ARCHITECTURE =====
    IMAGE_SIZE = 224
    DEPTH_CHANNELS = 1
    NORMAL_CHANNELS = 3
    EMBEDDING_DIM = 512
    USE_ATTENTION_FUSION = True
    
    # Model architectures
    RGB_ARCH = 'resnet50'
    DEPTH_ARCH = 'resnet18'
    NORMAL_ARCH = 'resnet18'
    
    # ===== TRAINING HYPERPARAMETERS - CRITICAL FIXES =====
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4  # 🔧 FIX 1: Giảm learning rate
    WEIGHT_DECAY = 1e-4
    EPOCHS = 120
    NUM_WORKERS = 4
    SEED = 42
    
    # ===== OPTIMIZER & SCHEDULER =====
    SCHEDULER = 'cosine'
    WARMUP_EPOCHS = 10  # 🔧 FIX 2: Tăng warmup epochs
    MIN_LR = 1e-6
    
    # ===== MIXED PRECISION & GRADIENT =====
    USE_AMP = True
    MAX_GRAD_NORM = 1.0  # 🔧 FIX 3: Giảm gradient clipping
    USE_GRADIENT_CHECKPOINT = False
    
    # ===== EARLY STOPPING =====
    PATIENCE = 25
    
    # ===== 3D MESH SETTINGS =====
    USE_MESH = True
    MESH_MAX_VERTICES = 1024
    USE_FPS = True
    
    # ===== ARCFACE SETTINGS - MORE RELAXED =====
    ARC_FACE_S = 30  # 🔧 FIX 4: Giảm scale factor
    ARC_FACE_M = 0.3   # 🔧 FIX 5: Giảm margin
    ARC_EASY_MARGIN = True
    
    # ===== LOSS WEIGHTS - REBALANCED =====
    SPOOF_LOSS_WEIGHT = 0.1  # 🔧 FIX 6: GIẢM MẠNH spoof weight
    
    # CENTER LOSS
    USE_CENTER_LOSS = False
    CENTER_LOSS_WEIGHT = 0.000001
    
    # DEPTH AUXILIARY
    DEPTH_AUX_WEIGHT = 0.02  # 🔧 FIX 7: Giảm depth weight
    
    # ===== AUGMENTATION - LIGHTER =====
    USE_AUGMENTATION = True
    AUG_ROTATION = 5  # 🔧 FIX 8: Giảm augmentation
    AUG_COLOR_JITTER = 0.1
    AUG_RANDOM_BRIGHTNESS = 0.1
    AUG_RANDOM_CONTRAST = 0.1
    AUG_GAUSSIAN_NOISE = 0.005
    AUG_MOTION_BLUR = False
    AUG_CUTOUT_PROB = 0.1
    AUG_CUTOUT_SIZE = 0.05
    
    # ===== DEVICE =====
    if torch.cuda.is_available():
        DEVICE = "cuda"
    elif torch.backends.mps.is_available():
        DEVICE = "mps"
    else:
        DEVICE = "cpu"
    
    # ===== CHECKPOINTING =====
    CHECKPOINT_DIR = "checkpoints"  # New version
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
    LABEL_SMOOTHING = 0.1  # 🔧 FIX 9: Tăng label smoothing

config = Config()