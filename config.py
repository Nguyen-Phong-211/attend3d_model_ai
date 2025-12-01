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
    
    NUM_CLASSES = 374
    
    # Model architectures - BALANCED
    RGB_ARCH = 'resnet50'      # Strong backbone for RGB
    DEPTH_ARCH = 'resnet34'    # Lighter for depth
    NORMAL_ARCH = 'resnet34'   # Lighter for normals
    
    # ===== TRAINING HYPERPARAMETERS - OPTIMIZED =====
    BATCH_SIZE = 32            # GIẢM để model học tốt hơn
    LEARNING_RATE = 1e-4       # TĂNG cho faster convergence
    WEIGHT_DECAY = 1e-4        # TĂNG regularization
    EPOCHS = 100               # TĂNG để model hội tụ đầy đủ
    NUM_WORKERS = 4
    SEED = 42
    
    # ===== OPTIMIZER & SCHEDULER =====
    SCHEDULER = 'plateau'      # Tốt hơn cho validation accuracy
    WARMUP_EPOCHS = 10
    MIN_LR = 1e-7
    PATIENCE_SCHEDULER = 7     # TĂNG patience
    
    # ===== MIXED PRECISION & GRADIENT =====
    USE_AMP = True
    MAX_GRAD_NORM = 1.0
    USE_GRADIENT_CHECKPOINT = False  # Bật nếu thiếu memory
    
    # ===== EARLY STOPPING =====
    PATIENCE = 50              # TĂNG để model có thời gian học
    
    # ===== 3D MESH SETTINGS =====
    USE_MESH = True
    MESH_MAX_VERTICES = 1024
    USE_FPS = True
    
    # ===== ARCFACE SETTINGS - BALANCED =====
    ARC_FACE_S = 64.0          # GIẢM cho easier training
    ARC_FACE_M = 0.3           # GIẢM margin
    ARC_EASY_MARGIN = False
    
    # ===== LOSS WEIGHTS - OPTIMIZED =====
    # Note: Actual weights are in trainer._compute_loss()
    SPOOF_LOSS_WEIGHT = 1.5
    CENTER_LOSS_WEIGHT = 0.001
    CONTRASTIVE_LOSS_WEIGHT = 0.5
    DEPTH_AUX_WEIGHT = 0.1
    
    # CENTER LOSS
    USE_CENTER_LOSS = True
    USE_DEPTH_AUX = True
    
    # ===== AUGMENTATION - MODERATE =====
    USE_AUGMENTATION = True
    AUG_ROTATION = 10          # TĂNG augmentation
    AUG_COLOR_JITTER = 0.2     # TĂNG
    AUG_RANDOM_BRIGHTNESS = 0.2
    AUG_RANDOM_CONTRAST = 0.2
    AUG_GAUSSIAN_NOISE = 0.01  # TĂNG
    AUG_MOTION_BLUR = True     # BẬT motion blur
    AUG_CUTOUT_PROB = 0.2      # TĂNG cutout
    AUG_CUTOUT_SIZE = 0.15
    
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
    LOG_DIR = "runs/improved_experiment"
    JSON_LOG_DIR = "logs/experiments"
    LOG_INTERVAL = 20
    SAVE_EVERY = 5
    
    # ===== VALIDATION =====
    TRAIN_SPLIT = 0.8
    VAL_SPLIT = 0.2
    
    # ===== ANTI-SPOOFING METRICS =====
    SPOOF_THRESHOLD = 0.5
    
    # ===== LABEL SMOOTHING =====
    LABEL_SMOOTHING = 0.1      # TĂNG để model không quá confident

config = Config()