import torch
import os

class Config:
    
    # ===== DATA PATHS =====
    DATA_PATH_REAL = "/Volumes/WD 500GB EL/DATA_ROOT/REAL"

    DATA_PATHS_FAKE = [
        "/Volumes/WD 500GB EL/DATA_ROOT/FAKE_RENDER/FAKE_DATASET",
        "/Volumes/WD 500GB EL/DATA_ROOT/render_3d/RENDER_DATASET"
    ]
    
    # ===== MODEL ARCHITECTURE =====
    IMAGE_SIZE = 224
    DEPTH_CHANNELS = 1
    NORMAL_CHANNELS = 3
    EMBEDDING_DIM = 512
    
    # IMPROVED: Enable attention fusion
    USE_ATTENTION_FUSION = True
    
    # Model architectures
    RGB_ARCH = 'resnet50'      
    DEPTH_ARCH = 'resnet18'    
    NORMAL_ARCH = 'resnet18'   
    
    # ===== TRAINING HYPERPARAMETERS - IMPROVED =====
    BATCH_SIZE = 8
    LEARNING_RATE = 3e-4
    WEIGHT_DECAY = 5e-4        
    EPOCHS = 120
    NUM_WORKERS = 4
    SEED = 42
    
    # ===== OPTIMIZER & SCHEDULER =====
    SCHEDULER = 'cosine'       
    WARMUP_EPOCHS = 10 
    MIN_LR = 1e-6              
    
    # ===== MIXED PRECISION & GRADIENT =====
    USE_AMP = True             
    MAX_GRAD_NORM = 1.0        
    USE_GRADIENT_CHECKPOINT = False
    
    # ===== EARLY STOPPING =====
    PATIENCE = 20           
    
    # ===== 3D MESH SETTINGS =====
    USE_MESH = True
    MESH_MAX_VERTICES = 1024
    USE_FPS = True             
    
    # ===== ARCFACE SETTINGS =====
    ARC_FACE_S = 64.0          
    ARC_FACE_M = 0.5           
    ARC_EASY_MARGIN = False
    
    # ===== LOSS WEIGHTS - IMPROVED =====
    SPOOF_LOSS_WEIGHT = 1.0    
    
    # NEW: Center Loss
    USE_CENTER_LOSS = True
    CENTER_LOSS_WEIGHT = 0.001
    
    # NEW: Depth Auxiliary Loss
    DEPTH_AUX_WEIGHT = 0.1
    
    # ===== AUGMENTATION - AGGRESSIVE =====
    USE_AUGMENTATION = True
    AUG_ROTATION = 15         
    AUG_COLOR_JITTER = 0.2  
    
    # NEW: Advanced augmentation
    AUG_RANDOM_BRIGHTNESS = 0.3
    AUG_RANDOM_CONTRAST = 0.3
    AUG_GAUSSIAN_NOISE = 0.02
    AUG_MOTION_BLUR = True
    AUG_CUTOUT_PROB = 0.3
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
    LOG_INTERVAL = 20
    SAVE_EVERY = 5             
    
    # ===== VALIDATION =====
    TRAIN_SPLIT = 0.8
    VAL_SPLIT = 0.2
    
    # ===== ANTI-SPOOFING METRICS =====
    SPOOF_THRESHOLD = 0.5      
    
    # ===== LABEL SMOOTHING =====
    LABEL_SMOOTHING = 0.1      

config = Config()