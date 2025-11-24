import json
import os
from datetime import datetime
from pathlib import Path
import numpy as np


class TrainingLogger:
    """
    Comprehensive JSON logger for training metrics
    Saves to:
      - training_log.json (full history)
      - best_metrics.json (best results)
      - experiment_summary.json (final summary)
    """
    
    def __init__(self, log_dir, config=None, experiment_name=None):
        """
        Args:
            log_dir: Directory to save logs
            config: Training config object
            experiment_name: Optional name for this experiment
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Log file paths
        self.training_log_path = self.log_dir / 'training_log.json'
        self.best_metrics_path = self.log_dir / 'best_metrics.json'
        self.summary_path = self.log_dir / 'experiment_summary.json'
        
        # Initialize data structures
        self.training_history = {
            'experiment_info': {
                'name': experiment_name or 'unnamed_experiment',
                'start_time': datetime.now().isoformat(),
                'config': self._serialize_config(config) if config else {}
            },
            'epochs': []
        }
        
        self.best_metrics = {
            'best_train_acc': 0.0,
            'best_val_acc': 0.0,
            'best_spoof_auc': 0.0,
            'best_epoch': 0,
            'best_val_loss': float('inf')
        }
        
        print(f"✓ Logger initialized at: {self.log_dir}")
    
    def _serialize_config(self, config):
        """Convert config object to dict"""
        if config is None:
            return {}
        
        config_dict = {}
        for key in dir(config):
            if not key.startswith('_'):
                value = getattr(config, key)
                # Only serialize basic types
                if isinstance(value, (int, float, str, bool, list)):
                    config_dict[key] = value
                elif isinstance(value, Path):
                    config_dict[key] = str(value)
        return config_dict
    
    def _convert_numpy(self, obj):
        """Convert numpy types to Python native types"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_numpy(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy(item) for item in obj]
        return obj
    
    def log_epoch(self, epoch, train_metrics, val_metrics, lr, duration=None):
        """
        Log metrics for one epoch
        
        Args:
            epoch: Current epoch number
            train_metrics: Dict of training metrics
            val_metrics: Dict of validation metrics
            lr: Current learning rate
            duration: Epoch duration in seconds
        """
        epoch_data = {
            'epoch': epoch + 1,  # 1-indexed for readability
            'timestamp': datetime.now().isoformat(),
            'learning_rate': float(lr),
            'duration_seconds': float(duration) if duration else None,
            'train': self._convert_numpy(train_metrics),
            'validation': self._convert_numpy(val_metrics)
        }
        
        self.training_history['epochs'].append(epoch_data)
        
        # Update best metrics
        train_acc = train_metrics.get('cls_acc', 0)
        val_acc = val_metrics.get('cls_acc', 0)
        val_loss = val_metrics.get('loss', float('inf'))
        
        spoof_auc = 0.0
        if val_metrics.get('spoof_metrics'):
            spoof_auc = val_metrics['spoof_metrics'].get('auc', 0.0)
        
        if train_acc > self.best_metrics['best_train_acc']:
            self.best_metrics['best_train_acc'] = float(train_acc)
        
        if val_acc > self.best_metrics['best_val_acc']:
            self.best_metrics['best_val_acc'] = float(val_acc)
            self.best_metrics['best_epoch'] = epoch + 1
            self.best_metrics['best_epoch_data'] = epoch_data
        
        if spoof_auc > self.best_metrics['best_spoof_auc']:
            self.best_metrics['best_spoof_auc'] = float(spoof_auc)
        
        if val_loss < self.best_metrics['best_val_loss']:
            self.best_metrics['best_val_loss'] = float(val_loss)
        
        # Save after each epoch
        self._save_logs()
    
    def log_early_stop(self, epoch, reason="Patience exceeded"):
        """Log early stopping event"""
        self.training_history['early_stop'] = {
            'epoch': epoch + 1,
            'reason': reason,
            'timestamp': datetime.now().isoformat()
        }
        print(f"⚠️  Early stopping logged: {reason} at epoch {epoch+1}")
    
    def finalize(self, total_epochs_completed):
        """
        Finalize training and create summary
        
        Args:
            total_epochs_completed: Total number of epochs trained
        """
        self.training_history['experiment_info']['end_time'] = datetime.now().isoformat()
        self.training_history['experiment_info']['total_epochs'] = total_epochs_completed
        
        # Create summary
        summary = {
            'experiment_name': self.training_history['experiment_info']['name'],
            'start_time': self.training_history['experiment_info']['start_time'],
            'end_time': self.training_history['experiment_info']['end_time'],
            'total_epochs': total_epochs_completed,
            'best_metrics': self.best_metrics,
            'final_train_acc': self.training_history['epochs'][-1]['train']['cls_acc'] if self.training_history['epochs'] else 0,
            'final_val_acc': self.training_history['epochs'][-1]['validation']['cls_acc'] if self.training_history['epochs'] else 0,
            'config': self.training_history['experiment_info']['config']
        }
        
        # Save summary
        with open(self.summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n{'='*70}")
        print(f"Training Summary Saved:")
        print(f"  Full log: {self.training_log_path}")
        print(f"  Best metrics: {self.best_metrics_path}")
        print(f"  Summary: {self.summary_path}")
        print(f"{'='*70}\n")
        
        self._save_logs()
    
    def _save_logs(self):
        """Save all logs to disk"""
        # Save full training history
        with open(self.training_log_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        # Save best metrics
        with open(self.best_metrics_path, 'w') as f:
            json.dump(self.best_metrics, f, indent=2)
    
    def get_best_metrics(self):
        """Return best metrics dict"""
        return self.best_metrics
    
    def get_history(self):
        """Return full training history"""
        return self.training_history
    
    def print_summary(self):
        """Print a formatted summary of training"""
        print(f"\n{'='*70}")
        print(f"Training Summary:")
        print(f"  Best Train Accuracy: {self.best_metrics['best_train_acc']:.2f}%")
        print(f"  Best Val Accuracy: {self.best_metrics['best_val_acc']:.2f}%")
        print(f"  Best Spoof AUC: {self.best_metrics['best_spoof_auc']:.4f}")
        print(f"  Best Val Loss: {self.best_metrics['best_val_loss']:.4f}")
        print(f"  Best Epoch: {self.best_metrics['best_epoch']}")
        print(f"  Total Epochs: {len(self.training_history['epochs'])}")
        print(f"{'='*70}\n")


class InferenceLogger:
    """
    Logger for inference results
    """
    
    def __init__(self, log_dir, experiment_name=None):
        """
        Args:
            log_dir: Directory to save logs
            experiment_name: Optional name for this inference session
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_path = self.log_dir / f'inference_log_{timestamp}.json'
        
        self.inference_data = {
            'session_info': {
                'name': experiment_name or 'unnamed_session',
                'timestamp': datetime.now().isoformat()
            },
            'results': []
        }
        
        print(f"✓ Inference logger initialized at: {self.log_path}")
    
    def log_prediction(self, sample_path, prediction_result):
        """
        Log a single prediction
        
        Args:
            sample_path: Path to the sample
            prediction_result: Dict with prediction results
        """
        result = {
            'timestamp': datetime.now().isoformat(),
            'sample_path': str(sample_path),
            'prediction': self._convert_to_serializable(prediction_result)
        }
        
        self.inference_data['results'].append(result)
        self._save()
    
    def log_verification(self, path1, path2, verification_result):
        """
        Log a verification result
        
        Args:
            path1, path2: Paths to the two samples
            verification_result: Dict with verification results
        """
        result = {
            'timestamp': datetime.now().isoformat(),
            'type': 'verification',
            'sample1': str(path1),
            'sample2': str(path2),
            'result': self._convert_to_serializable(verification_result)
        }
        
        self.inference_data['results'].append(result)
        self._save()
    
    def _convert_to_serializable(self, obj):
        """Convert numpy/torch types to JSON serializable"""
        import torch
        
        if isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: self._convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_serializable(item) for item in obj]
        return obj
    
    def _save(self):
        """Save inference log to disk"""
        with open(self.log_path, 'w') as f:
            json.dump(self.inference_data, f, indent=2)
    
    def finalize(self):
        """Finalize the inference session"""
        self.inference_data['session_info']['end_time'] = datetime.now().isoformat()
        self.inference_data['session_info']['total_predictions'] = len(self.inference_data['results'])
        self._save()
        
        print(f"\n✓ Inference log saved: {self.log_path}")
        print(f"  Total predictions: {len(self.inference_data['results'])}")


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_training_log(log_path):
    """Load training log from JSON file"""
    with open(log_path, 'r') as f:
        return json.load(f)


def compare_experiments(log_paths):
    """
    Compare multiple experiments
    
    Args:
        log_paths: List of paths to experiment_summary.json files
        
    Returns:
        Comparison dict
    """
    experiments = []
    
    for path in log_paths:
        with open(path, 'r') as f:
            experiments.append(json.load(f))
    
    comparison = {
        'experiments': [
            {
                'name': exp['experiment_name'],
                'best_val_acc': exp['best_metrics']['best_val_acc'],
                'best_spoof_auc': exp['best_metrics']['best_spoof_auc'],
                'total_epochs': exp['total_epochs']
            }
            for exp in experiments
        ]
    }
    
    return comparison


def plot_training_curves(log_path, save_path=None):
    """
    Plot training curves from log file
    
    Args:
        log_path: Path to training_log.json
        save_path: Optional path to save plot
    """
    import matplotlib.pyplot as plt
    
    with open(log_path, 'r') as f:
        data = json.load(f)
    
    epochs = [e['epoch'] for e in data['epochs']]
    train_acc = [e['train']['cls_acc'] for e in data['epochs']]
    val_acc = [e['validation']['cls_acc'] for e in data['epochs']]
    train_loss = [e['train']['loss'] for e in data['epochs']]
    val_loss = [e['validation']['loss'] for e in data['epochs']]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Accuracy
    axes[0, 0].plot(epochs, train_acc, label='Train', marker='o')
    axes[0, 0].plot(epochs, val_acc, label='Validation', marker='s')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy (%)')
    axes[0, 0].set_title('Classification Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Loss
    axes[0, 1].plot(epochs, train_loss, label='Train', marker='o')
    axes[0, 1].plot(epochs, val_loss, label='Validation', marker='s')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Training Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Spoof AUC
    try:
        train_auc = [e['train']['spoof_metrics']['auc'] if e['train'].get('spoof_metrics') else 0 
                     for e in data['epochs']]
        val_auc = [e['validation']['spoof_metrics']['auc'] if e['validation'].get('spoof_metrics') else 0 
                   for e in data['epochs']]
        
        axes[1, 0].plot(epochs, train_auc, label='Train', marker='o')
        axes[1, 0].plot(epochs, val_auc, label='Validation', marker='s')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('AUC')
        axes[1, 0].set_title('Anti-Spoofing AUC')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    except:
        axes[1, 0].text(0.5, 0.5, 'No spoof metrics', ha='center', va='center')
    
    # Learning rate
    try:
        lr = [e['learning_rate'] for e in data['epochs']]
        axes[1, 1].plot(epochs, lr, marker='o', color='green')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True, alpha=0.3)
    except:
        axes[1, 1].text(0.5, 0.5, 'No LR data', ha='center', va='center')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Example: Create a training logger
    from config import config
    
    logger = TrainingLogger(
        log_dir='logs/experiment_1',
        config=config,
        experiment_name='3D_Face_Recognition_v1'
    )
    
    # Simulate some epochs
    for epoch in range(3):
        train_metrics = {
            'loss': 2.5 - epoch * 0.5,
            'cls_acc': 10 + epoch * 5,
            'spoof_metrics': {
                'auc': 0.7 + epoch * 0.05,
                'acc': 0.65 + epoch * 0.05
            }
        }
        
        val_metrics = {
            'loss': 2.8 - epoch * 0.4,
            'cls_acc': 8 + epoch * 4,
            'spoof_metrics': {
                'auc': 0.68 + epoch * 0.06,
                'acc': 0.63 + epoch * 0.06
            }
        }
        
        logger.log_epoch(
            epoch=epoch,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            lr=0.001 * (0.9 ** epoch),
            duration=120.5
        )
    
    logger.finalize(total_epochs_completed=3)
    logger.print_summary()
    
    # Plot curves
    plot_training_curves('logs/experiment_1/training_log.json')