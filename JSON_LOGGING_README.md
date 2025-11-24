# 📊 JSON Logging System - Usage Guide

## 📁 File Structure

```
project/
├── logger.py              # JSON logging classes
├── trainer.py            # Updated trainer with JSON logging
├── main.py               # Updated main script
├── view_logs.py          # Log analysis script
├── config.py             # Config with JSON_LOG_DIR
└── logs/
    └── experiments/
        └── <experiment_name>/
            ├── training_log.json        # Full training history
            ├── best_metrics.json        # Best metrics only
            ├── experiment_summary.json  # Final summary
            ├── label_map.json          # Identity mapping
            └── training_curves.png     # Visualization (if plotted)
```

## 🚀 Quick Start

### 1. Training with Auto-logging

```bash
# Basic training (auto-generated experiment name)
python main.py

# Training with custom experiment name
python main.py --exp-name my_first_experiment

# Resume from checkpoint
python main.py --exp-name resume_exp --resume checkpoints_v2/best_acc.pth
```

### 2. View Training Logs

```bash
# List all experiments
python view_logs.py --list

# View specific experiment details
python view_logs.py --exp my_first_experiment

# Compare multiple experiments
python view_logs.py --compare exp1 exp2 exp3

# Plot training curves
python view_logs.py --plot my_first_experiment

# Export to CSV
python view_logs.py --export-csv my_first_experiment
```

## 📄 JSON File Formats

### training_log.json
Complete training history with all epochs:

```json
{
  "experiment_info": {
    "name": "3d_face_20250524_143022",
    "start_time": "2025-05-24T14:30:22.123456",
    "end_time": "2025-05-24T18:45:33.654321",
    "total_epochs": 50,
    "config": {
      "BATCH_SIZE": 16,
      "LEARNING_RATE": 0.001,
      "DEVICE": "cuda",
      ...
    }
  },
  "epochs": [
    {
      "epoch": 1,
      "timestamp": "2025-05-24T14:35:10.123456",
      "learning_rate": 0.0002,
      "duration_seconds": 285.5,
      "train": {
        "loss": 8.234,
        "cls_acc": 5.23,
        "cls_loss": 8.120,
        "spoof_loss": 0.089,
        "center_loss": 0.0,
        "depth_loss": 0.025,
        "spoof_metrics": {
          "auc": 0.8567,
          "acc": 0.7823,
          "apcer": 0.2891,
          "bpcer": 0.1286,
          "eer": 0.2089,
          "f1": 0.7234,
          "tp": 1234,
          "tn": 2345,
          "fp": 123,
          "fn": 234
        }
      },
      "validation": {
        "loss": 8.456,
        "cls_acc": 4.89,
        "cls_f1": 0.0234,
        "spoof_metrics": {
          "auc": 0.8823,
          ...
        }
      }
    },
    ...
  ]
}
```

### best_metrics.json
Only the best results:

```json
{
  "best_train_acc": 45.67,
  "best_val_acc": 42.34,
  "best_spoof_auc": 0.9512,
  "best_epoch": 38,
  "best_val_loss": 2.1234,
  "best_epoch_data": {
    "epoch": 38,
    ...
  }
}
```

### experiment_summary.json
High-level summary:

```json
{
  "experiment_name": "3d_face_20250524_143022",
  "start_time": "2025-05-24T14:30:22.123456",
  "end_time": "2025-05-24T18:45:33.654321",
  "total_epochs": 50,
  "best_metrics": { ... },
  "final_train_acc": 43.21,
  "final_val_acc": 40.56,
  "config": { ... }
}
```

## 🔍 Programmatic Access

### Load and analyze logs in Python:

```python
from logger import load_training_log, compare_experiments
import json

# Load full training log
log = load_training_log('logs/experiments/my_exp/training_log.json')

# Access experiment info
print(log['experiment_info']['name'])
print(log['experiment_info']['total_epochs'])

# Get all epochs
epochs = log['epochs']
for epoch in epochs:
    print(f"Epoch {epoch['epoch']}: Val Acc = {epoch['validation']['cls_acc']:.2f}%")

# Load best metrics
with open('logs/experiments/my_exp/best_metrics.json', 'r') as f:
    best = json.load(f)
    print(f"Best Val Acc: {best['best_val_acc']:.2f}%")

# Compare experiments
comparison = compare_experiments([
    'logs/experiments/exp1/experiment_summary.json',
    'logs/experiments/exp2/experiment_summary.json'
])
```

## 📊 Plotting Examples

### Plot training curves:

```python
from logger import plot_training_curves

# Interactive plot
plot_training_curves('logs/experiments/my_exp/training_log.json')

# Save to file
plot_training_curves(
    'logs/experiments/my_exp/training_log.json',
    save_path='my_curves.png'
)
```

### Custom analysis:

```python
import json
import matplotlib.pyplot as plt

# Load log
with open('logs/experiments/my_exp/training_log.json', 'r') as f:
    data = json.load(f)

# Extract data
epochs = [e['epoch'] for e in data['epochs']]
train_acc = [e['train']['cls_acc'] for e in data['epochs']]
val_acc = [e['validation']['cls_acc'] for e in data['epochs']]

# Plot
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_acc, label='Train', marker='o')
plt.plot(epochs, val_acc, label='Validation', marker='s')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Classification Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('custom_plot.png', dpi=150, bbox_inches='tight')
plt.show()
```

## 🔧 Advanced Usage

### Custom Experiment Naming:

```bash
# Use descriptive names for experiments
python main.py --exp-name baseline_resnet50
python main.py --exp-name with_center_loss_0.001
python main.py --exp-name no_augmentation
python main.py --exp-name large_batch_32
```

### Batch Comparison:

```bash
# Compare all experiments with a prefix
python view_logs.py --compare baseline_* 

# Compare different configurations
python view_logs.py --compare \
    baseline_resnet50 \
    with_center_loss_0.001 \
    no_augmentation
```

### Export for External Analysis:

```bash
# Export to CSV for Excel/Google Sheets
python view_logs.py --export-csv my_experiment

# Then open in Excel or use pandas
import pandas as pd
df = pd.read_csv('logs/experiments/my_exp/training_log.csv')
print(df.describe())
```

## 📈 Monitoring During Training

### Real-time monitoring script:

```python
# monitor.py
import json
import time
from pathlib import Path

exp_name = "my_running_experiment"
log_path = Path(f"logs/experiments/{exp_name}/training_log.json")

print(f"Monitoring: {exp_name}")
print("Press Ctrl+C to stop\n")

last_epoch = 0
while True:
    if log_path.exists():
        with open(log_path, 'r') as f:
            data = json.load(f)
        
        epochs = data.get('epochs', [])
        if epochs and len(epochs) > last_epoch:
            latest = epochs[-1]
            last_epoch = len(epochs)
            
            print(f"\rEpoch {latest['epoch']}: "
                  f"Train Acc={latest['train']['cls_acc']:.2f}% | "
                  f"Val Acc={latest['validation']['cls_acc']:.2f}% | "
                  f"LR={latest['learning_rate']:.2e}", 
                  end='', flush=True)
    
    time.sleep(5)  # Check every 5 seconds
```

```bash
# Run in separate terminal
python monitor.py
```

## 🎯 Best Practices

1. **Use descriptive experiment names:**
   ```bash
   python main.py --exp-name baseline_v1_lr0.001_bs16
   ```

2. **Keep logs organized:**
   - One experiment = one directory
   - Never manually edit JSON files
   - Back up important experiments

3. **Regular cleanup:**
   ```bash
   # Archive old experiments
   mkdir logs/archive
   mv logs/experiments/old_exp_* logs/archive/
   ```

4. **Version control:**
   - Add `logs/` to `.gitignore`
   - Only commit important experiment summaries
   - Use Git tags for major experiments

5. **Documentation:**
   - Add notes in experiment names
   - Keep a separate notes file for observations
   - Screenshot important plots

## 🐛 Troubleshooting

### "Experiment not found"
```bash
# Check available experiments
python view_logs.py --list

# Verify path
ls -la logs/experiments/
```

### "JSON decode error"
- Training was interrupted before saving
- File is corrupted
- Delete and retrain

### "No data to plot"
- Experiment has 0 epochs completed
- Check if training actually ran

### Missing metrics
- Some metrics (e.g., spoof_metrics) might be None
- Check if anti-spoofing head is enabled
- Verify fake samples in validation set

## 📝 Example Workflow

```bash
# 1. Start experiment
python main.py --exp-name baseline_test

# 2. Monitor in another terminal
python view_logs.py --exp baseline_test

# 3. After training, view details
python view_logs.py --exp baseline_test

# 4. Plot curves
python view_logs.py --plot baseline_test

# 5. Export for analysis
python view_logs.py --export-csv baseline_test

# 6. Compare with other experiments
python view_logs.py --compare baseline_test improved_v1 improved_v2

# 7. Archive if successful
mkdir logs/archive
cp -r logs/experiments/baseline_test logs/archive/
```

## 🔗 Integration with Other Tools

### TensorBoard Integration:
- JSON logs complement TensorBoard
- Use TensorBoard for real-time monitoring
- Use JSON logs for final analysis and archiving

### Weights & Biases / MLflow:
```python
# Easy to export to W&B
import wandb
with open('logs/experiments/my_exp/training_log.json', 'r') as f:
    data = json.load(f)
    
for epoch in data['epochs']:
    wandb.log({
        'train_acc': epoch['train']['cls_acc'],
        'val_acc': epoch['validation']['cls_acc'],
        'epoch': epoch['epoch']
    })
```

## ✅ Checklist

Before starting a new experiment:
- [ ] Clear old checkpoints if needed
- [ ] Choose descriptive experiment name
- [ ] Verify config settings
- [ ] Check disk space for logs

After experiment completes:
- [ ] View experiment summary
- [ ] Plot training curves
- [ ] Export to CSV if needed
- [ ] Compare with previous experiments
- [ ] Archive important experiments
- [ ] Document findings

---

**Happy Training! 🚀**