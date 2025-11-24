import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
import time
import math
from torch.utils.tensorboard import SummaryWriter

# Import CenterLoss from model.py
from model import CenterLoss

# Import JSON Logger
from logger import TrainingLogger


class WarmupCosineScheduler:
    """Cosine annealing with linear warmup"""
    def __init__(self, optimizer, warmup_epochs, max_epochs, base_lr, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.current_epoch = 0
    
    def step(self, epoch=None):
        if epoch is not None:
            self.current_epoch = epoch
        else:
            self.current_epoch += 1
        
        if self.current_epoch < self.warmup_epochs:
            lr = self.base_lr * (self.current_epoch + 1) / self.warmup_epochs
        else:
            progress = (self.current_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr


class Trainer:
    """
    Enhanced Trainer with JSON logging
    """
    
    def __init__(self, model, config, experiment_name=None):
        self.model = model
        self.config = config
        self.device = torch.device(config.DEVICE)
        self.model.to(self.device)

        # TensorBoard
        log_dir = getattr(config, 'LOG_DIR', 'runs/experiment_1')
        self.writer = SummaryWriter(log_dir=log_dir)
        print(f"\nTensorBoard logging to: {log_dir}")

        # JSON Logger - NEW!
        json_log_dir = getattr(config, 'JSON_LOG_DIR', 'logs/experiments')
        if experiment_name is None:
            experiment_name = f"exp_{time.strftime('%Y%m%d_%H%M%S')}"
        
        self.json_logger = TrainingLogger(
            log_dir=os.path.join(json_log_dir, experiment_name),
            config=config,
            experiment_name=experiment_name
        )

        # === LOSS FUNCTIONS ===
        self.criterion_cls = nn.CrossEntropyLoss(
            label_smoothing=getattr(config, 'LABEL_SMOOTHING', 0.1)
        )
        self.criterion_bce = nn.BCEWithLogitsLoss()
        self.criterion_mse = nn.MSELoss()
        
        # CENTER LOSS
        self.use_center_loss = getattr(config, 'USE_CENTER_LOSS', True)
        if self.use_center_loss:
            num_classes = model.num_classes
            self.criterion_center = CenterLoss(
                num_classes, 
                config.EMBEDDING_DIM, 
                self.device
            )
            self.optimizer_center = optim.SGD(
                self.criterion_center.parameters(), 
                lr=0.5
            )
            print(f"\nUsing Center Loss enabled")
        
        # === OPTIMIZER ===
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=config.LEARNING_RATE,
            weight_decay=getattr(config, 'WEIGHT_DECAY', 5e-4),
            betas=(0.9, 0.999)
        )
        
        # === SCHEDULER ===
        warmup_epochs = getattr(config, 'WARMUP_EPOCHS', 5)
        min_lr = getattr(config, 'MIN_LR', 1e-6)
        
        self.scheduler = WarmupCosineScheduler(
            self.optimizer,
            warmup_epochs=warmup_epochs,
            max_epochs=config.EPOCHS,
            base_lr=config.LEARNING_RATE,
            min_lr=min_lr
        )
        
        print(f"\nScheduler: Warmup Cosine")
        print(f"  Warmup epochs: {warmup_epochs}")
        print(f"  Base LR: {config.LEARNING_RATE:.2e}")
        print(f"  Min LR: {min_lr:.2e}")
        
        # === MIXED PRECISION ===
        self.use_amp = getattr(config, 'USE_AMP', False) and config.DEVICE == 'cuda'
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        
        # === GRADIENT CLIPPING ===
        self.max_grad_norm = getattr(config, 'MAX_GRAD_NORM', 1.0)
        
        # === EARLY STOPPING ===
        self.patience = getattr(config, 'PATIENCE', 15)
        self.early_stop_counter = 0
        self.best_val_acc = 0.0
        self.best_spoof_auc = 0.0
        
        os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
        
        print(f"\nTrainer initialized:")
        print(f"  Device: {self.device}")
        print(f"  Optimizer: AdamW (lr={config.LEARNING_RATE:.2e})")
        print(f"  Mixed Precision: {self.use_amp}")
        print(f"  Gradient Clipping: {self.max_grad_norm}")
        print(f"  Early Stopping: {self.patience}")

    def compute_spoofing_metrics(self, spoof_scores, spoof_labels, threshold=0.5):
        """ Compute spoofing metrics """
        spoof_scores = np.array(spoof_scores)
        spoof_labels = np.array(spoof_labels)
        
        if len(spoof_scores) == 0:
            return None
        
        unique_labels = np.unique(spoof_labels)
        if len(unique_labels) < 2:
            print(f"[WARN] Only one class in spoof_labels: {unique_labels}")
            return {
                'auc': 0.0,
                'acc': 0.0,
                'apcer': 0.0,
                'bpcer': 0.0,
                'eer': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0
            }
        
        try:
            auc = roc_auc_score(spoof_labels, spoof_scores)
        except Exception as e:
            print(f"[WARN] Cannot compute AUC: {e}")
            auc = 0.0
        
        spoof_preds = (spoof_scores >= threshold).astype(int)
        
        try:
            tn, fp, fn, tp = confusion_matrix(
                spoof_labels, 
                spoof_preds, 
                labels=[0, 1]
            ).ravel()
        except Exception as e:
            print(f"[WARN] Confusion matrix error: {e}")
            return {'auc': auc, 'acc': 0.0}
        
        apcer = fn / (fn + tp + 1e-8)
        bpcer = fp / (fp + tn + 1e-8)
        eer = (apcer + bpcer) / 2
        
        acc = (tp + tn) / (tp + tn + fp + fn + 1e-8)
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        
        return {
            'auc': auc,
            'apcer': apcer,
            'bpcer': bpcer,
            'eer': eer,
            'acc': acc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': int(tp),
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn)
        }

    def train_epoch(self, loader, epoch):
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0.0
        total_cls_loss = 0.0
        total_spoof_loss = 0.0
        total_center_loss = 0.0
        total_depth_loss = 0.0
        total_samples = 0
        
        all_preds = []
        all_labels = []
        all_spoof_scores = []
        all_spoof_labels = []
        
        pbar = tqdm(loader, desc=f"Train Epoch {epoch+1}/{self.config.EPOCHS}")
        
        for batch_idx, (inputs, labels, is_spoof) in enumerate(pbar):
            labels = labels.to(self.device)
            is_spoof = is_spoof.to(self.device)
            
            inputs_cuda = {}
            for k, v in inputs.items():
                if v is not None:
                    inputs_cuda[k] = v.to(self.device)
            
            # === FORWARD ===
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    loss = self._compute_loss(inputs_cuda, labels, is_spoof)
                
                self.optimizer.zero_grad()
                if self.use_center_loss:
                    self.optimizer_center.zero_grad()
                
                self.scaler.scale(loss['total']).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.scaler.step(self.optimizer)
                
                if self.use_center_loss:
                    self.scaler.step(self.optimizer_center)
                
                self.scaler.update()
            else:
                loss = self._compute_loss(inputs_cuda, labels, is_spoof)
                
                self.optimizer.zero_grad()
                if self.use_center_loss:
                    self.optimizer_center.zero_grad()
                
                loss['total'].backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                if self.use_center_loss:
                    self.optimizer_center.step()
            
            # === STATISTICS ===
            batch_size = labels.size(0)
            total_loss += loss['total'].item() * batch_size
            total_cls_loss += loss['cls'].item() * batch_size
            total_spoof_loss += loss['spoof'].item() * batch_size
            total_center_loss += loss.get('center', torch.tensor(0.0)).item() * batch_size
            total_depth_loss += loss.get('depth', torch.tensor(0.0)).item() * batch_size
            total_samples += batch_size
            
            # Predictions
            preds = torch.argmax(loss['logits'], dim=1).detach().cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.detach().cpu().numpy().tolist())
            
            if loss.get('spoof_score') is not None:
                spf_prob = torch.sigmoid(loss['spoof_score']).detach().cpu().numpy().flatten()
                all_spoof_scores.extend(spf_prob.tolist())
                all_spoof_labels.extend(is_spoof.detach().cpu().numpy().flatten().tolist())
            
            # Progress bar
            current_acc = accuracy_score(all_labels, all_preds) * 100
            pbar.set_postfix({
                'loss': f'{loss["total"].item():.4f}',
                'cls': f'{loss["cls"].item():.3f}',
                'spf': f'{loss["spoof"].item():.3f}',
                'acc': f'{current_acc:.1f}%'
            })
        
        # Compute metrics
        avg_loss = total_loss / total_samples
        avg_cls_loss = total_cls_loss / total_samples
        avg_spoof_loss = total_spoof_loss / total_samples
        avg_center_loss = total_center_loss / total_samples
        avg_depth_loss = total_depth_loss / total_samples
        
        cls_acc = accuracy_score(all_labels, all_preds) * 100
        
        spoof_metrics = None
        if len(all_spoof_scores) > 0:
            spoof_metrics = self.compute_spoofing_metrics(
                all_spoof_scores, 
                all_spoof_labels,
                threshold=getattr(self.config, 'SPOOF_THRESHOLD', 0.5)
            )
        
        # TensorBoard logging
        self.writer.add_scalar('Train/Loss', avg_loss, epoch)
        self.writer.add_scalar('Train/ClassificationLoss', avg_cls_loss, epoch)
        self.writer.add_scalar('Train/SpoofLoss', avg_spoof_loss, epoch)
        self.writer.add_scalar('Train/CenterLoss', avg_center_loss, epoch)
        self.writer.add_scalar('Train/DepthLoss', avg_depth_loss, epoch)
        self.writer.add_scalar('Train/Accuracy', cls_acc, epoch)
        
        if spoof_metrics is not None:
            self.writer.add_scalar('Train/SpoofAUC', spoof_metrics['auc'], epoch)
            self.writer.add_scalar('Train/SpoofAccuracy', spoof_metrics['acc'] * 100, epoch)
        
        return {
            'loss': avg_loss,
            'cls_loss': avg_cls_loss,
            'spoof_loss': avg_spoof_loss,
            'center_loss': avg_center_loss,
            'depth_loss': avg_depth_loss,
            'cls_acc': cls_acc,
            'spoof_metrics': spoof_metrics
        }
    
    def _compute_loss(self, inputs_cuda, labels, is_spoof):
        """IMPROVED LOSS COMPUTATION"""
        outputs = self.model(inputs_cuda, labels)
        logits = outputs['logits']
        embeddings = outputs['embeddings']
        
        # 1. Classification Loss - ONLY FOR REAL
        real_mask = (is_spoof == 0)

        if real_mask.sum() > 0:
            loss_cls = self.criterion_cls(logits[real_mask], labels[real_mask])
        else:
            loss_cls = torch.tensor(0.0).to(self.device)
        
        total_loss = loss_cls
        
        # 2. Anti-Spoofing Loss
        spoof_score = outputs.get('spoof_score')
        loss_spf = torch.tensor(0.0).to(self.device)
        
        if spoof_score is not None:
            spoof_labels = is_spoof.view_as(spoof_score)
            loss_spf = self.criterion_bce(spoof_score, spoof_labels)
            
            # 3. Depth Auxiliary Loss (only for REAL faces)
            loss_depth = torch.tensor(0.0).to(self.device)
            if 'depth_pred' in outputs and 'depth' in inputs_cuda:
                real_mask = (is_spoof == 0).float()
                if real_mask.sum() > 0:
                    depth_gt = torch.nn.functional.interpolate(
                        inputs_cuda['depth'], 
                        size=(32, 32), 
                        mode='bilinear',
                        align_corners=False
                    )
                    depth_pred = outputs['depth_pred']
                    
                    loss_depth = self.criterion_mse(
                        depth_pred * real_mask.view(-1, 1, 1, 1),
                        depth_gt * real_mask.view(-1, 1, 1, 1)
                    )
                    
                    depth_weight = getattr(self.config, 'DEPTH_AUX_WEIGHT', 0.1)
                    loss_spf = loss_spf + depth_weight * loss_depth
            
            spoof_weight = getattr(self.config, 'SPOOF_LOSS_WEIGHT', 1.0)
            total_loss = total_loss + spoof_weight * loss_spf
        
        # 4. Center Loss - ONLY FOR REAL
        loss_center = torch.tensor(0.0).to(self.device)
        if self.use_center_loss and real_mask.sum() > 0:
            loss_center = self.criterion_center(
                embeddings[real_mask], 
                labels[real_mask]
            )
            center_weight = getattr(self.config, 'CENTER_LOSS_WEIGHT', 0.000001)
            total_loss = total_loss + center_weight * loss_center
        
        return {
            'total': total_loss,
            'cls': loss_cls,
            'spoof': loss_spf,
            'center': loss_center,
            'depth': loss_depth,
            'logits': logits,
            'spoof_score': spoof_score
        }

    @torch.no_grad()
    def validate(self, loader, epoch):
        """Validate for one epoch"""
        self.model.eval()
        
        total_loss = 0.0
        total_samples = 0
        
        all_preds = []
        all_labels = []
        all_spoof_scores = []
        all_spoof_labels = []
        
        pbar = tqdm(loader, desc=f"Val Epoch {epoch+1}/{self.config.EPOCHS}")
        
        for inputs, labels, is_spoof in pbar:
            labels = labels.to(self.device)
            is_spoof = is_spoof.to(self.device)
            
            inputs_cuda = {}
            for k, v in inputs.items():
                if v is not None:
                    inputs_cuda[k] = v.to(self.device)
            
            outputs = self.model(inputs_cuda, labels)
            logits = outputs['logits']
            
            loss = self.criterion_cls(logits, labels)
            total_loss += loss.item() * labels.size(0)
            total_samples += labels.size(0)
            
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
            
            if outputs.get('spoof_score') is not None:
                spf_prob = torch.sigmoid(outputs['spoof_score']).cpu().numpy().flatten()
                all_spoof_scores.extend(spf_prob.tolist())
                all_spoof_labels.extend(is_spoof.cpu().numpy().flatten().tolist())
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / total_samples
        cls_acc = accuracy_score(all_labels, all_preds) * 100
        cls_f1 = f1_score(all_labels, all_preds, average='weighted')
        
        spoof_metrics = None
        if len(all_spoof_scores) > 0:
            spoof_metrics = self.compute_spoofing_metrics(
                all_spoof_scores,
                all_spoof_labels,
                threshold=getattr(self.config, 'SPOOF_THRESHOLD', 0.5)
            )
        
        self.writer.add_scalar('Val/Loss', avg_loss, epoch)
        self.writer.add_scalar('Val/Accuracy', cls_acc, epoch)
        self.writer.add_scalar('Val/F1', cls_f1, epoch)
        
        if spoof_metrics is not None:
            self.writer.add_scalar('Val/SpoofAUC', spoof_metrics['auc'], epoch)
            self.writer.add_scalar('Val/SpoofAccuracy', spoof_metrics['acc'] * 100, epoch)
        
        return {
            'loss': avg_loss,
            'cls_acc': cls_acc,
            'cls_f1': cls_f1,
            'spoof_metrics': spoof_metrics
        }

    def print_metrics(self, prefix, metrics):
        """Print metrics"""
        print(f"\n{prefix}:")
        print(f"  Loss: {metrics['loss']:.4f}")
        print(f"  Classification:")
        print(f"    - Accuracy: {metrics['cls_acc']:.2f}%")
        
        if 'cls_f1' in metrics:
            print(f"    - F1 Score: {metrics['cls_f1']:.4f}")
        
        if 'cls_loss' in metrics:
            print(f"    - Loss: {metrics['cls_loss']:.4f}")
        
        if 'center_loss' in metrics and metrics.get('center_loss', 0) > 0:
            print(f"    - Center Loss: {metrics['center_loss']:.4f}")
        
        if 'depth_loss' in metrics and metrics.get('depth_loss', 0) > 0:
            print(f"    - Depth Loss: {metrics['depth_loss']:.4f}")
        
        if metrics.get('spoof_metrics'):
            sm = metrics['spoof_metrics']
            print(f"  Anti-Spoofing:")
            print(f"    - AUC: {sm['auc']:.4f}")
            print(f"    - Accuracy: {sm['acc']*100:.2f}%")
            print(f"    - APCER: {sm['apcer']*100:.2f}%")
            print(f"    - BPCER: {sm['bpcer']*100:.2f}%")
            print(f"    - EER: {sm['eer']*100:.2f}%")
            print(f"    - F1: {sm['f1']:.4f}")

    def save_checkpoint(self, epoch, metrics, fname=None):
        """Save checkpoint"""
        if fname is None:
            fname = f"checkpoint_e{epoch+1}.pth"
        
        state = {
            'epoch': epoch,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'metrics': metrics,
            'config': vars(self.config)
        }
        
        if self.use_center_loss:
            state['center_loss'] = self.criterion_center.state_dict()
            state['optimizer_center'] = self.optimizer_center.state_dict()
        
        path = os.path.join(self.config.CHECKPOINT_DIR, fname)
        torch.save(state, path)
        print(f"  ✓ Saved: {fname}")

    def train(self, train_loader, val_loader, num_classes):
        """Full training loop with JSON logging"""
        print("\n" + "="*70)
        print(f"Starting Training")
        print(f"  Epochs: {self.config.EPOCHS}")
        print(f"  Device: {self.device}")
        print(f"  Num Classes: {num_classes}")
        print("="*70)
        
        for epoch in range(self.config.EPOCHS):
            epoch_start = time.time()
            
            train_metrics = self.train_epoch(train_loader, epoch)
            val_metrics = self.validate(val_loader, epoch)
            
            current_lr = self.scheduler.step(epoch)
            epoch_duration = time.time() - epoch_start
            
            # TensorBoard
            self.writer.add_scalar('LearningRate', current_lr, epoch)
            
            # JSON Logger - Log this epoch
            self.json_logger.log_epoch(
                epoch=epoch,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                lr=current_lr,
                duration=epoch_duration
            )
            
            print("\n" + "="*70)
            print(f"Epoch {epoch+1}/{self.config.EPOCHS} (LR: {current_lr:.2e})")
            self.print_metrics("Train", train_metrics)
            self.print_metrics("Validation", val_metrics)
            print(f"  Time: {epoch_duration:.1f}s")
            print("="*70)
            
            if (epoch + 1) % self.config.SAVE_EVERY == 0:
                self.save_checkpoint(epoch, val_metrics)
            
            current_val_acc = val_metrics['cls_acc']
            current_spoof_auc = val_metrics['spoof_metrics']['auc'] if val_metrics.get('spoof_metrics') else 0
            
            if current_val_acc > self.best_val_acc:
                self.best_val_acc = current_val_acc
                self.save_checkpoint(epoch, val_metrics, fname='best_acc.pth')
                print(f"  ★ New best accuracy: {current_val_acc:.2f}%")
                self.early_stop_counter = 0
            else:
                self.early_stop_counter += 1
            
            if current_spoof_auc > self.best_spoof_auc:
                self.best_spoof_auc = current_spoof_auc
                self.save_checkpoint(epoch, val_metrics, fname='best_spoof_auc.pth')
                print(f"  ★ New best spoof AUC: {current_spoof_auc:.4f}")
            
            if self.early_stop_counter >= self.patience:
                print(f"\n⚠️  Early stopping triggered after {epoch+1} epochs")
                self.json_logger.log_early_stop(epoch, "Patience exceeded")
                break
        
        # Finalize logging
        self.json_logger.finalize(total_epochs_completed=epoch+1)
        self.json_logger.print_summary()
        
        self.writer.close()
        
        print("\n" + "="*70)
        print("Training Complete!")
        print(f"  Best Classification Accuracy: {self.best_val_acc:.2f}%")
        print(f"  Best Spoofing AUC: {self.best_spoof_auc:.4f}")
        print("="*70 + "\n")