# trainer.py - COMPLETE VERSION
import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
import time

class Trainer:
    """
    Trainer cho Face Recognition + Anti-Spoofing
    
    Features:
    - Multi-task learning (identity + spoofing)
    - Comprehensive metrics (ACC, F1, AUC, APCER, BPCER, EER)
    - Mixed precision training support
    - Gradient clipping
    - Early stopping
    - Best model tracking
    """
    
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = torch.device(config.DEVICE)
        self.model.to(self.device)

        # Loss functions
        self.criterion_cls = nn.CrossEntropyLoss(label_smoothing=0.1)  # Label smoothing
        self.criterion_bce = nn.BCEWithLogitsLoss()
        
        # Optimizer với weight decay
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=config.LEARNING_RATE,
            weight_decay=getattr(config, 'WEIGHT_DECAY', 1e-4),
            betas=(0.9, 0.999)
        )
        
        # Learning rate scheduler
        scheduler_type = getattr(config, 'SCHEDULER', 'cosine')
        if scheduler_type == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer, 
                T_0=10,      # Restart every 10 epochs
                T_mult=2,    # Double period after each restart
                eta_min=1e-6
            )
        elif scheduler_type == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.1
            )
        else:
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, 
                T_max=max(1, config.EPOCHS)
            )
        
        # Mixed precision training
        self.use_amp = getattr(config, 'USE_AMP', False)
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        
        # Gradient clipping
        self.max_grad_norm = getattr(config, 'MAX_GRAD_NORM', 1.0)
        
        # Early stopping
        self.patience = getattr(config, 'PATIENCE', 10)
        self.early_stop_counter = 0
        self.best_val_acc = 0.0
        self.best_spoof_auc = 0.0
        
        os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
        
        print(f"\nTrainer initialized:")
        print(f"  Device: {self.device}")
        print(f"  Optimizer: AdamW (lr={config.LEARNING_RATE}, wd={getattr(config, 'WEIGHT_DECAY', 1e-4)})")
        print(f"  Scheduler: {scheduler_type}")
        print(f"  Mixed Precision: {self.use_amp}")
        print(f"  Gradient Clipping: {self.max_grad_norm}")
        print(f"  Early Stopping Patience: {self.patience}")

    def compute_spoofing_metrics(self, spoof_scores, spoof_labels, threshold=0.5):
        """
        Tính các metrics cho anti-spoofing
        
        Args:
            spoof_scores: numpy array of probabilities (0=real, 1=fake)
            spoof_labels: numpy array of ground truth (0=real, 1=fake)
            threshold: decision threshold
            
        Returns:
            dict: metrics including AUC, APCER, BPCER, EER, etc.
        """
        spoof_scores = np.array(spoof_scores)
        spoof_labels = np.array(spoof_labels)
        
        if len(spoof_scores) == 0:
            return None
        
        # AUC
        try:
            auc = roc_auc_score(spoof_labels, spoof_scores)
        except:
            auc = 0.0
        
        # Predictions
        spoof_preds = (spoof_scores >= threshold).astype(int)
        
        # Confusion matrix
        try:
            tn, fp, fn, tp = confusion_matrix(
                spoof_labels, 
                spoof_preds, 
                labels=[0, 1]
            ).ravel()
        except:
            return {'auc': auc, 'acc': 0.0}
        
        # APCER: Attack Presentation Classification Error Rate
        # Tỉ lệ fake (1) bị nhận nhầm là real (predict 0)
        apcer = fn / (fn + tp + 1e-8)
        
        # BPCER: Bona Fide Presentation Classification Error Rate  
        # Tỉ lệ real (0) bị nhận nhầm là fake (predict 1)
        bpcer = fp / (fp + tn + 1e-8)
        
        # EER: Equal Error Rate (ước lượng)
        eer = (apcer + bpcer) / 2
        
        # Accuracy
        acc = (tp + tn) / (tp + tn + fp + fn + 1e-8)
        
        # Precision & Recall
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
            'tp': tp,
            'tn': tn,
            'fp': fp,
            'fn': fn
        }

    def train_epoch(self, loader, epoch):
        """Train for one epoch"""
        self.model.train()
        
        # Metrics tracking
        total_loss = 0.0
        total_cls_loss = 0.0
        total_spoof_loss = 0.0
        total_samples = 0
        
        all_preds = []
        all_labels = []
        all_spoof_scores = []
        all_spoof_labels = []
        
        pbar = tqdm(loader, desc=f"Train Epoch {epoch+1}/{self.config.EPOCHS}")
        
        for batch_idx, (inputs, labels, is_spoof) in enumerate(pbar):
            batch_start = time.time()
            
            labels = labels.to(self.device)
            is_spoof = is_spoof.to(self.device)
            
            # Move inputs to device
            inputs_cuda = {}
            for k, v in inputs.items():
                if v is not None:
                    inputs_cuda[k] = v.to(self.device)
            
            # Forward pass with mixed precision
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(inputs_cuda, labels)
                    logits = outputs['logits']
                    loss_cls = self.criterion_cls(logits, labels)
                    loss = loss_cls
                    
                    # Spoofing loss
                    loss_spf = torch.tensor(0.0).to(self.device)
                    if outputs.get('spoof_score') is not None:
                        spoof_score = outputs['spoof_score']
                        spoof_labels = is_spoof.view_as(spoof_score)
                        loss_spf = self.criterion_bce(spoof_score, spoof_labels)
                        loss += self.config.SPOOF_LOSS_WEIGHT * loss_spf
                
                # Backward with gradient scaling
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard training
                self.optimizer.zero_grad()
                
                outputs = self.model(inputs_cuda, labels)
                logits = outputs['logits']
                loss_cls = self.criterion_cls(logits, labels)
                loss = loss_cls
                
                # Spoofing loss
                loss_spf = torch.tensor(0.0).to(self.device)
                if outputs.get('spoof_score') is not None:
                    spoof_score = outputs['spoof_score']
                    spoof_labels = is_spoof.view_as(spoof_score)
                    loss_spf = self.criterion_bce(spoof_score, spoof_labels)
                    loss += self.config.SPOOF_LOSS_WEIGHT * loss_spf
                
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                
                self.optimizer.step()
            
            # Statistics
            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_cls_loss += loss_cls.item() * batch_size
            total_spoof_loss += loss_spf.item() * batch_size
            total_samples += batch_size
            
            # Predictions
            preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.detach().cpu().numpy().tolist())
            
            # Spoof predictions
            if outputs.get('spoof_score') is not None:
                spf_prob = torch.sigmoid(spoof_score).detach().cpu().numpy().flatten()
                all_spoof_scores.extend(spf_prob.tolist())
                all_spoof_labels.extend(is_spoof.detach().cpu().numpy().flatten().tolist())
            
            # Update progress bar
            current_acc = accuracy_score(all_labels, all_preds) * 100
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'cls': f'{loss_cls.item():.3f}',
                'spf': f'{loss_spf.item():.3f}',
                'acc': f'{current_acc:.1f}%'
            })
        
        # Compute epoch metrics
        avg_loss = total_loss / total_samples
        avg_cls_loss = total_cls_loss / total_samples
        avg_spoof_loss = total_spoof_loss / total_samples
        
        cls_acc = accuracy_score(all_labels, all_preds) * 100
        
        # Spoofing metrics
        spoof_metrics = None
        if len(all_spoof_scores) > 0:
            spoof_metrics = self.compute_spoofing_metrics(
                all_spoof_scores, 
                all_spoof_labels,
                threshold=getattr(self.config, 'SPOOF_THRESHOLD', 0.5)
            )
        
        return {
            'loss': avg_loss,
            'cls_loss': avg_cls_loss,
            'spoof_loss': avg_spoof_loss,
            'cls_acc': cls_acc,
            'spoof_metrics': spoof_metrics
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
            
            # Loss (chỉ classification cho validation)
            loss = self.criterion_cls(logits, labels)
            total_loss += loss.item() * labels.size(0)
            total_samples += labels.size(0)
            
            # Predictions
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
            
            # Spoof predictions
            if outputs.get('spoof_score') is not None:
                spf_prob = torch.sigmoid(outputs['spoof_score']).cpu().numpy().flatten()
                all_spoof_scores.extend(spf_prob.tolist())
                all_spoof_labels.extend(is_spoof.cpu().numpy().flatten().tolist())
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Compute metrics
        avg_loss = total_loss / total_samples
        cls_acc = accuracy_score(all_labels, all_preds) * 100
        cls_f1 = f1_score(all_labels, all_preds, average='weighted')
        
        # Spoofing metrics
        spoof_metrics = None
        if len(all_spoof_scores) > 0:
            spoof_metrics = self.compute_spoofing_metrics(
                all_spoof_scores,
                all_spoof_labels,
                threshold=getattr(self.config, 'SPOOF_THRESHOLD', 0.5)
            )
        
        return {
            'loss': avg_loss,
            'cls_acc': cls_acc,
            'cls_f1': cls_f1,
            'spoof_metrics': spoof_metrics
        }

    def print_metrics(self, prefix, metrics):
        """Print metrics in formatted way"""
        print(f"\n{prefix}:")
        print(f"  Loss: {metrics['loss']:.4f}")
        print(f"  Classification:")
        print(f"    - Accuracy: {metrics['cls_acc']:.2f}%")
        
        if 'cls_f1' in metrics:
            print(f"    - F1 Score: {metrics['cls_f1']:.4f}")
        
        if 'cls_loss' in metrics:
            print(f"    - Loss: {metrics['cls_loss']:.4f}")
        
        if metrics.get('spoof_metrics'):
            sm = metrics['spoof_metrics']
            print(f"  Anti-Spoofing:")
            print(f"    - AUC: {sm['auc']:.4f}")
            print(f"    - Accuracy: {sm['acc']*100:.2f}%")
            print(f"    - APCER (fake→real): {sm['apcer']*100:.2f}%")
            print(f"    - BPCER (real→fake): {sm['bpcer']*100:.2f}%")
            print(f"    - EER: {sm['eer']*100:.2f}%")
            print(f"    - F1: {sm['f1']:.4f}")
            
            if 'spoof_loss' in metrics:
                print(f"    - Loss: {metrics['spoof_loss']:.4f}")

    def save_checkpoint(self, epoch, metrics, fname=None):
        """Save model checkpoint"""
        if fname is None:
            fname = f"checkpoint_e{epoch+1}.pth"
        
        state = {
            'epoch': epoch,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': vars(self.config)
        }
        
        path = os.path.join(self.config.CHECKPOINT_DIR, fname)
        torch.save(state, path)
        print(f"  ✓ Saved: {fname}")

    def train(self, train_loader, val_loader, num_classes):
        """Full training loop"""
        print("\n" + "="*70)
        print(f"Starting Training")
        print(f"  Epochs: {self.config.EPOCHS}")
        print(f"  Device: {self.device}")
        print(f"  Num Classes: {num_classes}")
        print(f"  Train Batches: {len(train_loader)}")
        print(f"  Val Batches: {len(val_loader)}")
        print("="*70)
        
        for epoch in range(self.config.EPOCHS):
            epoch_start = time.time()
            
            # Train
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_metrics = self.validate(val_loader, epoch)
            
            # Learning rate step
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Print metrics
            print("\n" + "="*70)
            print(f"Epoch {epoch+1}/{self.config.EPOCHS} (LR: {current_lr:.2e})")
            self.print_metrics("Train", train_metrics)
            self.print_metrics("Validation", val_metrics)
            print(f"  Time: {time.time() - epoch_start:.1f}s")
            print("="*70)
            
            # Save regular checkpoint
            if (epoch + 1) % self.config.SAVE_EVERY == 0:
                self.save_checkpoint(epoch, val_metrics)
            
            # Track best models
            current_val_acc = val_metrics['cls_acc']
            current_spoof_auc = val_metrics['spoof_metrics']['auc'] if val_metrics.get('spoof_metrics') else 0
            
            # Best accuracy
            if current_val_acc > self.best_val_acc:
                self.best_val_acc = current_val_acc
                self.save_checkpoint(epoch, val_metrics, fname='best_acc.pth')
                print(f"  ★ New best accuracy: {current_val_acc:.2f}%")
                self.early_stop_counter = 0
            else:
                self.early_stop_counter += 1
            
            # Best spoof AUC
            if current_spoof_auc > self.best_spoof_auc:
                self.best_spoof_auc = current_spoof_auc
                self.save_checkpoint(epoch, val_metrics, fname='best_spoof_auc.pth')
                print(f"  ★ New best spoof AUC: {current_spoof_auc:.4f}")
            
            # Early stopping
            if self.early_stop_counter >= self.patience:
                print(f"\n⚠ Early stopping triggered after {epoch+1} epochs")
                print(f"  No improvement for {self.patience} epochs")
                break
        
        print("\n" + "="*70)
        print("Training Complete!")
        print(f"  Best Classification Accuracy: {self.best_val_acc:.2f}%")
        print(f"  Best Spoofing AUC: {self.best_spoof_auc:.4f}")
        print("="*70 + "\n")