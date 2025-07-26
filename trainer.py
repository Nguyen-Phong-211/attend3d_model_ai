import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import numpy as np

class Trainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = config.DEVICE
        
        # Loss functions
        self.criterion_cls = nn.CrossEntropyLoss()
        self.criterion_anti_spoofing = nn.BCELoss()
        
        # Optimizer
        self.optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
        # self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.5)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=config.EPOCHS)
        
        # Move model to device
        self.model.to(self.device)
        
        # Create checkpoint directory
        os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
        
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_loader, desc="Training")
        
        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            # Move data to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            targets = targets.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(inputs)
            
            # Calculate losses
            loss_cls = self.criterion_cls(outputs['logits'], targets)
            loss = loss_cls
            
            # Anti-spoofing loss (assume all training samples are real)
            if outputs['spoofing_score'] is not None:
                real_labels = torch.ones_like(outputs['spoofing_score'])
                loss_anti_spoofing = self.criterion_anti_spoofing(
                    outputs['spoofing_score'], 
                    real_labels
                )
                loss += 0.1 * loss_anti_spoofing
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = outputs['logits'].max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc="Validation")
            
            for inputs, targets in progress_bar:
                # Move data to device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                targets = targets.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                
                # Calculate loss
                loss = self.criterion_cls(outputs['logits'], targets)
                total_loss += loss.item()
                
                # Predictions
                _, predicted = outputs['logits'].max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                
                progress_bar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100.*correct/total:.2f}%'
                })
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100. * correct / total
        sklearn_acc = accuracy_score(all_targets, all_predictions) * 100
        
        # add f1 score
        f1 = f1_score(all_targets, all_predictions, average='weighted')
        print(f"F1 Score: {f1:.4f}")
        
        return avg_loss, accuracy, sklearn_acc
    
    def train(self, train_loader, val_loader, num_classes):
        best_acc = 0.0
        early_stopper = EarlyStopping(patience=7, verbose=True)
        
        print(f"Starting training on {self.device}")
        print(f"Number of classes: {num_classes}")
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")
        
        for epoch in range(self.config.EPOCHS):
            print(f"\nEpoch {epoch+1}/{self.config.EPOCHS}")
            print("-" * 50)
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_acc, sklearn_acc = self.validate(val_loader)
            
            # Update learning rate
            self.scheduler.step()
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print(f"Sklearn Acc: {sklearn_acc:.2f}%")
            
            # Save best model
            if val_acc > best_acc:
                best_acc = val_acc
                self.save_checkpoint(f"best_model_acc_{val_acc:.2f}.pth")
                print(f"New best model saved with accuracy: {best_acc:.2f}%")
                
            early_stopper(val_acc)
            if early_stopper.early_stop:
                print("Early stopping triggered. Stopping training.")
                break
        
        print(f"\nTraining completed. Best validation accuracy: {best_acc:.2f}%")
    
    def save_checkpoint(self, filename):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }
        torch.save(checkpoint, os.path.join(self.config.CHECKPOINT_DIR, filename))
    
    def load_checkpoint(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0.0):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        self.best_acc = 0.0

    def __call__(self, val_acc):

        score = val_acc

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0