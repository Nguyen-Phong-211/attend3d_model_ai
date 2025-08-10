# trainer.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from torch.nn.functional import one_hot

class Trainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = torch.device(config.DEVICE)
        self.model.to(self.device)

        self.criterion_cls = nn.CrossEntropyLoss()
        self.criterion_bce = nn.BCELoss()
        # optional: triplet
        self.criterion_triplet = nn.TripletMarginLoss(margin=0.2)

        self.optimizer = optim.Adam(self.model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=max(1, config.EPOCHS))

        os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    def train_epoch(self, loader, epoch):
        self.model.train()
        total_loss = 0.0
        total_cls = 0.0
        total_samples = 0
        all_preds, all_labels = [], []

        pbar = tqdm(loader, desc=f"Train E{epoch+1}")
        for batch_idx, (inputs, labels) in enumerate(pbar):
            # move labels
            labels = labels.to(self.device)
            # move inputs fields carefully
            inputs_cuda = {}
            for k, v in inputs.items():
                if v is None:
                    inputs_cuda[k] = None
                    continue
                # mesh might be shape (batch_size, M, 3) or a list/tuple; ensure correct
                inputs_cuda[k] = v.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs_cuda, labels)

            logits = outputs['logits']
            loss_cls = self.criterion_cls(logits, labels)

            loss = loss_cls

            # anti-spoofing: assume training samples are real -> label=1
            if outputs.get('spoof_score') is not None:
                real_labels = torch.ones_like(outputs['spoof_score']).to(self.device)
                loss_spf = self.criterion_bce(outputs['spoof_score'], real_labels)
                loss += 0.1 * loss_spf

            # optional: add triplet loss on embeddings if batch allows (we use anchor, pos, neg by label sampling)
            # skip complex batch mining for now

            loss.backward()
            self.optimizer.step()

            # stats
            total_loss += loss.item() * labels.size(0)
            total_cls += loss_cls.item() * labels.size(0)
            total_samples += labels.size(0)

            preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.detach().cpu().numpy().tolist())

            pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{(accuracy_score(all_labels, all_preds)*100):.2f}%"})

        avg_loss = total_loss / total_samples
        avg_cls = total_cls / total_samples
        acc = accuracy_score(all_labels, all_preds) * 100
        return avg_loss, avg_cls, acc

    @torch.no_grad()
    def validate(self, loader, epoch):
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        all_preds, all_labels = [], []

        pbar = tqdm(loader, desc=f"Val E{epoch+1}")
        for inputs, labels in pbar:
            labels = labels.to(self.device)
            inputs_cuda = {}
            for k, v in inputs.items():
                if v is None:
                    inputs_cuda[k] = None
                    continue
                inputs_cuda[k] = v.to(self.device)

            outputs = self.model(inputs_cuda, labels)
            logits = outputs['logits']
            loss = self.criterion_cls(logits, labels)
            total_loss += loss.item() * labels.size(0)
            total_samples += labels.size(0)

            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / total_samples
        acc = accuracy_score(all_labels, all_preds) * 100
        f1 = f1_score(all_labels, all_preds, average='weighted')
        return avg_loss, acc, f1

    def save_checkpoint(self, epoch, val_acc):
        fname = f"best_epoch_{epoch+1}_acc_{val_acc:.2f}.pth"
        state = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'epoch': epoch
        }
        path = os.path.join(self.config.CHECKPOINT_DIR, fname)
        torch.save(state, path)
        print(f"Saved checkpoint: {path}")

    def train(self, train_loader, val_loader, num_classes):
        best_acc = 0.0
        for epoch in range(self.config.EPOCHS):
            train_loss, train_cls_loss, train_acc = self.train_epoch(train_loader, epoch)
            val_loss, val_acc, val_f1 = self.validate(val_loader, epoch)
            print(f"Epoch {epoch+1}/{self.config.EPOCHS} - Train loss: {train_loss:.4f}, Train acc: {train_acc:.2f}%")
            print(f"Val loss: {val_loss:.4f}, Val acc: {val_acc:.2f}%, Val f1: {val_f1:.4f}")

            # scheduler step
            self.scheduler.step()

            if val_acc > best_acc:
                best_acc = val_acc
                self.save_checkpoint(epoch, val_acc)