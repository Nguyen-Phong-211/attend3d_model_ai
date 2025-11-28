#!/usr/bin/env python3
"""
Script to view and analyze training logs
Usage:
    python view_logs.py --exp <experiment_name>
    python view_logs.py --list
    python view_logs.py --compare exp1 exp2 exp3
    python view_logs.py --plot <experiment_name>
"""

import json
import os
import argparse
from pathlib import Path
from datetime import datetime
import sys


def list_experiments(log_dir='logs/experiments'):
    """List all available experiments"""
    log_path = Path(log_dir)
    
    if not log_path.exists():
        print(f" Log directory not found: {log_dir}")
        return
    
    experiments = [d for d in log_path.iterdir() if d.is_dir()]
    
    if not experiments:
        print(f"No experiments found in {log_dir}")
        return
    
    print(f"\n{'='*80}")
    print(f"Available Experiments ({len(experiments)})")
    print(f"{'='*80}\n")
    
    for exp_dir in sorted(experiments):
        exp_name = exp_dir.name
        summary_path = exp_dir / 'experiment_summary.json'
        
        if summary_path.exists():
            with open(summary_path, 'r') as f:
                summary = json.load(f)
            
            print(f"📁 {exp_name}")
            print(f"   Start: {summary.get('start_time', 'N/A')}")
            print(f"   Epochs: {summary.get('total_epochs', 'N/A')}")
            print(f"   Best Val Acc: {summary['best_metrics'].get('best_val_acc', 0):.2f}%")
            print(f"   Best Spoof AUC: {summary['best_metrics'].get('best_spoof_auc', 0):.4f}")
        else:
            print(f"📁 {exp_name} (incomplete)")
        print()


def view_experiment(exp_name, log_dir='logs/experiments'):
    """View detailed information about an experiment"""
    exp_path = Path(log_dir) / exp_name
    
    if not exp_path.exists():
        print(f" Experiment not found: {exp_name}")
        return
    
    # Load summary
    summary_path = exp_path / 'experiment_summary.json'
    if not summary_path.exists():
        print(f" Summary file not found for {exp_name}")
        return
    
    with open(summary_path, 'r') as f:
        summary = json.load(f)
    
    # Load full training log
    log_path = exp_path / 'training_log.json'
    if log_path.exists():
        with open(log_path, 'r') as f:
            full_log = json.load(f)
    else:
        full_log = None
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"Experiment: {exp_name}")
    print(f"{'='*80}\n")
    
    print(f" Timeline:")
    print(f"   Start: {summary.get('start_time', 'N/A')}")
    print(f"   End:   {summary.get('end_time', 'N/A')}")
    print(f"   Total Epochs: {summary.get('total_epochs', 'N/A')}")
    
    print(f"\n Best Metrics:")
    best = summary.get('best_metrics', {})
    print(f"   Best Train Accuracy: {best.get('best_train_acc', 0):.2f}%")
    print(f"   Best Val Accuracy:   {best.get('best_val_acc', 0):.2f}%")
    print(f"   Best Spoof AUC:      {best.get('best_spoof_auc', 0):.4f}")
    print(f"   Best Val Loss:       {best.get('best_val_loss', float('inf')):.4f}")
    print(f"   Best Epoch:          {best.get('best_epoch', 'N/A')}")
    
    print(f"\n Final Performance:")
    print(f"   Final Train Acc: {summary.get('final_train_acc', 0):.2f}%")
    print(f"   Final Val Acc:   {summary.get('final_val_acc', 0):.2f}%")
    
    # Show last 5 epochs
    if full_log and 'epochs' in full_log:
        epochs = full_log['epochs']
        print(f"\n Last 5 Epochs:")
        print(f"   {'Epoch':<8} {'Train Acc':<12} {'Val Acc':<12} {'Val Loss':<12} {'LR':<12}")
        print(f"   {'-'*56}")
        
        for epoch in epochs[-5:]:
            epoch_num = epoch['epoch']
            train_acc = epoch['train'].get('cls_acc', 0)
            val_acc = epoch['validation'].get('cls_acc', 0)
            val_loss = epoch['validation'].get('loss', 0)
            lr = epoch.get('learning_rate', 0)
            
            print(f"   {epoch_num:<8} {train_acc:<12.2f} {val_acc:<12.2f} {val_loss:<12.4f} {lr:<12.2e}")
    
    # Show config highlights
    if 'config' in summary:
        config = summary['config']
        print(f"\n  Configuration Highlights:")
        print(f"   Batch Size:    {config.get('BATCH_SIZE', 'N/A')}")
        print(f"   Learning Rate: {config.get('LEARNING_RATE', 'N/A')}")
        print(f"   Epochs:        {config.get('EPOCHS', 'N/A')}")
        print(f"   Device:        {config.get('DEVICE', 'N/A')}")
        print(f"   Use Mesh:      {config.get('USE_MESH', 'N/A')}")
        print(f"   Center Loss:   {config.get('USE_CENTER_LOSS', 'N/A')}")
    
    print(f"\n{'='*80}\n")


def compare_experiments(exp_names, log_dir='logs/experiments'):
    """Compare multiple experiments"""
    print(f"\n{'='*100}")
    print(f"Experiment Comparison")
    print(f"{'='*100}\n")
    
    # Table header
    header = f"{'Experiment':<30} {'Best Val Acc':<15} {'Spoof AUC':<15} {'Epochs':<10} {'Device':<10}"
    print(header)
    print('-' * 100)
    
    results = []
    for exp_name in exp_names:
        exp_path = Path(log_dir) / exp_name
        summary_path = exp_path / 'experiment_summary.json'
        
        if not summary_path.exists():
            print(f"{' ' + exp_name:<30} {'N/A':<15} {'N/A':<15} {'N/A':<10} {'N/A':<10}")
            continue
        
        with open(summary_path, 'r') as f:
            summary = json.load(f)
        
        best_val_acc = summary['best_metrics'].get('best_val_acc', 0)
        spoof_auc = summary['best_metrics'].get('best_spoof_auc', 0)
        total_epochs = summary.get('total_epochs', 0)
        device = summary.get('config', {}).get('DEVICE', 'N/A')
        
        results.append({
            'name': exp_name,
            'val_acc': best_val_acc,
            'spoof_auc': spoof_auc,
            'epochs': total_epochs,
            'device': device
        })
        
        print(f"{exp_name:<30} {best_val_acc:<15.2f} {spoof_auc:<15.4f} {total_epochs:<10} {device:<10}")
    
    # Find best
    if results:
        best_acc = max(results, key=lambda x: x['val_acc'])
        best_auc = max(results, key=lambda x: x['spoof_auc'])
        
        print(f"\n{'='*100}")
        print(f" Best Val Accuracy:  {best_acc['name']} ({best_acc['val_acc']:.2f}%)")
        print(f" Best Spoof AUC:     {best_auc['name']} ({best_auc['spoof_auc']:.4f})")
        print(f"{'='*100}\n")


def plot_experiment(exp_name, log_dir='logs/experiments', save=True):
    """Plot training curves for an experiment"""
    try:
        from logger import plot_training_curves
    except ImportError:
        print(" Cannot import plot_training_curves from logger.py")
        return
    
    exp_path = Path(log_dir) / exp_name
    log_path = exp_path / 'training_log.json'
    
    if not log_path.exists():
        print(f" Training log not found: {log_path}")
        return
    
    if save:
        save_path = exp_path / 'training_curves.png'
    else:
        save_path = None
    
    print(f"\n Plotting training curves for {exp_name}...")
    plot_training_curves(str(log_path), save_path=str(save_path) if save_path else None)
    
    if save:
        print(f"✓ Saved plot to {save_path}")


def export_to_csv(exp_name, log_dir='logs/experiments'):
    """Export training log to CSV"""
    import csv
    
    exp_path = Path(log_dir) / exp_name
    log_path = exp_path / 'training_log.json'
    
    if not log_path.exists():
        print(f" Training log not found: {log_path}")
        return
    
    with open(log_path, 'r') as f:
        data = json.load(f)
    
    csv_path = exp_path / 'training_log.csv'
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow([
            'epoch', 'train_loss', 'train_acc', 'train_spoof_auc',
            'val_loss', 'val_acc', 'val_spoof_auc', 'learning_rate', 'duration'
        ])
        
        # Data
        for epoch in data['epochs']:
            train = epoch['train']
            val = epoch['validation']
            
            train_spoof_auc = train.get('spoof_metrics', {}).get('auc', 0) if train.get('spoof_metrics') else 0
            val_spoof_auc = val.get('spoof_metrics', {}).get('auc', 0) if val.get('spoof_metrics') else 0
            
            writer.writerow([
                epoch['epoch'],
                train.get('loss', 0),
                train.get('cls_acc', 0),
                train_spoof_auc,
                val.get('loss', 0),
                val.get('cls_acc', 0),
                val_spoof_auc,
                epoch.get('learning_rate', 0),
                epoch.get('duration_seconds', 0)
            ])
    
    print(f"✓ Exported to CSV: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description='View and analyze training logs')
    parser.add_argument('--list', action='store_true', help='List all experiments')
    parser.add_argument('--exp', type=str, help='View specific experiment')
    parser.add_argument('--compare', nargs='+', help='Compare multiple experiments')
    parser.add_argument('--plot', type=str, help='Plot training curves for experiment')
    parser.add_argument('--export-csv', type=str, help='Export experiment to CSV')
    parser.add_argument('--log-dir', type=str, default='logs/experiments', help='Log directory')
    
    args = parser.parse_args()
    
    if args.list:
        list_experiments(args.log_dir)
    elif args.exp:
        view_experiment(args.exp, args.log_dir)
    elif args.compare:
        compare_experiments(args.compare, args.log_dir)
    elif args.plot:
        plot_experiment(args.plot, args.log_dir)
    elif args.export_csv:
        export_to_csv(args.export_csv, args.log_dir)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()