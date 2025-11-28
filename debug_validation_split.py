"""
Debug script to check validation split quality
"""
import sys
from config import config
from dataset import get_dataloaders

def analyze_split():
    print("="*70)
    print("VALIDATION SPLIT ANALYSIS")
    print("="*70)
    
    train_loader, val_loader, num_classes = get_dataloaders(config)
    
    # Analyze train set
    train_real = 0
    train_fake = 0
    train_identities = set()
    
    print("\n[1/2] Analyzing training set...")
    for batch_idx, (inputs, labels, is_spoof) in enumerate(train_loader):
        train_real += (is_spoof == 0).sum().item()
        train_fake += (is_spoof == 1).sum().item()
        train_identities.update(labels.tolist())
        
        if batch_idx % 10 == 0:
            print(f"  Batch {batch_idx}/{len(train_loader)}", end='\r')
    
    print(f"\nTrain Set:")
    print(f"  Total samples: {train_real + train_fake}")
    print(f"  Real: {train_real} ({train_real/(train_real+train_fake)*100:.1f}%)")
    print(f"  Fake: {train_fake} ({train_fake/(train_real+train_fake)*100:.1f}%)")
    print(f"  Unique identities: {len(train_identities)}")
    
    # Analyze validation set
    val_real = 0
    val_fake = 0
    val_identities = set()
    
    print("\n[2/2] Analyzing validation set...")
    for batch_idx, (inputs, labels, is_spoof) in enumerate(val_loader):
        val_real += (is_spoof == 0).sum().item()
        val_fake += (is_spoof == 1).sum().item()
        val_identities.update(labels.tolist())
        
        if batch_idx % 10 == 0:
            print(f"  Batch {batch_idx}/{len(val_loader)}", end='\r')
    
    print(f"\nValidation Set:")
    print(f"  Total samples: {val_real + val_fake}")
    print(f"  Real: {val_real} ({val_real/(val_real+val_fake)*100:.1f}%)" if val_real+val_fake > 0 else "  Real: 0")
    print(f"  Fake: {val_fake} ({val_fake/(val_real+val_fake)*100:.1f}%)" if val_real+val_fake > 0 else "  Fake: 0")
    print(f"  Unique identities: {len(val_identities)}")
    
    # Check for issues
    print("\n" + "="*70)
    print("DIAGNOSTIC:")
    print("="*70)
    
    issues = []
    
    # Issue 1: Validation has only one class
    if val_real == 0 or val_fake == 0:
        issues.append("CRITICAL: Validation set has only ONE class!")
        if val_fake == 0:
            issues.append("   → No fake samples in validation")
        else:
            issues.append("   → No real samples in validation")
    else:
        issues.append("Validation has both real and fake samples")
    
    # Issue 2: Class imbalance
    if val_real > 0 and val_fake > 0:
        ratio = max(val_real, val_fake) / min(val_real, val_fake)
        if ratio > 5:
            issues.append(f"WARNING: Severe class imbalance (ratio {ratio:.1f}:1)")
        elif ratio > 2:
            issues.append(f"Moderate class imbalance (ratio {ratio:.1f}:1)")
        else:
            issues.append(f"Good class balance (ratio {ratio:.1f}:1)")
    
    # Issue 3: Too few validation samples
    if val_real + val_fake < 50:
        issues.append(f"WARNING: Very few validation samples ({val_real + val_fake})")
    else:
        issues.append(f"Sufficient validation samples ({val_real + val_fake})")
    
    # Issue 4: Identity overlap
    overlap = train_identities & val_identities
    if len(overlap) > 0:
        issues.append(f"Identity overlap is expected: {len(overlap)} shared IDs")
    
    for issue in issues:
        print(issue)
    
    print("="*70)
    
    # Recommendation
    if val_real == 0 or val_fake == 0:
        print("\n SOLUTION:")
        print("Your FAKE dataset might be too small or improperly labeled.")
        print("Check:")
        print("  1. Are files in FAKE folders properly named with 'spoof'?")
        print("  2. Do you have enough fake samples (min 50+)?")
        print("  3. Try: ls -R /Volumes/WD\\ 500GB\\ EL/DATA_ROOT/FAKE_RENDER/ | grep -i spoof")

if __name__ == "__main__":
    try:
        analyze_split()
    except Exception as e:
        print(f"\n ERROR: {e}")
        import traceback
        traceback.print_exc()