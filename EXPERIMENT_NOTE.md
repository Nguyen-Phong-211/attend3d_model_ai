PREPARE

============================================================
Model Architecture:
  Total parameters: 52,482,453
  Trainable parameters: 52,482,453
  Embedding dimension: 512
  Number of identities: 3491
  ArcFace (s=64.0, m=0.5)
  Use mesh: True
  Use attention fusion: True
============================================================


Trainer initialized:
  Device: cuda
  Optimizer: AdamW (lr=0.0001, wd=0.0001)
  Scheduler: cosine
  Mixed Precision: False
  Gradient Clipping: 1.0
  Early Stopping Patience: 10

======================================================================
Starting Training
  Epochs: 50
  Device: cuda
  Num Classes: 3491
  Train Batches: 435
  Val Batches: 91
======================================================================

EPOCH 1:  (LR: 9.76e-05)

Train:
  Loss: 34.1698
  Loss: 34.1698
  Classification:
  Classification:
    - Accuracy: 1.75%
    - Loss: 34.1698
  Anti-Spoofing:
    - AUC: 0.9508
    - Accuracy: 90.11%
    - APCER (fake→real): 13.84%
    - BPCER (real→fake): 8.75%
    - EER: 11.29%
    - F1: 0.7960
    - Loss: 0.2937

Validation:
  Loss: 29.9145
  Classification:
    - Accuracy: 16.08%
    - F1 Score: 0.1115
  Anti-Spoofing:
    - AUC: nan
    - Accuracy: 98.04%
    - APCER (fake→real): 0.00%
    - BPCER (real→fake): 1.96%
    - EER: 0.98%
    - F1: 0.0000
  Time: 2724.7s
======================================================================
  ✓ Saved: checkpoint_e1.pth
  ✓ Saved: best_acc.pth
  ★ New best accuracy: 16.08%


EPOCH 2/50 (LR: 9.05e-05)

Train:
  Loss: 29.0002
  Classification:
    - Accuracy: 12.90%
    - Loss: 29.0002
  Anti-Spoofing:
    - AUC: 0.9827
    - Accuracy: 94.99%
    - APCER (fake→real): 8.64%
    - BPCER (real→fake): 3.97%
    - EER: 6.30%
    - F1: 0.8908
    - Loss: 0.1525

Validation:
  Loss: 24.5754
  Classification:
    - Accuracy: 24.52%
    - F1 Score: 0.2337
  Anti-Spoofing:
    - AUC: nan
    - Accuracy: 96.56%
    - APCER (fake→real): 0.00%
    - BPCER (real→fake): 3.44%
    - EER: 1.72%
    - F1: 0.0000
  Time: 2727.6s
======================================================================
  ✓ Saved: checkpoint_e2.pth
  ✓ Saved: best_acc.pth
  ★ New best accuracy: 24.52%


EPOCH 3: (LR: 7.96e-05)

Train:
  Loss: 25.7127
  Classification:
    - Accuracy: 20.93%
    - Loss: 25.7127
  Anti-Spoofing:
    - AUC: 0.9895
    - Accuracy: 96.42%
    - APCER (fake→real): 6.71%
    - BPCER (real→fake): 2.68%
    - EER: 4.70%
    - F1: 0.9209
    - Loss: 0.1121

Validation:
  Loss: 21.2308
  Classification:
    - Accuracy: 31.92%
    - F1 Score: 0.3046
  Anti-Spoofing:
    - AUC: nan
    - Accuracy: 97.93%
    - APCER (fake→real): 0.00%
    - BPCER (real→fake): 2.07%
    - EER: 1.03%
    - F1: 0.0000
  Time: 2726.7s
======================================================================
  ✓ Saved: checkpoint_e3.pth
  ✓ Saved: best_acc.pth
  ★ New best accuracy: 31.92%


EPOCH 4: (LR: 6.58e-05)

Train:
  Loss: 23.1643
  Classification:
    - Accuracy: 25.66%
    - Loss: 23.1643
  Anti-Spoofing:
    - AUC: 0.9939
    - Accuracy: 97.17%
    - APCER (fake→real): 5.71%
    - BPCER (real→fake): 2.00%
    - EER: 3.86%
    - F1: 0.9371
    - Loss: 0.0864

Validation:
  Loss: 19.1822
  Classification:
    - Accuracy: 36.71%
    - F1 Score: 0.3394
  Anti-Spoofing:
    - AUC: nan
    - Accuracy: 98.21%
    - APCER (fake→real): 0.00%
    - BPCER (real→fake): 1.79%
    - EER: 0.90%
    - F1: 0.0000
  Time: 2728.5s
======================================================================
  ✓ Saved: checkpoint_e4.pth
  ✓ Saved: best_acc.pth
  ★ New best accuracy: 36.71%


======================================================================
Epoch 5/50 (LR: 5.05e-05)

Train:
  Loss: 21.2616
  Classification:
    - Accuracy: 28.42%
    - Loss: 21.2616
  Anti-Spoofing:
    - AUC: 0.9957
    - Accuracy: 97.67%
    - APCER (fake→real): 5.07%
    - BPCER (real→fake): 1.55%
    - EER: 3.31%
    - F1: 0.9479
    - Loss: 0.0716

Validation:
  Loss: 16.1470
  Classification:
    - Accuracy: 38.02%
    - F1 Score: 0.3780
  Anti-Spoofing:
    - AUC: nan
    - Accuracy: 98.00%
    - APCER (fake→real): 0.00%
    - BPCER (real→fake): 2.00%
    - EER: 1.00%
    - F1: 0.0000
  Time: 2705.4s
======================================================================
  ✓ Saved: checkpoint_e5.pth
  ✓ Saved: best_acc.pth
  ★ New best accuracy: 38.02%


Epoch 6/50 (LR: 3.52e-05)

Train:
  Loss: 19.5815
  Classification:
    - Accuracy: 30.52%
    - Loss: 19.5815
  Anti-Spoofing:
    - AUC: 0.9975
    - Accuracy: 98.39%
    - APCER (fake→real): 3.34%
    - BPCER (real→fake): 1.11%
    - EER: 2.22%
    - F1: 0.9641
    - Loss: 0.0528

Validation:
  Loss: 14.2666
  Classification:
    - Accuracy: 39.63%
    - F1 Score: 0.3895
  Anti-Spoofing:
    - AUC: nan
    - Accuracy: 99.35%
    - APCER (fake→real): 0.00%
    - BPCER (real→fake): 0.65%
    - EER: 0.33%
    - F1: 0.0000
  Time: 2701.1s
======================================================================
  ✓ Saved: checkpoint_e6.pth
  ✓ Saved: best_acc.pth
  ★ New best accuracy: 39.63%

  Epoch 7/50 (LR: 2.14e-05)

Train:
  Loss: 18.1535
  Classification:
    - Accuracy: 31.65%
    - Loss: 18.1535
  Anti-Spoofing:
    - AUC: 0.9983
    - Accuracy: 98.86%
    - APCER (fake→real): 2.25%
    - BPCER (real→fake): 0.82%
    - EER: 1.54%
    - F1: 0.9746
    - Loss: 0.0426

Validation:
  Loss: 12.3332
  Classification:
    - Accuracy: 41.39%
    - F1 Score: 0.4040
  Anti-Spoofing:
    - AUC: nan
    - Accuracy: 99.66%
    - APCER (fake→real): 0.00%
    - BPCER (real→fake): 0.34%
    - EER: 0.17%
    - F1: 0.0000
  Time: 2635.0s
======================================================================
  ✓ Saved: checkpoint_e7.pth
  ✓ Saved: best_acc.pth
  ★ New best accuracy: 41.39%

Epoch 8/50 (LR: 1.05e-05)

Train:
  Loss: 17.0003
  Classification:
    - Accuracy: 32.43%
    - Loss: 17.0003
  Anti-Spoofing:
    - AUC: 0.9990
    - Accuracy: 99.18%
    - APCER (fake→real): 1.77%
    - BPCER (real→fake): 0.55%
    - EER: 1.16%
    - F1: 0.9817
    - Loss: 0.0325

Validation:
  Loss: 10.8232
  Classification:
    - Accuracy: 44.32%
    - F1 Score: 0.4262
  Anti-Spoofing:
    - AUC: nan
    - Accuracy: 99.69%
    - APCER (fake→real): 0.00%
    - BPCER (real→fake): 0.31%
    - EER: 0.15%
    - F1: 0.0000
  Time: 2671.8s
======================================================================
  ✓ Saved: checkpoint_e8.pth
  ✓ Saved: best_acc.pth
  ★ New best accuracy: 44.32%

Epoch 9/50 (LR: 3.42e-06)

Train:
  Loss: 16.2361
  Classification:
    - Accuracy: 32.81%
    - Loss: 16.2361
  Anti-Spoofing:
    - AUC: 0.9997
    - Accuracy: 99.48%
    - APCER (fake→real): 1.00%
    - BPCER (real→fake): 0.39%
    - EER: 0.69%
    - F1: 0.9883
    - Loss: 0.0236

Validation:
  Loss: 9.9335
  Classification:
    - Accuracy: 45.18%
    - F1 Score: 0.4365
  Anti-Spoofing:
    - AUC: nan  
    - Accuracy: 99.79%
    - APCER (fake→real): 0.00%
    - BPCER (real→fake): 0.21%
    - EER: 0.10%
    - F1: 0.0000
  Time: 2631.3s
======================================================================
  ✓ Saved: checkpoint_e9.pth
  ✓ Saved: best_acc.pth
  ★ New best accuracy: 45.18%

Epoch 28/50 (LR: 3.42e-06)

Train:
  Loss: 7.6529
  Classification:
    - Accuracy: 73.14%
    - Loss: 7.6529
  Anti-Spoofing:
    - AUC: 1.0000
    - Accuracy: 99.99%
    - APCER (fake→real): 0.00%
    - BPCER (real→fake): 0.01%
    - EER: 0.00%
    - F1: 0.9998
    - Loss: 0.0005

Validation:
  Loss: 1.9638
  Classification:
    - Accuracy: 99.76%
    - F1 Score: 0.9973
  Anti-Spoofing:
    - AUC: nan
    - Accuracy: 100.00%
    - APCER (fake→real): 0.00%
    - BPCER (real→fake): 0.00%
    - EER: 0.00%
    - F1: 0.0000
  Time: 2549.5s
======================================================================
  ✓ Saved: checkpoint_e28.pth

Epoch 29/50 (LR: 1.61e-06)

Train:
  Loss: 7.6385
  Classification:
    - Accuracy: 73.05%
    - Loss: 7.6385
  Anti-Spoofing:
    - AUC: 1.0000
    - Accuracy: 100.00%
    - APCER (fake→real): 0.00%
    - BPCER (real→fake): 0.00%
    - EER: 0.00%
    - F1: 1.0000
    - Loss: 0.0004

Validation:
  Loss: 1.9660
  Classification:
    - Accuracy: 99.76%
    - F1 Score: 0.9973
  Anti-Spoofing:
    - AUC: nan
    - Accuracy: 100.00%
    - APCER (fake→real): 0.00%
    - BPCER (real→fake): 0.00%
    - EER: 0.00%
    - F1: 0.0000
  Time: 2552.6s
======================================================================
  ✓ Saved: checkpoint_e29.pth

======================================================================
Epoch 30/50 (LR: 1.00e-04)

Train:
  Loss: 7.6273
  Classification:
    - Accuracy: 73.09%
    - Loss: 7.6273
  Anti-Spoofing:
    - AUC: 1.0000
    - Accuracy: 99.99%
    - APCER (fake→real): 0.03%
    - BPCER (real→fake): 0.00%
    - EER: 0.02%
    - F1: 0.9998
    - Loss: 0.0005

Validation:
  Loss: 1.9597
  Classification:
    - Accuracy: 99.86%
    - F1 Score: 0.9984
  Anti-Spoofing:
    - AUC: nan
    - Accuracy: 100.00%
    - APCER (fake→real): 0.00%
    - BPCER (real→fake): 0.00%
    - EER: 0.00%
    - F1: 0.0000
  Time: 2547.5s
======================================================================
  ✓ Saved: checkpoint_e30.pth

======================================================================
Epoch 31/50 (LR: 9.98e-05)

Train:
  Loss: 9.8519
  Classification:
    - Accuracy: 51.92%
    - Loss: 9.8519
  Anti-Spoofing:
    - AUC: 0.9989
    - Accuracy: 99.70%
    - APCER (fake→real): 0.61%
    - BPCER (real→fake): 0.21%
    - EER: 0.41%
    - F1: 0.9933
    - Loss: 0.0155

Validation:
  Loss: 2.5565
  Classification:
    - Accuracy: 95.45%
    - F1 Score: 0.9528
  Anti-Spoofing:
    - AUC: nan
    - Accuracy: 99.93%
    - APCER (fake→real): 0.00%
    - BPCER (real→fake): 0.07%
    - EER: 0.03%
    - F1: 0.0000
  Time: 2547.5s
======================================================================
  ✓ Saved: checkpoint_e31.pth

======================================================================
Epoch 32/50 (LR: 9.94e-05)

Train:
  Loss: 8.5807
  Classification:
    - Accuracy: 60.63%
    - Loss: 8.5807
  Anti-Spoofing:
    - AUC: 0.9995
    - Accuracy: 99.76%
    - APCER (fake→real): 0.39%
    - BPCER (real→fake): 0.20%
    - EER: 0.29%
    - F1: 0.9945
    - Loss: 0.0104

Validation:
  Loss: 2.0470
  Classification:
    - Accuracy: 99.00%
    - F1 Score: 0.9897
  Anti-Spoofing:
    - AUC: nan
    - Accuracy: 99.90%
    - APCER (fake→real): 0.00%
    - BPCER (real→fake): 0.10%
    - EER: 0.05%
    - F1: 0.0000
  Time: 2543.0s
======================================================================
  ✓ Saved: checkpoint_e32.pth

======================================================================
Epoch 33/50 (LR: 9.86e-05)

Train:
  Loss: 8.2382
  Classification:
    - Accuracy: 65.91%
    - Loss: 8.2382
  Anti-Spoofing:
    - AUC: 0.9986
    - Accuracy: 99.76%
    - APCER (fake→real): 0.45%
    - BPCER (real→fake): 0.18%
    - EER: 0.31%
    - F1: 0.9947
    - Loss: 0.0141

Validation:
  Loss: 2.1754
  Classification:
    - Accuracy: 98.69%
    - F1 Score: 0.9863
  Anti-Spoofing:
    - AUC: nan
    - Accuracy: 100.00%
    - APCER (fake→real): 0.00%
    - BPCER (real→fake): 0.00%
    - EER: 0.00%
    - F1: 0.0000
  Time: 2544.6s
======================================================================
  ✓ Saved: checkpoint_e33.pth

======================================================================
Epoch 34/50 (LR: 9.76e-05)

Train:
  Loss: 8.1155
  Classification:
    - Accuracy: 68.12%
    - Loss: 8.1155
  Anti-Spoofing:
    - AUC: 0.9994
    - Accuracy: 99.83%
    - APCER (fake→real): 0.19%
    - BPCER (real→fake): 0.16%
    - EER: 0.17%
    - F1: 0.9963
    - Loss: 0.0099

Validation:
  Loss: 2.0561
  Classification:
    - Accuracy: 99.48%
    - F1 Score: 0.9945
  Anti-Spoofing:
    - AUC: nan
    - Accuracy: 100.00%
    - APCER (fake→real): 0.00%
    - BPCER (real→fake): 0.00%
    - EER: 0.00%
    - F1: 0.0000
  Time: 2563.0s
======================================================================
  ✓ Saved: checkpoint_e34.pth



================================================================= NEW VERSION ===================================================================================

======================================================================
Epoch 1/50 (LR: 9.76e-05)

Train:
  Loss: 34.4128
  Classification:
    - Accuracy: 1.01%
    - Loss: 34.4128
  Anti-Spoofing:
    - AUC: 0.9452
    - Accuracy: 89.04%
    - APCER (fake→real): 13.83%
    - BPCER (real→fake): 10.14%
    - EER: 11.98%
    - F1: 0.7787
    - Loss: 0.2786

Validation:
  Loss: 33.2223
  Classification:
    - Accuracy: 13.05%
    - F1 Score: 0.0975
  Anti-Spoofing:
    - AUC: nan
    - Accuracy: 89.39%
    - APCER (fake→real): 0.00%
    - BPCER (real→fake): 10.61%
    - EER: 5.30%
    - F1: 0.0000
  Time: 4197.3s
======================================================================
  ✓ Saved: checkpoint_e1.pth
  ✓ Saved: best_acc.pth

======================================================================
Epoch 2/50 (LR: 9.05e-05)

Train:
  Loss: 31.2605
  Classification:
    - Accuracy: 7.16%
    - Loss: 31.2605
  Anti-Spoofing:
    - AUC: 0.9638
    - Accuracy: 92.39%
    - APCER (fake→real): 13.31%
    - BPCER (real→fake): 5.97%
    - EER: 9.64%
    - F1: 0.8360
    - Loss: 0.1937

Validation:
  Loss: 28.4355
  Classification:
    - Accuracy: 11.64%
    - F1 Score: 0.0797
  Anti-Spoofing:
    - AUC: nan
    - Accuracy: 83.61%
    - APCER (fake→real): 0.00%
    - BPCER (real→fake): 16.39%
    - EER: 8.20%
    - F1: 0.0000
  Time: 4149.5s
======================================================================
  ✓ Saved: checkpoint_e2.pth

======================================================================
Epoch 3/50 (LR: 7.96e-05)

Train:
  Loss: 29.1255
  Classification:
    - Accuracy: 12.21%
    - Loss: 29.1255
  Anti-Spoofing:
    - AUC: 0.9743
    - Accuracy: 93.89%
    - APCER (fake→real): 12.70%
    - BPCER (real→fake): 4.21%
    - EER: 8.46%
    - F1: 0.8648
    - Loss: 0.1629

Validation:
  Loss: 26.3371
  Classification:
    - Accuracy: 19.32%
    - F1 Score: 0.1210
  Anti-Spoofing:
    - AUC: nan
    - Accuracy: 74.24%
    - APCER (fake→real): 0.00%
    - BPCER (real→fake): 25.76%
    - EER: 12.88%
    - F1: 0.0000
  Time: 3895.2s
======================================================================
  ✓ Saved: checkpoint_e3.pth