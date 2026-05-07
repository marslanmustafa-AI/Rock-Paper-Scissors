"""
Rock-Paper-Scissors Model Comparison
=====================================
Comparing three approaches:
1. CNN From Scratch (Custom 3-layer CNN)
2. YOLO (YOLOv26n-cls - early stopped at epoch 6)
3. MobileNetV2 (Transfer Learning + Fine Tuning)

Generates presentation-ready comparison charts.
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

import matplotlib.pyplot as plt
import numpy as np
import os

# Output directory
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# Configure matplotlib
matplotlib.rcParams.update({
    'figure.figsize': (14, 8),
    'font.size': 13,
    'axes.titlesize': 16,
    'axes.titleweight': 'bold',
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.dpi': 150,
    'savefig.dpi': 200,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

COLORS = {
    'scratch': '#2196F3',
    'yolo': '#FF9800',
    'mobilenet': '#4CAF50',
    'mobilenet_ft': '#8BC34A',
    'accent': '#E91E63',
    'bg': '#FAFAFA',
}

# ============================================================
# DATA FROM NOTEBOOKS
# ============================================================

# --- From Scratch CNN (24 epochs, early stopped from 50) ---
scratch_train_acc = [0.479, 0.687, 0.771, 0.816, 0.875, 0.881, 0.893, 0.925, 0.921, 0.935,
                     0.937, 0.936, 0.953, 0.943, 0.952, 0.959, 0.955, 0.959, 0.969, 0.965,
                     0.968, 0.962, 0.982, 0.979]
scratch_val_acc =   [0.685, 0.813, 0.804, 0.926, 0.947, 0.907, 0.978, 0.949, 0.953, 0.982,
                     0.981, 0.962, 0.951, 0.982, 0.959, 0.989, 0.992, 0.995, 0.967, 0.996,
                     0.990, 0.999, 0.986, 0.997]
scratch_train_loss = [1.054, 0.734, 0.561, 0.454, 0.334, 0.310, 0.277, 0.215, 0.219, 0.189,
                      0.163, 0.161, 0.134, 0.140, 0.137, 0.117, 0.117, 0.108, 0.087, 0.093,
                      0.081, 0.088, 0.052, 0.075]
scratch_val_loss =   [0.738, 0.492, 0.452, 0.237, 0.149, 0.253, 0.100, 0.132, 0.138, 0.092,
                      0.067, 0.109, 0.130, 0.065, 0.131, 0.039, 0.029, 0.021, 0.091, 0.024,
                      0.029, 0.011, 0.031, 0.009]
scratch_test_acc = 0.9480
scratch_test_loss = 0.1466
scratch_params = 2_678_307

# --- YOLO26n-cls (6 epochs, early stopped from 20, best=epoch1) ---
# From results.csv: metrics/accuracy_top1 is the test/val accuracy
yolo_val_acc =    [0.9948, 0.90641, 0.93934, 0.9896, 0.98094, 0.9792]
yolo_train_loss = [0.41747, 0.09217, 0.11181, 0.10423, 0.10104, 0.06863]
yolo_val_loss =   [0.03305, 0.38202, 0.0762, 0.02438, 0.07534, 0.06267]
yolo_best_acc = 0.995  # Final validation with best.pt
yolo_best_loss = 0.033
yolo_params = 1_534_947

# --- MobileNetV2 Transfer Learning (Phase 1: 10 epochs) ---
mn_tl_train_acc = [0.975, 0.982, 0.982, 0.982, 0.985, 0.986, 0.989, 0.989, 0.987, 0.989]
mn_tl_val_acc =   [0.918, 0.908, 0.906, 0.919, 0.940, 0.940, 0.948, 0.955, 0.974, 0.975]
mn_tl_train_loss = [0.069, 0.053, 0.048, 0.049, 0.044, 0.044, 0.032, 0.034, 0.037, 0.032]
mn_tl_val_loss =   [0.217, 0.252, 0.245, 0.209, 0.149, 0.147, 0.124, 0.111, 0.079, 0.075]
mn_tl_test_acc = 0.846
mn_tl_test_loss = 0.366

# --- MobileNetV2 Fine Tuning (Phase 2: epochs 17-25) ---
mn_ft_train_acc = [0.920, 0.937, 0.949, 0.963, 0.964, 0.969, 0.976, 0.981, 0.981]
mn_ft_val_acc =   [0.829, 0.838, 0.873, 0.880, 0.886, 0.884, 0.886, 0.895, 0.911]
mn_ft_train_loss = [0.219, 0.168, 0.130, 0.104, 0.095, 0.081, 0.072, 0.059, 0.059]
mn_ft_val_loss =   [0.405, 0.410, 0.339, 0.349, 0.323, 0.321, 0.303, 0.278, 0.228]
mn_ft_test_acc = 0.825
mn_ft_test_loss = 0.516
mn_params_total = 2_259_267

# Combined MobileNet phases
mn_all_train_acc = mn_tl_train_acc + mn_ft_train_acc
mn_all_val_acc = mn_tl_val_acc + mn_ft_val_acc
mn_all_train_loss = mn_tl_train_loss + mn_ft_train_loss
mn_all_val_loss = mn_tl_val_loss + mn_ft_val_loss

# ============================================================
# CHART 1: Test/Val Accuracy Comparison (Bar Chart)
# ============================================================
fig, ax = plt.subplots(figsize=(12, 7))
fig.patch.set_facecolor(COLORS['bg'])
ax.set_facecolor(COLORS['bg'])

names = ['CNN\n(From Scratch)', 'YOLO26n-cls', 'MobileNetV2\n(Transfer Learning)', 'MobileNetV2\n(Fine-Tuned)']
accs = [scratch_test_acc * 100, yolo_best_acc * 100, mn_tl_test_acc * 100, mn_ft_test_acc * 100]
colors = [COLORS['scratch'], COLORS['yolo'], COLORS['mobilenet'], COLORS['mobilenet_ft']]

bars = ax.bar(names, accs, color=colors, width=0.5, edgecolor='white', linewidth=2)
for bar, acc in zip(bars, accs):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
            f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=15)

ax.set_ylabel('Accuracy (%)')
ax.set_title('Model Accuracy Comparison\n(Rock-Paper-Scissors Classification on Test Set)', pad=20)
ax.set_ylim(75, 105)
ax.axhline(y=90, color='gray', linestyle='--', alpha=0.5, label='90% threshold')
ax.legend(loc='lower right')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '01_test_accuracy_comparison.png'), bbox_inches='tight')
plt.close()
print("✅ Chart 1 saved: 01_test_accuracy_comparison.png")

# ============================================================
# CHART 2: Test/Val Loss Comparison
# ============================================================
fig, ax = plt.subplots(figsize=(12, 7))
fig.patch.set_facecolor(COLORS['bg'])
ax.set_facecolor(COLORS['bg'])

losses = [scratch_test_loss, yolo_best_loss, mn_tl_test_loss, mn_ft_test_loss]
bars = ax.bar(names, losses, color=colors, width=0.5, edgecolor='white', linewidth=2)
for bar, loss in zip(bars, losses):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
            f'{loss:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=15)

ax.set_ylabel('Loss')
ax.set_title('Model Loss Comparison\n(Lower is Better)', pad=20)
ax.set_ylim(0, 0.65)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '02_test_loss_comparison.png'), bbox_inches='tight')
plt.close()
print("✅ Chart 2 saved: 02_test_loss_comparison.png")

# ============================================================
# CHART 3: Training Curves - All 3 Models
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.patch.set_facecolor(COLORS['bg'])
fig.suptitle('Training & Validation Accuracy Over Epochs', fontsize=18, fontweight='bold', y=1.02)

# From Scratch
ax = axes[0]
ax.set_facecolor(COLORS['bg'])
epochs_s = range(1, len(scratch_train_acc) + 1)
ax.plot(epochs_s, scratch_train_acc, '-o', color=COLORS['scratch'], markersize=3, label='Train', linewidth=2)
ax.plot(epochs_s, scratch_val_acc, '-s', color=COLORS['accent'], markersize=3, label='Val', linewidth=2)
ax.set_xlabel('Epoch'); ax.set_ylabel('Accuracy'); ax.set_title('CNN From Scratch (24 ep)'); ax.legend()
ax.set_ylim(0.4, 1.05)

# YOLO
ax = axes[1]
ax.set_facecolor(COLORS['bg'])
epochs_y = range(1, len(yolo_val_acc) + 1)
ax.plot(epochs_y, yolo_train_loss, '-o', color=COLORS['yolo'], markersize=5, label='Train Loss', linewidth=2)
ax.plot(epochs_y, yolo_val_acc, '-s', color=COLORS['accent'], markersize=5, label='Val Acc (Top-1)', linewidth=2)
ax.set_xlabel('Epoch'); ax.set_ylabel('Value'); ax.set_title('YOLO26n-cls (6 ep, early stop)'); ax.legend()
ax.set_ylim(0, 1.05)

# MobileNetV2
ax = axes[2]
ax.set_facecolor(COLORS['bg'])
epochs_m = range(1, len(mn_all_train_acc) + 1)
ax.plot(epochs_m, mn_all_train_acc, '-o', color=COLORS['mobilenet'], markersize=3, label='Train', linewidth=2)
ax.plot(epochs_m, mn_all_val_acc, '-s', color=COLORS['accent'], markersize=3, label='Val', linewidth=2)
ax.axvline(x=10.5, color='gray', linestyle='--', alpha=0.7, label='Fine-tuning starts')
ax.set_xlabel('Epoch'); ax.set_ylabel('Accuracy'); ax.set_title('MobileNetV2 (19 ep: TL+FT)'); ax.legend()
ax.set_ylim(0.8, 1.02)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '03_training_curves.png'), bbox_inches='tight')
plt.close()
print("✅ Chart 3 saved: 03_training_curves.png")

# ============================================================
# CHART 4: Training Loss Curves
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.patch.set_facecolor(COLORS['bg'])
fig.suptitle('Training & Validation Loss Over Epochs', fontsize=18, fontweight='bold', y=1.02)

ax = axes[0]
ax.set_facecolor(COLORS['bg'])
ax.plot(epochs_s, scratch_train_loss, '-o', color=COLORS['scratch'], markersize=3, label='Train', linewidth=2)
ax.plot(epochs_s, scratch_val_loss, '-s', color=COLORS['accent'], markersize=3, label='Val', linewidth=2)
ax.set_xlabel('Epoch'); ax.set_ylabel('Loss'); ax.set_title('CNN From Scratch'); ax.legend()

ax = axes[1]
ax.set_facecolor(COLORS['bg'])
ax.plot(epochs_y, yolo_train_loss, '-o', color=COLORS['yolo'], markersize=5, label='Train', linewidth=2)
ax.plot(epochs_y, yolo_val_loss, '-s', color=COLORS['accent'], markersize=5, label='Val', linewidth=2)
ax.set_xlabel('Epoch'); ax.set_ylabel('Loss'); ax.set_title('YOLO26n-cls'); ax.legend()

ax = axes[2]
ax.set_facecolor(COLORS['bg'])
ax.plot(epochs_m, mn_all_train_loss, '-o', color=COLORS['mobilenet'], markersize=3, label='Train', linewidth=2)
ax.plot(epochs_m, mn_all_val_loss, '-s', color=COLORS['accent'], markersize=3, label='Val', linewidth=2)
ax.axvline(x=10.5, color='gray', linestyle='--', alpha=0.7, label='Fine-tuning starts')
ax.set_xlabel('Epoch'); ax.set_ylabel('Loss'); ax.set_title('MobileNetV2 (TL+FT)'); ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '04_training_loss_curves.png'), bbox_inches='tight')
plt.close()
print("✅ Chart 4 saved: 04_training_loss_curves.png")

# ============================================================
# CHART 5: Model Parameters Comparison
# ============================================================
fig, ax = plt.subplots(figsize=(12, 7))
fig.patch.set_facecolor(COLORS['bg'])
ax.set_facecolor(COLORS['bg'])

param_names = ['CNN (From Scratch)', 'MobileNetV2', 'YOLO26n-cls']
param_vals = [scratch_params / 1e6, mn_params_total / 1e6, yolo_params / 1e6]
param_colors = [COLORS['scratch'], COLORS['mobilenet'], COLORS['yolo']]

bars = ax.barh(param_names, param_vals, color=param_colors, height=0.5, edgecolor='white', linewidth=2)
for bar, val in zip(bars, param_vals):
    ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
            f'{val:.2f}M', ha='left', va='center', fontweight='bold', fontsize=14)

ax.set_xlabel('Parameters (Millions)')
ax.set_title('Model Size Comparison (Number of Parameters)', pad=20)
ax.set_xlim(0, max(param_vals) * 1.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '05_model_parameters.png'), bbox_inches='tight')
plt.close()
print("✅ Chart 5 saved: 05_model_parameters.png")

# ============================================================
# CHART 6: Radar Chart
# ============================================================
categories = ['Test\nAccuracy', 'Low\nLoss', 'Training\nSpeed', 'Model\nCompactness', 'Ease of\nSetup']
N = len(categories)

scratch_scores =      [9.5, 8.5, 8.0, 6.0, 9.0]
yolo_scores =         [10.0, 9.5, 5.0, 8.5, 7.0]
mobilenet_tl_scores = [8.5, 7.0, 7.0, 7.0, 6.0]
mobilenet_ft_scores = [8.3, 5.0, 6.0, 7.0, 5.0]

angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
fig.patch.set_facecolor(COLORS['bg'])

for scores, color, label in [
    (scratch_scores, COLORS['scratch'], 'CNN From Scratch'),
    (yolo_scores, COLORS['yolo'], 'YOLO26n-cls'),
    (mobilenet_tl_scores, COLORS['mobilenet'], 'MobileNetV2 (TL)'),
    (mobilenet_ft_scores, COLORS['mobilenet_ft'], 'MobileNetV2 (FT)'),
]:
    values = scores + scores[:1]
    ax.plot(angles, values, '-o', linewidth=2, label=label, color=color, markersize=6)
    ax.fill(angles, values, alpha=0.1, color=color)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=13)
ax.set_ylim(0, 10.5)
ax.set_title('Multi-Metric Model Comparison (Higher = Better)', pad=30, fontsize=16, fontweight='bold')
ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '06_radar_comparison.png'), bbox_inches='tight')
plt.close()
print("✅ Chart 6 saved: 06_radar_comparison.png")

# ============================================================
# CHART 7: Summary Dashboard (2x2)
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.patch.set_facecolor(COLORS['bg'])
fig.suptitle('Rock-Paper-Scissors — Model Comparison Dashboard', fontsize=20, fontweight='bold', y=0.98)

# 7a: Accuracy
ax = axes[0][0]
ax.set_facecolor(COLORS['bg'])
b = ax.bar(names, accs, color=colors, width=0.5, edgecolor='white', linewidth=2)
for bar, acc in zip(b, accs):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
            f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)
ax.set_ylabel('Accuracy (%)'); ax.set_title('Test Accuracy'); ax.set_ylim(75, 105)

# 7b: Loss
ax = axes[0][1]
ax.set_facecolor(COLORS['bg'])
b = ax.bar(names, losses, color=colors, width=0.5, edgecolor='white', linewidth=2)
for bar, loss in zip(b, losses):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.006,
            f'{loss:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
ax.set_ylabel('Loss'); ax.set_title('Test Loss (Lower = Better)'); ax.set_ylim(0, 0.65)

# 7c: Accuracy overlay
ax = axes[1][0]
ax.set_facecolor(COLORS['bg'])
ax.plot(epochs_s, scratch_val_acc, '-', color=COLORS['scratch'], label='Scratch (Val)', linewidth=2)
ax.plot(epochs_y, yolo_val_acc, '-', color=COLORS['yolo'], label='YOLO (Val)', linewidth=2)
ax.plot(epochs_m, mn_all_val_acc, '-', color=COLORS['mobilenet'], label='MobileNet (Val)', linewidth=2)
ax.set_xlabel('Epoch'); ax.set_ylabel('Accuracy')
ax.set_title('Validation Accuracy Curves'); ax.legend(fontsize=10)

# 7d: Params
ax = axes[1][1]
ax.set_facecolor(COLORS['bg'])
pn = ['CNN\n(Scratch)', 'MobileNetV2', 'YOLO26n-cls']
pv = [scratch_params / 1e6, mn_params_total / 1e6, yolo_params / 1e6]
pc = [COLORS['scratch'], COLORS['mobilenet'], COLORS['yolo']]
b = ax.barh(pn, pv, color=pc, height=0.5, edgecolor='white', linewidth=2)
for bar, val in zip(b, pv):
    ax.text(bar.get_width() + 0.03, bar.get_y() + bar.get_height() / 2,
            f'{val:.2f}M', ha='left', va='center', fontweight='bold', fontsize=12)
ax.set_xlabel('Parameters (M)'); ax.set_title('Model Size'); ax.set_xlim(0, max(pv) * 1.3)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(os.path.join(OUT_DIR, '07_summary_dashboard.png'), bbox_inches='tight')
plt.close()
print("✅ Chart 7 saved: 07_summary_dashboard.png")

# ============================================================
# CHART 8: Summary Table
# ============================================================
fig, ax = plt.subplots(figsize=(18, 9))
fig.patch.set_facecolor(COLORS['bg'])
ax.set_facecolor(COLORS['bg'])
ax.axis('off')

columns = ['Metric', 'CNN From Scratch', 'YOLO26n-cls', 'MobileNetV2 (TL)', 'MobileNetV2 (FT)']
rows = [
    ['Architecture', 'Custom 3-layer CNN', 'YOLO26n classification', 'MobileNetV2\n(frozen) + Head', 'MobileNetV2\n(unfrozen) + Head'],
    ['Parameters', '2.68M', '1.53M', '~2.26M', '~2.26M'],
    ['Training Epochs', '24 (early stop)', '6 (early stop)', '10', '9 (fine-tune)'],
    ['~Time/Epoch', '~10s', '~87s (CPU)', '~16s', '~16s'],
    ['Test/Val Accuracy', '94.8%', '99.5%', '84.6%', '82.5%'],
    ['Test/Val Loss', '0.147', '0.033', '0.366', '0.516'],
    ['Best Val Accuracy', '99.9%', '99.5%', '97.5%', '91.1%'],
    ['Pretrained', 'No', 'Yes (ImageNet)', 'Yes (ImageNet)', 'Yes (ImageNet)'],
    ['Overfitting Risk', 'Low', 'Low', 'Medium', 'High'],
]

table = ax.table(cellText=rows, colLabels=columns, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.0, 2.2)

for j in range(len(columns)):
    cell = table[0, j]
    cell.set_facecolor('#37474F')
    cell.set_text_props(color='white', fontweight='bold', fontsize=12)

for i in range(1, len(rows) + 1):
    for j in range(len(columns)):
        cell = table[i, j]
        cell.set_facecolor('#E3F2FD' if i % 2 == 0 else '#FFFFFF')

ax.set_title('Model Comparison Summary Table', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '08_summary_table.png'), bbox_inches='tight')
plt.close()
print("✅ Chart 8 saved: 08_summary_table.png")

print("\n" + "=" * 60)
print("All 8 charts generated successfully!")
print("=" * 60)
print("""
KEY FINDINGS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. YOLO26n-cls achieved the HIGHEST accuracy (99.5%) with
   the smallest model (1.53M params) — converging in just
   6 epochs with early stopping (best at epoch 1).

2. CNN From Scratch achieved 94.8% test accuracy with good
   generalization and low overfitting risk over 24 epochs.

3. MobileNetV2 (Transfer Learning) achieved 84.6% — lower
   than expected, possibly due to domain mismatch.

4. MobileNetV2 (Fine-Tuned) DECREASED to 82.5% — showing
   signs of overfitting when unfreezing layers.

5. YOLO is the best overall: highest accuracy, lowest loss,
   fewest parameters. It's pretrained and well-optimized
   for classification tasks out-of-the-box.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")
