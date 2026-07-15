import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc

# ==========================================
# 1. LOAD CACHED HIGH-DIMENSIONAL DATA
# ==========================================
print("Loading cached representations...")
# Adjust filenames if you saved dev set instead of test set
X_train = np.load("./probe_cache_multi/X_Train_layer12.npy")
y_train = np.load("./probe_cache_multi/y_Train_layer12.npy")
X_test = np.load("./probe_cache_multi/X_Test_layer12.npy")
y_test = np.load("./probe_cache_multi/y_Test_layer12.npy")

classes = [0, 1, 2]
n_classes = len(classes)
label_names = {0: "Fact", 1: "Implication", 2: "Negation"}
colors = {0: "#4f46e5", 1: "#10b981", 2: "#ef4444"}

# ==========================================
# 2. TRAIN PROBE & GET SOFT PROBABILITIES
# ==========================================
print("Fitting optimized linear probe...")
probe = LogisticRegression(C=1.0, l1_ratio=0.0, max_iter=5000)
probe.fit(X_train, y_train)

# Crucial: We need continuous probability estimates, NOT hard predictions
# y_score shape: [N_samples, 3]
y_score = probe.predict_proba(X_test)

# Binarize the true labels for One-vs-Rest calculation
# Shape becomes: [N_samples, 3] of 0s and 1s
y_test_bin = label_binarize(y_test, classes=classes)

# ==========================================
# 3. COMPUTE ROC AND AUC FOR EACH CLASS
# ==========================================
fpr = {}
tpr = {}
roc_auc = {}

for i in classes:
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute Macro-Average ROC
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in classes]))

# Interpolate all ROC curves at these points
mean_tpr = np.zeros_like(all_fpr)
for i in classes:
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

# Average and compute AUC
mean_tpr /= n_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# ==========================================
# 4. PLOT MULTI-CLASS ROC CURVES
# ==========================================
plt.figure(figsize=(5, 3.5), dpi=300)

# Plot the diagonal baseline (chance)
plt.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random Guessing (AUC = 0.50)")

# Plot individual class OvR curves
for i in classes:
    plt.plot(
        fpr[i],
        tpr[i],
        color=colors[i],
        lw=2,
        label=f"Class: {label_names[i]} (AUC = {roc_auc[i]:.4f})"
    )

# Plot the overall macro average
plt.plot(
    fpr["macro"],
    tpr["macro"],
    color="#0f172a", # Dark Charcoal
    linestyle=":",
    lw=3,
    label=f"Macro-Average (AUC = {roc_auc['macro']:.4f})"
)

# Customizing layout for presentation slides
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=10, fontweight='semibold')
plt.ylabel("True Positive Rate (Sensitivity)", fontsize=10, fontweight='semibold')
plt.title("ROC Curve of Linear Probe", fontsize=12, fontweight="bold", pad=15)
plt.legend(loc="lower right", frameon=True, fontsize=9)
plt.grid(True, linestyle=":", alpha=0.5)
plt.tight_layout()

# Save the plot for your thesis results slide
plt.savefig("probe_multiclass_auroc.png", dpi=300)
plt.show()

print("AUROC visualization generated and saved successfully!")