import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 1. Load the cached NumPy files
print("Loading cached embeddings...")
X = np.load("./probe_cache_multi/X_Train_layer12.npy")  # High-dimensional representations
y = np.load("./probe_cache_multi/y_Train_layer12.npy")  # Labels mapped to integers (0, 1, 2)

# Optional: Downsample if the file is too massive for a quick local plot
# e.g., Take the first 5000 instances
if len(X) > 5000:
    indices = np.arange(5000)
    X = X[indices]
    y = y[indices]

# 2. Standardize the features (Mean=0, Variance=1)
# PCA is highly sensitive to variance scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Fit PCA down to 2 dimensions
print("Computing Principal Components...")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Calculate explained variance for your axis titles
explained_var = pca.explained_variance_ratio_ * 100
print(f"PC1 explains {explained_var[0]:.2f}% of the variance.")
print(f"PC2 explains {explained_var[1]:.2f}% of the variance.")

# 4. Format into a DataFrame for Plotting
label_mapping = {0: "Fact", 1: "Implication", 2: "Negation"}
plot_df = pd.DataFrame({
    "Principal Component 1": X_pca[:, 0],
    "Principal Component 2": X_pca[:, 1],
    "Logical Concept": [label_mapping[label] for label in y]
})

# 5. Render the Visualization
plt.figure(figsize=(5, 3.5), dpi=300)
sns.set_theme(style="whitegrid")

# Create a clean scatter plot with distinct colors
ax = sns.scatterplot(
    data=plot_df,
    x="Principal Component 1",
    y="Principal Component 2",
    hue="Logical Concept",
    palette={"Fact": "#4f46e5", "Implication": "#10b981", "Negation": "#ef4444"},
    alpha=0.6,
    s=25,
    edgecolor=None
)

# Customize title and labels with the exact variance math
plt.title("PCA of Llama 3.2 Output (Layer 12)", fontsize=14, fontweight='bold', pad=15)
plt.xlabel(f"PC1 ({explained_var[0]:.2f}% Explained Variance)", fontsize=11)
plt.ylabel(f"PC2 ({explained_var[1]:.2f}% Explained Variance)", fontsize=11)

# Position the legend cleanly
plt.legend(title="Logical Class", title_fontsize='11', loc='upper right', frameon=True)
plt.tight_layout()

# Save the plot for your thesis slides
plt.savefig("llama_pca_projection.png", dpi=300)
plt.show()