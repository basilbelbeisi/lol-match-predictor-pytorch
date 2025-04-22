"""
League of Legends Match Predictor using Logistic Regression in PyTorch
-----------------------------------------------------------------------

This script is designed for educational purposes to demonstrate a complete machine learning
workflow—from data loading to model training, evaluation, and interpretation.

For full explanation and step-by-step breakdown, visit the blog post:
🔗 https://dataskillblog.com/[ADD-YOUR-POST-LINK-HERE]

Website: https://dataskillblog.com

Author: [Your Name]
Date: April 2025
"""

# ------------------------------------
# 1. Import Required Libraries
# ------------------------------------
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc
)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ------------------------------------
# 2. Load the Dataset
# ------------------------------------
# The dataset contains match stats and a target column 'win' (1 for win, 0 for loss)
data = pd.read_csv("league_of_legends_data.csv")

# ------------------------------------
# 3. Prepare Features and Labels
# ------------------------------------
X = data.drop("win", axis=1)
y = data["win"]

# ------------------------------------
# 4. Split into Train/Test Sets (80/20)
# ------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------------------------
# 5. Standardize Features
# ------------------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ------------------------------------
# 6. Convert to PyTorch Tensors
# ------------------------------------
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.float32)
y_test = torch.tensor(y_test.values, dtype=torch.float32)

# ------------------------------------
# 7. Define Logistic Regression Model
# ------------------------------------
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_units):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_units, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

# Initialize model
model = LogisticRegressionModel(input_units=X_train.shape[1])
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=0.01)  # With L2 Regularization

# ------------------------------------
# 8. Train the Model
# ------------------------------------
loss_values = []
epochs = 1000

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train).squeeze()
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    loss_values.append(loss.item())

    if epoch % 100 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# ------------------------------------
# 9. Evaluate the Model
# ------------------------------------
model.eval()
with torch.no_grad():
    y_pred_probs = model(X_test).squeeze().numpy()
    y_pred_labels = (y_pred_probs >= 0.5).astype(int)
    y_true = y_test.numpy()

# ------------------------------------
# 10. Plot Training Loss
# ------------------------------------
plt.figure(figsize=(8, 5))
plt.plot(loss_values)
plt.title("Training Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.tight_layout()
plt.show()

# ------------------------------------
# 11. Confusion Matrix
# ------------------------------------
cm = confusion_matrix(y_true, y_pred_labels)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Loss", "Win"], yticklabels=["Loss", "Win"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

# ------------------------------------
# 12. Classification Report
# ------------------------------------
print("Classification Report:\n")
print(classification_report(y_true, y_pred_labels, target_names=["Loss", "Win"]))

# ------------------------------------
# 13. ROC Curve
# ------------------------------------
fpr, tpr, thresholds = roc_curve(y_true, y_pred_probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}", color="darkorange")
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()

# ------------------------------------
# 14. Feature Importance
# ------------------------------------
weights = model.linear.weight.data.numpy().flatten()
features = X.columns

importance_df = pd.DataFrame({
    'Feature': features,
    'Weight': weights
})
importance_df['AbsWeight'] = importance_df['Weight'].abs()
importance_df = importance_df.sort_values(by='AbsWeight', ascending=False)

plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Weight'], color='skyblue')
plt.xlabel("Weight (Importance)")
plt.title("Feature Importance in Logistic Regression")
plt.gca().invert_yaxis()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
