import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import t

# Set random seed for reproducibility
np.random.seed(42)

# Create four dataframes with MSE and MAE columns
dfs = []
for i in range(4):
    mse = np.random.normal(5, 1, 200)
    mae = np.random.normal(2, 0.5, 200)
    dfs.append(pd.DataFrame({'MSE': mse, 'MAE': mae}))

# Create figure and subplots
fig, axs = plt.subplots(2, 2, figsize=(12, 8))
names = ["Transformer", "Vision Transformer", "Unet", "Encoder-Decoder LSTM"]
colors = ['#1f77b4', '#ff7f0e']
labels = ['MSE', 'MAE']

for i, df in enumerate(dfs):
    # Determine the row and column index for the current dataframe
    row = i // 2
    col = i % 2

    # Create histogram for MSE and MAE columns
    sns.histplot(df['MSE'], ax=axs[row][col], kde=False, color=colors[0], label=labels[0])
    sns.histplot(df['MAE'], ax=axs[row][col], kde=False, color=colors[1], label=labels[1])

    # Calculate confidence intervals and mean for each column
    ci_low_mse, ci_high_mse = ci_low, ci_high = np.percentile(df["MSE"], [2.5, 97.5])
    ci_low_mae, ci_high_mae = ci_low, ci_high = np.percentile(df["MAE"], [2.5, 97.5])
    mean_mse = df['MSE'].mean()
    mean_mae = df['MAE'].mean()

    # Add vertical lines for confidence interval and mean
    axs[row][col].axvline(x=ci_low_mse, linestyle='--', color='grey')
    axs[row][col].axvline(x=ci_high_mse, linestyle='--', color='grey')
    axs[row][col].axvline(x=mean_mse, linestyle='-', color='black')
    axs[row][col].axvline(x=ci_low_mae, linestyle='--', color='grey')
    axs[row][col].axvline(x=ci_high_mae, linestyle='--', color='grey')
    axs[row][col].axvline(x=mean_mae, linestyle='-', color='black')

    # Add labels and title for each subplot
    axs[row][col].set_xlabel('')
    axs[row][col].set_ylabel('Frequency', fontsize = 12)
    axs[row][col].set_title(names[i], fontsize = 16)

# Add a common legend for all subplots
handles, _ = axs[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right', fontsize = 12)

plt.tight_layout()
plt.savefig("bootstrapped.pdf", dpi = 1000)
plt.show()

