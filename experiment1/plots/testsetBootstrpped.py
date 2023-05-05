import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import t
import matplotlib

matplotlib.rc('xtick', labelsize=11)
matplotlib.rc('ytick', labelsize=12)
# load dataframe
df = pd.read_csv("/media/jonas/B41ED7D91ED792AA/Arbeit_und_Studium/Kognitionswissenschaft/Semester_5/masterarbeit#/data_Code/experiment2/LSTMAttentionSmall_bootstraped_results.csv")
print(df.var())

ci_low_mse, ci_high_mse  = np.percentile(df["MSE"], [2.5, 97.5])
print(ci_low_mse, ci_high_mse)
ci_low_mae, ci_high_mae  = np.percentile(df["MAE"], [2.5, 97.5])
print(ci_low_mae, ci_high_mae)
mean_mse = df['MSE'].mean()
print(mean_mse)
mean_mae = df['MAE'].mean()
print(mean_mae)
"""
# create plot
colors = ['#1f77b4', '#ff7f0e']
labels = ['MSE', 'MAE']

fig, axs = plt.subplots(1, 2, figsize=(12, 8))

# Create histogram for MSE and MAE columns
sns.histplot(df['MSE'], ax=axs[0], kde=False, color=colors[0], label=labels[0])#, bins = 50)
sns.histplot(df['MAE'], ax=axs[1], kde=False, color=colors[1], label=labels[1])#, bins = 50)

# Calculate confidence intervals and mean for each column
ci_low_mse, ci_high_mse  = np.percentile(df["MSE"], [2.5, 97.5])
ci_low_mae, ci_high_mae  = np.percentile(df["MAE"], [2.5, 97.5])
mean_mse = df['MSE'].mean()
mean_mae = df['MAE'].mean()

# Add vertical lines for confidence interval and mean
axs[0].axvline(x=ci_low_mse, linestyle='--', color='grey')
axs[0].axvline(x=ci_high_mse, linestyle='--', color='grey')
axs[0].axvline(x=mean_mse, linestyle='-', color='black')
axs[1].axvline(x=ci_low_mae, linestyle='--', color='grey')
axs[1].axvline(x=ci_high_mae, linestyle='--', color='grey')
axs[1].axvline(x=mean_mae, linestyle='-', color='black')

# Add labels and title for each subplot
#axs[row][col].set_xlabel('')
axs[0].set_ylabel('Frequency', fontsize = 16)
axs[1].set_ylabel('Frequency', fontsize = 16)
axs[0].set_xlabel('MSE', fontsize = 16)
axs[1].set_xlabel('MAE', fontsize = 16)
#axs[r].set_title(names[i], fontsize = 16)




plt.tight_layout()
plt.savefig("bootstrapped.pdf", dpi = 1000)
plt.show()
"""
