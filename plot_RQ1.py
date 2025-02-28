import matplotlib.pyplot as plt

import numpy as np

# Define data
categories = ['TRV', 'LRV', 'MV-dt', 'STL-dt', 'STL-enum']
measures = ['Accuracy', 'Precision', 'Recall', 'F1-score']

# Data for each category
## TRAFFIC-CONES CASE STUDY 
TRV = [1-0.07, 0.81, 0.96, 0.83]#
LRV = [1-0.08, 0.71, 0.87, 0.73]
MV_dt = [1-0.055, 1, 0.04, 0.08]
STL_dt = [1-0.058, 0, 0, 0]
STL_enum = [1-0.06, 0.33, 0.04, 0.07]


# Combine data into a 2D array for easy indexing
data = np.array([TRV, LRV, MV_dt, STL_dt, STL_enum]).T

# Set up figure and parameters for the grouped bar plot
num_measures = len(measures)
num_categories = len(categories)
bar_width = 0.29
x = np.arange(num_measures) * 2  # Increased spacing between groups

# Plot each category as a separate set of bars
fig, ax = plt.subplots(figsize=(15, 8))  # Larger figure size for bigger bars

for i in range(num_categories):
    bars = ax.bar(x + i * bar_width, data[:, i], width=bar_width, label=categories[i])
    
    # Adding value labels above each bar with matching colors
    for bar in bars:
        yval = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2, yval + 0.02, f"{yval:.2f}", 
            ha="center", va="bottom", fontsize=15, color=bar.get_facecolor()
        )

# Add labels and title with larger font sizes
ax.set_xlabel('Measures', fontsize=23)
ax.set_ylabel('Values', fontsize=23)
ax.set_xticks(x + (num_categories - 1) * bar_width / 2)
ax.set_xticklabels(measures)
plt.xticks(fontsize=23)
plt.yticks(fontsize=23)

# Place the legend outside of the plot on the right
plt.legend(fontsize=16, loc="upper right")

# Adjust layout and show plot
plt.tight_layout()
plt.savefig('TC_bar_means.pdf')
plt.show()


#LEAD FOLLOWER CASE STUDIES 
TRV = [ 1- 0.05, 0.95, 0.77, 0.84]
LRV = [ 1- 0.05, 0.95, 0.77, 0.84]
MV_dt = [ 1- 0.17, 0, 0, 0]
STL_dt = [ 1- 0.17, 0, 0, 0]
STL_enum = [ 1- 0.2025, 0.125, 0.0289, 0.0469]



# Combine data into a 2D array for easy indexing
data = np.array([TRV, LRV, MV_dt, STL_dt, STL_enum]).T

# Set up figure and parameters for the grouped bar plot
num_measures = len(measures)
num_categories = len(categories)
bar_width = 0.29
x = np.arange(num_measures) * 2  # Increased spacing between groups

# Plot each category as a separate set of bars
fig, ax = plt.subplots(figsize=(15, 8))  # Larger figure size for bigger bars

for i in range(num_categories):
    bars = ax.bar(x + i * bar_width, data[:, i], width=bar_width, label=categories[i])
    
    # Adding value labels above each bar with matching colors
    for bar in bars:
        yval = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2, yval + 0.02, f"{yval:.2f}", 
            ha="center", va="bottom", fontsize=15, color=bar.get_facecolor()
        )

# Add labels and title with larger font sizes
ax.set_xlabel('Measures', fontsize=23)
ax.set_ylabel('Values', fontsize=23)
ax.set_xticks(x + (num_categories - 1) * bar_width / 2)
ax.set_xticklabels(measures)
plt.xticks(fontsize=23)
plt.yticks(fontsize=23)

# Place the legend outside of the plot on the right
plt.legend(fontsize=16, loc="upper right")

# Adjust layout and show plot
plt.tight_layout()
plt.savefig('LF_bar_means.pdf')
plt.show()

