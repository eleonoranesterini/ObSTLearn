import time
import pandas as pd
import numpy as np


import seaborn as sns
import matplotlib.pyplot as plt

from matplotlib.ticker import MaxNLocator

numb_exp = 10
size = [ 50 , 100, 150, 200 ]
proportions = [  0.05, 0.1 , 0.15 ]
horizons = [50, 100, 150]

value_base_f1_score = [0.9323308270676692, 0.9090909090909092, 0.6476190476190476, 0.660377358490566, 0.890625, 0.9558823529411764, 0.9323308270676692, 0.6476190476190476, 0.8837209302325583, 0.9402985074626865] 
# value_base_recall = [0.8985507246376812, 0.9420289855072463, 0.4927536231884058, 0.5072463768115942, 0.8260869565217391, 0.9420289855072463, 0.8985507246376812, 0.4927536231884058, 0.8260869565217391, 0.9130434782608695]
# value_base_precision = [0.96875, 0.8783783783783784, 0.9444444444444444, 0.9459459459459459, 0.9661016949152542, 0.9701492537313433, 0.96875, 0.9444444444444444, 0.95, 0.9692307692307692]
time_base = [52964.12783718109 , 70328.44760990143, 71107.64696264267, 55770.59610724449, 49449.84742593765, 61870.0241086483, 63241.81392288208, 54242.54557514191, 47272.126229047775, 50809.94619703293]


# Different proportions same size: first 0.05, then 0.1, then  0.14 (normal)
value_diff_prop_f1_score = [ [0.6728971962616822, 0.9558823529411764, 0.660377358490566, 0.9323308270676692, 0.9076923076923077, 0.9558823529411764, 0.9402985074626865, 0.9090909090909092, 0.6728971962616822, 0.6138613861386139] ,
                              [0.9420289855072463, 0.9076923076923077, 0.9062499999999999, 0.6666666666666666, 0.9160305343511451, 0.9558823529411764, 0.9323308270676692, 0.890625, 0.9558823529411764, 0.660377358490566] ,
                               value_base_f1_score ]
# value_diff_prop_recall = [ [0.5217391304347826, 0.9420289855072463, 0.5072463768115942, 0.8985507246376812, 0.855072463768116, 0.9420289855072463, 0.9130434782608695, 0.9420289855072463, 0.5217391304347826, 0.4492753623188406] ,
#                            [0.9420289855072463, 0.855072463768116, 0.8405797101449275, 0.5072463768115942, 0.8695652173913043, 0.9420289855072463, 0.8985507246376812, 0.8260869565217391, 0.9420289855072463, 0.5072463768115942] ,
#                            value_base_recall]
# value_diff_prop_precision = [ [0.9473684210526315, 0.9701492537313433, 0.9459459459459459, 0.96875, 0.9672131147540983, 0.9701492537313433, 0.9692307692307692, 0.8783783783783784, 0.9473684210526315, 0.96875] ,
#                               [0.9420289855072463, 0.9672131147540983, 0.9830508474576272, 0.9722222222222222, 0.967741935483871, 0.9701492537313433, 0.96875, 0.9661016949152542, 0.9701492537313433, 0.9459459459459459] ,
#                               value_base_precision ]
value_diff_prop_time = [ [53194.3929438591, 61223.78207874298,  56595.06815338135, 57823.14616417885,  65112.47718286514, 44471.13906145096,  55534.78371620178,  43873.859159469604, 67690.09403514862, 53824.23049020767],
                        [51251.05536317825, 50981.08186292648, 51543.556401729584, 54370.10843753815, 64209.707825660706, 62600.6410241127,  64319.707825660706, 55211.14034938812, 58836.215030908585, 64877.63947272301],
                          time_base]



# Different size random proportions: first 50, then 100 , then 150,  then 200 (normal)
value_diff_size_rand_prop_f1_score = [ [0.6078431372549019, 0.6728971962616822, 0.594059405940594, 0.6078431372549019, 0.9090909090909092, 0.6213592233009708, 0.6728971962616822, 0.921875, 0.8489208633093526, 0.6476190476190476] ,
                                       [ 0.2942430703624734, 0.8740740740740741, 0.6, 0.9558823529411764, 0.9323308270676692, 0.8965517241379309, 0.9014084507042253, 0.6346153846153846, 0.6542056074766355, 0.9090909090909092]   ,
                                       [0.9558823529411764, 0.660377358490566, 0.58, 0.9076923076923077, 0.6476190476190476, 0.9489051094890512, 0.9323308270676692, 0.9489051094890512, 0.8818897637795277, 0.9323308270676692] ,
                                      value_base_f1_score]
# value_diff_size_rand_prop_recall = [[0.4492753623188406, 0.5217391304347826, 0.43478260869565216, 0.4492753623188406, 0.9420289855072463, 0.463768115942029, 0.5217391304347826, 0.855072463768116, 0.855072463768116, 0.4927536231884058] ,
#                                     [1.0, 0.855072463768116, 0.43478260869565216, 0.9420289855072463, 0.8985507246376812, 0.9420289855072463, 0.927536231884058, 0.4782608695652174, 0.5072463768115942, 0.9420289855072463],
#                                 [0.9420289855072463, 0.5072463768115942, 0.42028985507246375, 0.855072463768116, 0.4927536231884058, 0.9420289855072463, 0.8985507246376812, 0.9420289855072463, 0.8115942028985508, 0.8985507246376812] ,
#                                      value_base_recall]
# value_diff_size_rand_prop_precision = [[0.9393939393939394, 0.9473684210526315, 0.9375, 0.9393939393939394, 0.8783783783783784, 0.9411764705882353, 0.9473684210526315, 1.0, 0.8428571428571429, 0.9444444444444444],
#                                         [0.1725, 0.8939393939393939, 0.967741935483871, 0.9701492537313433, 0.96875, 0.8552631578947368, 0.8767123287671232, 0.9428571428571428, 0.9210526315789473, 0.8783783783783784] ,
#                                     [0.9701492537313433, 0.9459459459459459, 0.9354838709677419, 0.9672131147540983, 0.9444444444444444, 0.9558823529411765, 0.96875, 0.9558823529411765, 0.9655172413793104, 0.96875]  ,
#                                        value_base_precision]
value_diff_size_rand_prop_time = [[ 13624.202483415604, 12121.19501376152,  11412.481880187988, 12823.67371225357, 14849.63854765892,  11644.736862182617, 11401.162071943283, 12304.095219612122, 13427.122241973877, 12820.248909950256],
                                   [ 19522.531714200974, 22687.04518175125, 34004.63700962067, 26345.157037973404, 26899.695798158646, 25163.83877968788, 26822.432949543, 28329.885142087936, 24699.68351840973, 23585.727315187454],
                                   [ 50144.02701497078, 46542.730831861496, 41496.09848880768, 37936.02001571655, 42538.23255825043, 44287.84572839737, 37618.35544013977, 42957.062307834625, 46075.56143593788, 38993.1008810997],
                                   time_base]

## Different horizons: first 50, then 100 (normal), then 150
value_diff_horizon_f1_score = [ [0.962962962962963, 0.660377358490566, 0.9635036496350365, 0.9558823529411764, 0.6728971962616822, 0.962962962962963, 0.9558823529411764, 0.9160305343511451, 0.660377358490566, 0.9558823529411764],
                               value_base_f1_score,
                               [0.9343065693430658, 0.9489051094890512, 0.660377358490566, 0.9090909090909092, 0.6476190476190476, 0.9242424242424243, 0.9323308270676692, 0.9558823529411764, 0.9323308270676692, 0.900763358778626]]

# value_diff_horizon_recall = [[0.9420289855072463, 0.5072463768115942, 0.9565217391304348, 0.9420289855072463, 0.5217391304347826, 0.9420289855072463, 0.9420289855072463, 0.8695652173913043, 0.5072463768115942, 0.9420289855072463],
#                              value_base_recall,
#                              [0.927536231884058, 0.9420289855072463, 0.5072463768115942, 0.9420289855072463, 0.4927536231884058, 0.8840579710144928, 0.8985507246376812, 0.9420289855072463, 0.8985507246376812, 0.855072463768116]]

# value_diff_horizon_precision = [ [0.9848484848484849, 0.9459459459459459, 0.9705882352941176, 0.9701492537313433, 0.9473684210526315, 0.9848484848484849, 0.9701492537313433, 0.967741935483871, 0.9459459459459459, 0.9701492537313433]  ,
#                                 value_base_precision,
#                                 [0.9411764705882353, 0.9558823529411765, 0.9459459459459459, 0.8783783783783784, 0.9444444444444444, 0.9682539682539683, 0.96875, 0.9701492537313433, 0.96875, 0.9516129032258065] ]
value_diff_horizon_time = [ [ 60657.80419778824, 50626.03344511986, 55791.05402135849, 54503.359788656235, 56435.91887593269,  56533.10512495041, 64853.64523124695, 52927.4490044117, 50932.790660858154, 65042.299266815186],
                           time_base,
                        [ 52912.798964738846, 56502.41941571236, 58986.034232378006, 57127.31408429146, 47889.52630162239, 54042.15654540062 , 52520.614600896835,  52936.015152454376 , 61160.373596429825, 56241.60578393936]]



f1_scores_imbalance = { "Imbalance": value_diff_prop_f1_score}
f1_scores_size = { "Batch Size": value_diff_size_rand_prop_f1_score}
f1_scores_horizon = { "Horizon": value_diff_horizon_f1_score}
times_imbalance = { "Imbalance": value_diff_prop_time}
times_size = { "Batch Size": value_diff_size_rand_prop_time}
times_horizon = { "Horizon": value_diff_horizon_time}


data_f1_imbalance = []
for _, groups in f1_scores_imbalance.items():
    for group_id, group_values in enumerate(groups):
        if group_id == 0: group_name = '0.05'
        elif group_id == 1: group_name = '0.1'
        elif group_id == 2: group_name = '0.15'
        for value in group_values:
            data_f1_imbalance.append({
                "Group": group_name,
                "Value": value
            })
df_f1_imbalance = pd.DataFrame(data_f1_imbalance)

data_time_imbalance = []
for _, groups in times_imbalance.items():
    for group_id, group_values in enumerate(groups):
        if group_id == 0: group_name = '0.05'
        elif group_id == 1: group_name = '0.1'
        elif group_id == 2: group_name = '0.15'
        for value in group_values:
            data_time_imbalance.append({
                "Group": group_name,
                "Value": value/3600
            })
df_time_imbalance = pd.DataFrame(data_time_imbalance)


data_f1_size = []
for _, groups in f1_scores_size.items():
    for group_id, group_values in enumerate(groups):
        if group_id == 0: group_name = '50'
        elif group_id == 1: group_name = '100'
        elif group_id == 2: group_name = '150'
        elif group_id == 3: group_name = '200'
        for value in group_values:
            data_f1_size.append({
                "Group": group_name,
                "Value": value
            })
df_f1_size = pd.DataFrame(data_f1_size)

data_time_size = []
for _, groups in times_size.items():
    for group_id, group_values in enumerate(groups):
        if group_id == 0: group_name = '50'
        elif group_id == 1: group_name = '100'
        elif group_id == 2: group_name = '150'
        elif group_id == 3: group_name = '200'
        for value in group_values:
            data_time_size.append({
                "Group": group_name,
                "Value": value/3600
            })
df_time_size = pd.DataFrame(data_time_size)


data_f1_horizon = []
for _, groups in f1_scores_horizon.items():
    for group_id, group_values in enumerate(groups):
        if group_id == 0: group_name = '50'
        elif group_id == 1: group_name = '100'
        elif group_id == 2: group_name = '150'
        for value in group_values:
            data_f1_horizon.append({
                "Group": group_name,
                "Value": value
            })
df_f1_horizon = pd.DataFrame(data_f1_horizon)

data_time_horizon = []
for _, groups in times_horizon.items():
    for group_id, group_values in enumerate(groups):
        if group_id == 0: group_name = '50'
        elif group_id == 1: group_name = '100'
        elif group_id == 2: group_name = '150'
        for value in group_values:
            data_time_horizon.append({
                "Group": group_name,
                "Value": value/3600
            })
df_time_horizon = pd.DataFrame(data_time_horizon)


fig, axes = plt.subplots(2, 3, figsize=(16, 8))  # 2 rows, 3 columns
fig, axes = plt.subplots(2, 3, figsize=(16, 8))  # 2 rows, 3 columns

# Generate some random data for demonstration
x_imbalance = [0.05, 0.1, 0.15]
x_size = [50, 100, 150, 200]
x_horizon = [50, 100, 150]
x_ensemble = [2, 4, 6, 8, 10, 15, 20]

LABEL_SIZE = 30

# Example plots for each cell in the 2x4 grid
sns.lineplot(data=df_f1_imbalance, errorbar='sd', x="Group", y="Value", ax=axes[0, 0], linewidth=3, marker = 'o')
sns.lineplot(data=df_f1_size, errorbar='sd',x="Group", y="Value", ax=axes[0, 1],linewidth=3, marker = 'o')
sns.lineplot(data=df_f1_horizon, errorbar='sd',x="Group", y="Value", ax=axes[0, 2], linewidth=3, marker = 'o')
# sns.lineplot(data=df_f1_ensemble, x="Group", y="Value", ax=axes[0, 3], linewidth=3, marker = 'o')

sns.lineplot(data=df_time_imbalance, errorbar='sd',x="Group", y="Value", ax=axes[1, 0],linewidth=3, marker = 'o').set_xlabel("Unsafe prop.", fontsize=LABEL_SIZE)
sns.lineplot(data=df_time_size,errorbar='sd',x="Group", y="Value",  ax=axes[1, 1], linewidth=3, marker = 'o').set_xlabel("Batch Size", fontsize=LABEL_SIZE)
sns.lineplot(data=df_time_horizon,errorbar='sd', x="Group", y="Value", ax=axes[1, 2], linewidth=3, marker = 'o').set_xlabel("Horizon", fontsize=LABEL_SIZE)
# sns.lineplot(data=df_time_ensemble, x="Group", y="Value", ax=axes[1, 3], linewidth=3, marker = 'o').set_xlabel("Ensemble Size", fontsize=LABEL_SIZE)

#Set bigger x ticks
axes[1, 0].tick_params(axis='x', labelsize=LABEL_SIZE)
axes[1, 1].tick_params(axis='x', labelsize=LABEL_SIZE)
axes[1, 2].tick_params(axis='x', labelsize=LABEL_SIZE)
# axes[1, 3].tick_params(axis='x', labelsize=LABEL_SIZE)
# Set bigger y ticks

desired_number_of_ticks = 5
axes[0, 0].yaxis.set_major_locator(MaxNLocator(nbins=desired_number_of_ticks))

axes[0, 0].tick_params(axis='y', labelsize=LABEL_SIZE)
axes[1, 0].tick_params(axis='y', labelsize=LABEL_SIZE)

for i in range(2):   # iterate through rows
    for j in range(3):   # iterate through columns
        if j == 0:
            if i==0:
                axes[i, j].set_ylabel("F1-score", fontsize=LABEL_SIZE)
            else:
                axes[i, j].set_ylabel("Time (h)", fontsize=LABEL_SIZE)
        if i <1:
            axes[i, j].set_xlabel("")
            axes[i, j].tick_params(labelbottom=False)
        if j > 0:
            axes[i, j].set_ylabel("")
            axes[i, j].tick_params(labelleft=False)


# Adjust layout and display
plt.tight_layout()
plt.savefig("LF_rq3.pdf")
plt.show()