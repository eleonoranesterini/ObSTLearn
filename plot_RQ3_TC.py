import time
import pandas as pd
import numpy as np


import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

numb_exp = 10
size = [ 50 , 100, 150, 200 ]
proportions = [  0.025, 0.05 , 0.1 ]
horizons = [50, 100, 150]

# value_base_msc = [0.0025, 0.1475, 0.0025, 0.0025, 0.0075, 0.53, 0.005, 0.0025, 0.0025, 0.005] 
value_base_f1_score = [0.9777777777777777, 0.42718446601941745, 0.9777777777777777, 0.9777777777777777, 0.9387755102040816, 0.17187500000000003, 0.9565217391304348, 0.9777777777777777, 0.9777777777777777, 0.9583333333333334] 
# value_base_recall = [0.9565217391304348, 0.9565217391304348, 0.9565217391304348, 0.9565217391304348, 1.0, 0.9565217391304348, 0.9565217391304348, 0.9565217391304348, 0.9565217391304348, 1.0]  
# value_base_precision = [1.0, 0.275, 1.0, 1.0, 0.8846153846153846, 0.0944206008583691, 0.9565217391304348, 1.0, 1.0, 0.92]  
time_base = [ 3097.3502683639526, 5888.776739358902, 5067.260992765427, 4304.231835126877, 4805.861090660095, 5910.561047077179, 3999.279230594635, 5528.968587398529, 4801.5988302230835, 5092.57714176178]


# Different proportions same size: first 0.025, then normal (0.05), then 0.1
# value_diff_prop_msc = [ [0.0125, 0.0075, 0.3725, 0.005, 0.0025, 0.0175, 0.01, 0.005, 0.0025, 0.0025] ,
#                        value_base_msc ,
#                               [0.6975, 0.0025, 0.0025, 0.015, 0.9425, 0.0, 0.0025, 0.0025, 0.0025, 0.015]]
value_diff_prop_f1_score = [ [0.8979591836734695, 0.9387755102040816, 0.1945945945945946, 0.9565217391304348, 0.9777777777777777, 0.8679245283018869, 0.9166666666666666, 0.9565217391304348, 0.9777777777777777, 0.9777777777777777]  , 
                             value_base_f1_score ,
                             [0.14153846153846153, 0.9777777777777777, 0.9777777777777777, 0.8800000000000001, 0.10874704491725767, 1.0, 0.9777777777777777, 0.9777777777777777, 0.9777777777777777, 0.8800000000000001]]
# value_diff_prop_recall = [  [0.9565217391304348, 1.0, 0.782608695652174, 0.9565217391304348, 0.9565217391304348, 1.0, 0.9565217391304348, 0.9565217391304348, 0.9565217391304348, 0.9565217391304348],
#                            value_base_recall,
#                             [1.0, 0.9565217391304348, 0.9565217391304348, 0.9565217391304348, 1.0, 1.0, 0.9565217391304348, 0.9565217391304348, 0.9565217391304348, 0.9565217391304348]]
# value_diff_prop_precision = [ [0.8461538461538461, 0.8846153846153846, 0.1111111111111111, 0.9565217391304348, 1.0, 0.7666666666666667, 0.88, 0.9565217391304348, 1.0, 1.0] , 
#                              value_base_precision, 
#                               [0.076158940397351, 1.0, 1.0, 0.8148148148148148, 0.0575, 1.0, 1.0, 1.0, 1.0, 0.8148148148148148]  ]
value_diff_prop_time = [[ 3386.5550286769867, 4361.726647853851, 7958.918249130249, 6621.143291711807, 5833.42288684845, 4186.825787782669, 3605.526744365692, 5118.451622724533,  5629.677280426025, 3676.456120491028], 
                         time_base,
                        [ 5027.2758412361145, 6741.739621639252, 4794.025274038315, 5680.4872610569, 3755.68217921257, 4096.36304807663, 6142.669348478317, 6762.979708433151, 5661.084757566452, 5652.175506830215] ]


# Different size random proportions: first 50, then 100 , then 150,  then 200 (normal)
# value_diff_size_rand_prop_msc = [ [0.0025, 0.0025, 0.0025, 0.005, 0.0025, 0.0175, 0.0025, 0.0025, 0.9425, 0.0025],
#                                  [0.0025, 0.0075, 0.0025, 0.0075, 0.0025, 0.0575, 0.0025, 0.0025, 0.01, 0.0025]  ,
#                                  [0.0075, 0.0175, 0.0225, 0.0025, 0.0025, 0.0025, 0.01, 0.0025, 0.94, 0.0025] ,
#                                  value_base_msc]
value_diff_size_rand_prop_f1_score = [ [0.9777777777777777, 0.9777777777777777, 0.9777777777777777, 0.9545454545454545, 0.9777777777777777, 0.8627450980392156, 0.9777777777777777, 0.9777777777777777, 0.10874704491725767, 0.9777777777777777] ,
                                      [0.9777777777777777, 0.9387755102040816, 0.9777777777777777, 0.9387755102040816, 0.9777777777777777, 0, 0.9777777777777777, 0.9777777777777777, 0.92, 0.9777777777777777] ,
                                      [0.9361702127659574, 0.8679245283018869, 0.8363636363636363, 0.9777777777777777, 0.9777777777777777, 0.9777777777777777, 0.92, 0.9777777777777777, 0.10900473933649288, 0.9777777777777777],
                                      value_base_f1_score]
# value_diff_size_rand_prop_recall = [ [0.9565217391304348, 0.9565217391304348, 0.9565217391304348, 0.9130434782608695, 0.9565217391304348, 0.9565217391304348, 0.9565217391304348, 0.9565217391304348, 1.0, 0.9565217391304348],
#                                     [0.9565217391304348, 1.0, 0.9565217391304348, 1.0, 0.9565217391304348, 0.0, 0.9565217391304348, 0.9565217391304348, 1.0, 0.9565217391304348] ,
#                                     [0.9565217391304348, 1.0, 1.0, 0.9565217391304348, 0.9565217391304348, 0.9565217391304348, 1.0, 0.9565217391304348, 1.0, 0.9565217391304348] ,
#                                     value_base_recall]
# value_diff_size_rand_prop_precision = [[1.0, 1.0, 1.0, 1.0, 1.0, 0.7857142857142857, 1.0, 1.0, 0.0575, 1.0] ,
#                                        [1.0, 0.8846153846153846, 1.0, 0.8846153846153846, 1.0, 0, 1.0, 1.0, 0.8518518518518519, 1.0] ,
#                                        [0.9166666666666666, 0.7666666666666667, 0.71875, 1.0, 1.0, 1.0, 0.8518518518518519, 1.0, 0.05764411027568922, 1.0]  ,
#                                        value_base_precision]
value_diff_size_rand_prop_time = [[  687.0462553501129, 961.6356990337372,  686.3943703174591, 1374.3855142593384,  842.3002526760101, 1195.6802237033844, 791.7965681552887, 1141.143426656723, 739.4137804508209, 1255.9744274616241],
                                   [ 2134.637687921524, 2697.7245075702667, 3415.430344581604, 2697.3452525138855, 2681.490562438965,  2844.5388522148132, 2403.2662949562073,  2884.540610551834, 2740.4872448444366, 2734.5238699913025],
                                   [ 3364.5564107894897, 2363.1581020355225, 3678.181117773056, 2845.246646642685, 3768.9465067386627, 3123.6902277469635, 3922.1076991558075, 3152.4955582618713, 3958.6550991535187, 3813.8184480667114],
                                   time_base]

## Different horizons: first 50, then 100 (normal), then 150

# value_diff_horizon_msc = [    [0.0025, 0.0025, 0.0025, 0.0025, 0.0025, 0.0025, 0.0025, 0.0025, 0.0025, 0.0025] ,
#                             value_base_msc,
#                                [0.0025, 0.0025, 0.9425, 0.0025, 0.0025, 0.0025, 0.0025, 0.0025, 0.0025, 0.0025]]
value_diff_horizon_f1_score = [ [0.9777777777777777, 0.9777777777777777, 0.9777777777777777, 0.9777777777777777, 0.9777777777777777, 0.9777777777777777, 0.9777777777777777, 0.9777777777777777, 0.9777777777777777, 0.9777777777777777],
                                value_base_f1_score,
                              [0.9777777777777777, 0.9777777777777777, 0.10874704491725767, 0.9777777777777777, 0.9777777777777777, 0.9777777777777777, 0.9787234042553191, 0.9777777777777777, 0.9777777777777777, 0.9777777777777777] ]
# value_diff_horizon_recall = [ [0.9565217391304348, 0.9565217391304348, 0.9565217391304348, 0.9565217391304348, 0.9565217391304348, 0.9565217391304348, 0.9565217391304348, 0.9565217391304348, 0.9565217391304348, 0.9565217391304348] ,
#                              value_base_recall,
#                              [0.9565217391304348, 0.9565217391304348, 1.0, 0.9565217391304348, 0.9565217391304348, 0.9565217391304348, 1.0, 0.9565217391304348, 0.9565217391304348, 0.9565217391304348]  ]

# value_diff_horizon_precision = [[ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]  ,
#                                 value_base_precision,
#                                 [1.0, 1.0, 0.0575, 1.0, 1.0, 1.0, 0.9583333333333334, 1.0, 1.0, 1.0]   ]
value_diff_horizon_time = [[   4519.501486778259, 3444.7734479904175, 2365.7171518802643, 3019.9772799015045, 1473.2404282093048, 3404.089997768402,  3130.677490711212, 4830.34830904007, 2277.1011230945587, 4033.969134092331],
                           time_base,
                           [    4656.334563732147, 2094.629937171936, 3063.488666534424, 6033.433007955551, 3163.762579679489, 2128.58322429657, 2579.430002450943, 3180.768491268158, 4280.723210096359,  4700.5621173381805]]

f1_scores_imbalance = { "Unsafe prop.": value_diff_prop_f1_score}
f1_scores_size = { "Batch Size": value_diff_size_rand_prop_f1_score}
f1_scores_horizon = { "Horizon": value_diff_horizon_f1_score}
times_imbalance = { "Unsafe prop.": value_diff_prop_time}
times_size = { "Batch Size": value_diff_size_rand_prop_time}
times_horizon = { "Horizon": value_diff_horizon_time}

data_f1_imbalance = []
for _, groups in f1_scores_imbalance.items():
    for group_id, group_values in enumerate(groups):
        if group_id == 0: group_name = '0.025'
        elif group_id == 1: group_name = '0.05'
        elif group_id == 2: group_name = '0.1'
        for value in group_values:
            data_f1_imbalance.append({
                "Group": group_name,
                "Value": value
            })
df_f1_imbalance = pd.DataFrame(data_f1_imbalance)

data_time_imbalance = []
for _, groups in times_imbalance.items():
    for group_id, group_values in enumerate(groups):
        if group_id == 0: group_name = '0.025'
        elif group_id == 1: group_name = '0.05'
        elif group_id == 2: group_name = '0.1'
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

# Generate some random data for demonstration
x_imbalance = [0.025, 0.05, 0.1]
x_size = [50, 100, 150, 200]
x_horizon = [50, 100, 150]

LABEL_SIZE = 30

# Example plots for each cell in the 2x4 grid
sns.lineplot(data=df_f1_imbalance, x="Group", y="Value",errorbar='sd', ax=axes[0, 0], linewidth=3, marker = 'o')
sns.lineplot(data=df_f1_size, x="Group", y="Value",errorbar='sd', ax=axes[0, 1],linewidth=3, marker = 'o')
sns.lineplot(data=df_f1_horizon, x="Group", y="Value", errorbar='sd', ax=axes[0, 2], linewidth=3, marker = 'o')

sns.lineplot(data=df_time_imbalance, x="Group", y="Value",errorbar='sd',  ax=axes[1, 0],linewidth=3, marker = 'o').set_xlabel("Unsafe prop.", fontsize=LABEL_SIZE)
sns.lineplot(data=df_time_size, x="Group", y="Value", errorbar='sd',  ax=axes[1, 1], linewidth=3, marker = 'o').set_xlabel("Batch Size", fontsize=LABEL_SIZE)
sns.lineplot(data=df_time_horizon, x="Group", y="Value",errorbar='sd',  ax=axes[1, 2], linewidth=3, marker = 'o').set_xlabel("Horizon", fontsize=LABEL_SIZE)


#Set bigger x ticks
axes[1, 0].tick_params(axis='x', labelsize=LABEL_SIZE)
axes[1, 1].tick_params(axis='x', labelsize=LABEL_SIZE)
axes[1, 2].tick_params(axis='x', labelsize=LABEL_SIZE)

#Set more ticks
desired_number_of_ticks = 5
axes[0, 0].yaxis.set_major_locator(MaxNLocator(nbins=desired_number_of_ticks))


# Set bigger y ticks
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
plt.savefig("TC_rq3.pdf")
plt.show()