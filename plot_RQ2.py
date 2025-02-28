import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def specific_plot(categories, values, kind_plot, title):
     data = []
     for i in range(len(values)):
          for j, category in enumerate(categories):
               data.append({'Ensemble Type': category, f'{title}': values[i][j], 'Set': f'Set {i+1}'})
     df = pd.DataFrame(data)
     plt.figure(figsize=(8, 6))
     if kind_plot == 'violin':
          sns.violinplot(x='Ensemble Type', y=f'{title}', data=df, inner="quartile", cut=0) # inner = 'point'
          # plt.title(f'Violin Plot of {title}')
          plt.xticks(fontsize=18)  # X-axis tick labels
          plt.yticks(fontsize=18)  # Y-axis tick labels
          plt.xlabel('Ensemble Type', fontsize=20)
          plt.ylabel(f'{title}', fontsize=20)
     elif kind_plot == 'box':
          sns.boxplot(x='Ensemble Type', y=f'{title}', data=df, color='lightblue')
          plt.xticks(fontsize=25)  # X-axis tick labels
          plt.yticks(fontsize=25)  # Y-axis tick labels
          plt.xlabel('Ensemble Type', fontsize=25)
          plt.ylabel(f'{title}', fontsize=25)
          #Define size of the plot
          plt.gcf().set_size_inches(13, 8)
          # plt.title(f'Box Plot of {title}')
     elif kind_plot == 'heatmap':
          sns.heatmap(values, annot=True, cmap='YlGnBu', xticklabels=categories, yticklabels=[f'Set {i+1}' for i in range(len(values))])
          # plt.title(f'Heatmap of {title}')
     elif kind_plot == 'point':
          marker = ['*', 'o', 's', 'x', 'D', 'P', 'H', 'v', '^', '<', '>', '1', '2', '3', '4', '8', 'p', 'h', '+', 'X', 'd']
          for i in range(len(values)):
               # dashed
               plt.plot(categories, values[i], marker = marker[i], linestyle=':')
          # plt.title(f'{title}')
     plt.show()
     plt.savefig(f"{title}_{kind_plot}.pdf", format='pdf')
     plt.close()
     return



categories = ['AV', 'MV', 'TRV', 'LRV']
f1_score_LF = [[0.6108745440321892, 0.6476190476190476, 0.9323308270676692, 0.9323308270676692], [0.6537702870466516, 0.660377358490566, 0.9090909090909092, 0.9027777777777778], [0.5154796434450712, 0.660377358490566, 0.6476190476190476, 0.6476190476190476], [0.5733641427206373, 0.6538461538461539, 0.660377358490566, 0.6476190476190476], [0.6227785896478959, 0.6476190476190476, 0.890625, 0.8799999999999999], [0.6514008926700379, 0.6538461538461539, 0.9558823529411764, 0.9558823529411764], [0.42093543561623203, 0.5102040816326531, 0.9323308270676692, 0.9393939393939393], [0.5139152760758345, 0.6481481481481483, 0.6476190476190476, 0.6476190476190476], [0.5036128114281226, 0.6226415094339622, 0.8837209302325583, 0.890625], [0.5946578435561058, 0.6666666666666666, 0.9402985074626865, 0.9402985074626865]]
specific_plot(categories, f1_score_LF, 'box', 'F1_score')

f1_score_TC = [[0.553882077799844, 0.9565217391304348, 0.9777777777777777, 0.9777777777777777], [0.5486595747752164, 0.7333333333333334, 0.42718446601941745, 0.423076923076923], [0.5854437195224866, 0.8979591836734695, 0.9777777777777777, 0.9777777777777777], [0.7468260073260072, 0.9166666666666666, 0.9777777777777777, 0], [0.44645080396348985, 0.31654676258992803, 0.9387755102040816, 0.9387755102040816], [0.29294958204705285, 0.27329192546583847, 0.17187500000000003, 0.17120622568093388], [0.6050232596467764, 0.6176470588235294, 0.9565217391304348, 0.9565217391304348], [0.6271187982178045, 0.7586206896551724, 0.9777777777777777, 0.9565217391304348], [0.5620010549807264, 0.3387096774193548, 0.9777777777777777, 0.9777777777777777], [0.5191754748777598, 0.8518518518518519, 0.9583333333333334, 0.9565217391304348]]
specific_plot(categories, f1_score_TC, 'box', 'F1_score')