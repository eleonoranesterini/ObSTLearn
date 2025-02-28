import pickle
import os
from matplotlib import lines
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns
import pandas as pd

from functions.final_evaluation import evaluation_single_formulas, evaluate_all_ensembles

def specific_plot(categories, values, kind_plot, title, path_start):
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

     plt.savefig(f"{path_start}/{title}_{kind_plot}.pdf", format='pdf')
     plt.close()
     return


horizon = 100
case_study= 'traffic_cones' #lead_follower
numb_experiments = 10
numb_formulas = 10
path_start = Path('ensemble_to_be_tested_in_here')


if case_study == 'lead_follower':
     from experiments.lead_follower.data_lead_follower import scale_variables_formula_lead_follower as scale_variable
     from experiments.lead_follower.data_lead_follower import CollectData
     
     name_folder_data_main = 'data/lead_follower/final_datasets/testing_main'
     indices_traces_main = list(range(1,201))
     name_folder_data_additional = 'data/lead_follower/final_datasets/testing_additional'
     indices_traces_additional = list(range(201,401))
  


elif case_study == 'traffic_cones':
     from experiments.traffic_cones.data_traffic_cones import scale_variables_formula_traffic_cones as scale_variable
     from experiments.traffic_cones.data_traffic_cones import CollectData


     name_folder_data_main = 'data/traffic_cones/final_datasets/testing_main'
     indices_traces_main = list(range(2001,2201))
     name_folder_data_additional = 'data/traffic_cones/final_datasets/testing_additional'
     indices_traces_additional = list(range(4201,4401))
     

msc = []
precision = []
recall = []
f1_score = []

#Collect data
pos_main, neg_main =  CollectData(f"{name_folder_data_main}", horizon , indices_traces_main )
pos_additional, neg_additional =  CollectData(f"{name_folder_data_additional}", horizon , indices_traces_additional )

if len(pos_main) * len(neg_main) * len(pos_additional) * len(neg_additional) == 0:
     print('Some set has no tracessss!!!!!!!')
     
# variables_ranges = compute_variables_ranges(positive_traces+negative_traces)
positive_traces_test = pos_main + pos_additional
negative_traces_test = neg_main + neg_additional

for formulas_folder in path_start.iterdir(): ## numb_experiments
     if not formulas_folder.is_dir():  continue
     print(f'Folder: {formulas_folder.name}')
     name_formulas_folder = formulas_folder.name 
     output_file_name = f'{path_start}/{name_formulas_folder}/testing.text'
     if os.path.exists(output_file_name): os.remove(output_file_name)
     with open(f'{output_file_name}','a') as f:
               f.write(f'Traces used for testing in {name_folder_data_main}: from {indices_traces_main[0]} to {indices_traces_main[-1]}')
               f.write(f'\nTraces used for testing in {name_folder_data_additional}: from {indices_traces_additional[0]} to {indices_traces_additional[-1]}')
               f.write(f'\nNumber of positive testing traces: {len(positive_traces_test)}')
               f.write(f'\nNumber of negative testing traces: {len(negative_traces_test)}\n\n')

     ## Find where the learned formulas are stored
     for folder in Path(f'{path_start}/{name_formulas_folder}/Learned_formulas/').iterdir():
          if folder.is_dir() and folder.name.startswith('learned_formula_seed'):
               path = f'{path_start}/{name_formulas_folder}/Learned_formulas/{folder.name}/'
               break

     learned_formulas_original = []
     for index_monitor in range(1, numb_formulas + 1 ):
          data = f'formula_to_be_stored{index_monitor}.obj'
          with open(f'{path}{data}', 'rb') as f:
               read = pickle.load(f)
               learned_formulas_original.append(read)
     print(f'Number of learned formulas: {len(learned_formulas_original)}')
     # Evaluate each formula seprately, that take average
     tp = []
     tn = []
     fp = []
     fn = []
     for index_formula, item in enumerate(learned_formulas_original):
          with open(f'{output_file_name}','a') as f:
               f.write(f'\n\n\n#####    {index_formula + 1}    #####')
          formula_string = scale_variable(item.string, item.variable_ranges)
          result = evaluation_single_formulas(formula_string, positive_traces_test, negative_traces_test, output_file_name, case_study, bool_features = False)
          tp.append(result[0])
          tn.append(result[1])
          fp.append(result[2])
          fn.append(result[3])

     MSC_av = []
     precision_av = []
     recall_av = []
     F1_score_av = []

     for index in range(len(tp)):
          MSC_av.append((fp[index]+fn[index])/(fp[index]+fn[index]+tp[index]+tn[index]))

          if fp[index] + tp[index] > 0: 
               p = (tp[index]/(fp[index]+tp[index]))
               precision_av.append(p)
          else : 
               precision_av.append(0)
               print('i no precision')

          if fn[index] + tp[index] > 0: 
               r = (tp[index]/(fn[index]+tp[index]))
               recall_av.append(r)
          else : 
               recall_av.append(0)
               print('i no recall')

          if (fp[index] + tp[index] > 0) and (fn[index] + tp[index] > 0) and (p+r)>0: 
               F1_score_av.append(2*(p*r)/(p+r))
          else: 
               print('i no f score')
               F1_score_av.append(0)

     AV = [np.mean(MSC_av), np.mean(precision_av), np.mean(recall_av), np.mean(F1_score_av)]

     with open(f'{output_file_name}','a') as f:
          f.write('\n\n#############################')
          f.write('\n### Average results     ######')
          f.write('\n#############################')
          f.write(f'\n\nnb true positives:{np.mean(tp)} +- {np.std(tp)}')
          f.write(f'\nnb true negatives:{np.mean(tn)} +- {np.std(tn)}')
          f.write(f'\nnb false positives:{np.mean(fp)} +- {np.std(fp)}')
          f.write(f'\nnb false negatives:{np.mean(fn)} +- {np.std(fn)}')
          f.write(f'\n MSCs:     {AV[0]}  +- {np.std(MSC_av)}')
          f.write(f'\nPrecision: {AV[1]}  +- {np.std(precision_av)}')
          f.write(f'\nRecall:    {AV[2]}  +- {np.std(recall_av)}')
          f.write(f'\nF1_score:  {AV[3]}  +- {np.std(F1_score_av)}')
          f.write('\n\n#############################')
          
     # Final evaluation
     MV , AR, LR = evaluate_all_ensembles(positive_traces_test, negative_traces_test, output_file_name, learned_formulas_original, case_study, bool_features = False)

     msc.append([AV[0], MV[0], AR[0], LR[0]])
     precision.append([AV[1], MV[1], AR[1], LR[1]])
     recall.append([AV[2], MV[2], AR[2], LR[2]])
     f1_score.append([AV[3], MV[3], AR[3], LR[3]])



if os.path.exists(f'{path_start}/results.txt'): os.remove(f'{path_start}/results.txt')
with open(f'{path_start}/results.txt','a') as f:
     f.write(f'MSC = {msc}')
     f.write(f'\n\nPrecision = {precision}')
     f.write(f'\n\nRecall = {recall}')
     f.write(f'\n\nF1_score = {f1_score}')

# Write average and startdar deviation for AV, MV, AR, LR
with open(f'{path_start}/results.txt','a') as f:
     f.write(f'\n\nAverage and standard deviation for AV, MV, AR, LR')
     f.write(f'\n\nMSC = {np.mean(msc, axis=0)} +- {np.std(msc, axis=0)}')
     f.write(f'\n\nPrecision = {np.mean(precision, axis=0)} +- {np.std(precision, axis=0)}')
     f.write(f'\n\nRecall = {np.mean(recall, axis=0)} +- {np.std(recall, axis=0)}')
     f.write(f'\n\nF1_score = {np.mean(f1_score, axis=0)} +- {np.std(f1_score, axis=0)}')

# Average     
for it_method in range(4):
     with open(f'{path_start}/results.txt','a') as f:
          if  it_method== 0: f.write(f'\n\n\n################### Average ###################')
          elif it_method == 1: f.write(f'\n\n\n################### Majority Voting  ###################')
          elif it_method ==2 : f.write(f'\n\n\n################### Total Robustness ###################')
          elif it_method ==3 : f.write(f'\n\n\n################### Largest Robustness ###################')
          f.write(f'\n\nMSC =  {[msc[i_mon][it_method] for i_mon in range(numb_experiments)]}  ')
          f.write(f'\n\nPrecision =  {[precision[i_mon][it_method] for i_mon in range(numb_experiments)]}  ')
          f.write(f'\n\nRecall =  {[recall[i_mon][it_method] for i_mon in range(numb_experiments)]}  ')
          f.write(f'\n\nF1_score =  {[f1_score[i_mon][it_method] for i_mon in range(numb_experiments)]}  ')

          f.write(f'\n\nMSC_average = [{np.mean([msc[i_mon][it_method] for i_mon in range(numb_experiments)])} , {np.std([msc[i_mon][it_method] for i_mon in range(numb_experiments)])} ]')
          f.write(f'\n\nPrecision_average = [{np.mean([precision[i_mon][it_method] for i_mon in range(numb_experiments)])} , {np.std([precision[i_mon][it_method] for i_mon in range(numb_experiments)])} ]')
          f.write(f'\n\nRecall_average = [{np.mean([recall[i_mon][it_method] for i_mon in range(numb_experiments)])} , {np.std([recall[i_mon][it_method] for i_mon in range(numb_experiments)])} ]')
          f.write(f'\n\nF1_score_average = [{np.mean([f1_score[i_mon][it_method] for i_mon in range(numb_experiments)])} , {np.std([f1_score[i_mon][it_method] for i_mon in range(numb_experiments)])} ]')

         
# ## Unconstrained

categories = ['AV', 'MV', 'TRV', 'LRV']

specific_plot(categories, msc, 'violin', 'MSC', path_start)
specific_plot(categories, precision, 'violin', 'Precision', path_start)
specific_plot(categories, recall, 'violin', 'Recall', path_start)
specific_plot(categories, f1_score, 'violin', 'F1_score', path_start)

specific_plot(categories, msc, 'box', 'MSC', path_start)
specific_plot(categories, precision, 'box', 'Precision', path_start)
specific_plot(categories, recall, 'box', 'Recall', path_start)
specific_plot(categories, f1_score, 'box', 'F1_score', path_start)


specific_plot(categories, msc, 'point', 'MSC', path_start)
specific_plot(categories, precision, 'point', 'Precision', path_start)
specific_plot(categories, recall, 'point', 'Recall', path_start)
specific_plot(categories, f1_score, 'point', 'F1_score', path_start)




