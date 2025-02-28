import os
import sys
import numpy as np
import pandas as pd
import copy
import time
import random
import scipy

import functions.create_grammar as gramm
import functions.syntax_eval as fun_eval
from functions.data_reader import replace_unknown_values

# Define an object called Trace with two attributes: variables and features
class Trace:
    def __init__(self, execution, features):
        self.execution = copy.deepcopy(execution)
        self.features =copy.deepcopy(features)



def structure_traces_lead_followerd(data_supervisor, path_to_data , index_trace, trim_traces):

    '''Structure the traces for the lead-follower case study.'''

    system_level_formula = 'always( (pi[0][9] > 5) and (pi[0][9] < 15) )'


    data_lead = pd.read_csv(f"{path_to_data}data_lead/yellow_line_angle_{index_trace}.csv")
    data_follower = pd.read_csv(f"{path_to_data}data_follower/black_car_angle_{index_trace}.csv")

    input_signature = ['time','ego_x','ego_y','ego_heading','ego_vel_lin','lead_x','lead_y','lead_heading','lead_vel_lin',\
                   'distance_true','direction','color_dist_from_black','dist_obst_follower','dist_obst_lead','ob_pos_x','ob_pos_y']

    execution = [] #List of lists (i.e. list of variables)
    for variable_name in input_signature: # 0 - 15 
        execution.append(list(data_supervisor[f"{variable_name}"]))

    mode = 'nearest_neighbor'
    unknown = 99999.99
    # Add variable yellow_line_angle # 16
    yellow_line_angle = replace_unknown_values(list(data_lead["yellow_line_angle"]), unknown, mode = mode)
    execution.append(yellow_line_angle)

    # Add variable black_car_angle # 17 
    black_car_angle = replace_unknown_values(list(data_follower["black_car_angle"]),unknown,  mode = mode)
    execution.append(black_car_angle)

    # Add variable distance_sensors # 18
    execution.append(replace_unknown_values(list(data_follower["distance_sensors"]),unknown,  mode = mode))
    
    # Add number of black pixels # 19
    execution.append(replace_unknown_values(list(data_follower["num_pixel_black"]),unknown,  mode = mode))


    # Add derivative of yellow_line_angle # 20 
    execution.append([0] + [ (yellow_line_angle[i+1] - yellow_line_angle[i]) \
                              for i in range(len(yellow_line_angle)-1) ])

    # Add derivative of black_line_angle # 21
    # execution.append([0] + [ (black_car_angle[i+1] - black_car_angle[i]) \
    #                         for i in range(len(black_car_angle)-1) ])
    execution.append([0]*10 + [ black_car_angle[i+10] - black_car_angle[i] for i in range(len(black_car_angle)-10) ])
    
    # Add How many agglomerates of black pixels are there? # 22
    execution.append(replace_unknown_values(list(data_follower["agglomerate_count"]),unknown,  mode = mode))

    # Add size of the biggest agglomerate of black pixels # 23
    execution.append(replace_unknown_values(list(data_follower["agglomerate_size_1"]),unknown,  mode = mode))

    # Add size of the second biggest agglomerate of black pixels # 24
    execution.append(replace_unknown_values(list(data_follower["agglomerate_size_2"]),unknown,  mode = mode))

    # Add the derivative of the number of black pixels # 25
    # execution.append([0] + [ (data_follower["num_pixel_black"][i+1] - data_follower["num_pixel_black"][i]) \
    #                         for i in range(len(data_follower["num_pixel_black"])-1) ])
    # Mean of the number of black pixels over 50 time steps
    execution.append([0]* 50 + [(data_follower["num_pixel_black"][i+50]- data_follower["num_pixel_black"][i]) \
                      for i in range(len(data_follower["num_pixel_black"])-50)])
    


    #Cut traces to the same length
    min_length = min([len(execution[var]) for var in range(len(execution))])
    max_length = max([len(execution[var]) for var in range(len(execution))])
    if min_length != max_length: 
        # print(f'Warning: traces have different lengths. Min length: {min_length}, max length: {max_length}')
        for var in range(len(execution)): execution[var] = execution[var][:min_length]

    # Determine the outcome of the trace
    outcome, _ = fun_eval.evaluate_rob_boolean( [execution], system_level_formula )

    ## Cut for prediction reasons
    if min_length > trim_traces:
        for var in range(len(execution)): execution[var] = execution[var][:len(execution[var])-trim_traces]
    else:
        print(f'It was not possible to remove {trim_traces} time steps for trace {index_trace} \n')

    return execution, outcome

def CollectData(folder_name, trim_traces , indices_traces):

    '''
    numb_pos = number of positive traces to be collected. 
    numb_neg = number of negative traces to be collected.

    In both cases : 70% of them will be used for training and 30% for testin
                    If there are less traces in the folder, all of them will be used.

    bool_trim_traces is used to say whether we want to remove the last {trim_traces} time units of the trace or not. 
      For predicitons (and learning) yes, but not when evaluating the system level spec'''
    
    initial_path = f'{folder_name}/'

    traces_pos, traces_neg = [] ,[]

    #Positive and Negative traces are in different folders
    for case in ['pos', 'neg' ] :
        path_supervisor = f'{initial_path}data_supervisor_{case}/'
        #Loop over different executions (different csv files)
        # Collecting max_numb traces for each case (pos or neg)
        for index_trace in indices_traces: # dataset_name in list_dataset[:max_numb]:

            # number_dataset = int(dataset_name[ 7 : -4]) #start after 'traces_' to catch {sample_ind}, then remove .csv'
            if not os.path.exists(f"{path_supervisor}traces_{index_trace}.csv"): continue
            data_supervisor = pd.read_csv(f"{path_supervisor}traces_{index_trace}.csv")

            execution, outcome = structure_traces_lead_followerd(data_supervisor, initial_path , index_trace, trim_traces)
            trace = execution
            
            if   outcome == 'sat': traces_pos.append(trace)
            elif outcome == 'unsat': traces_neg.append(trace) 

    return traces_pos, traces_neg



def CollectData_specific_trace(path, index_trace , case, trim_traces ):

    '''case = 'pos' or 'neg' depending on whether the trace satisfy the system-level specification or not
     
      bool_trim_traces is used to say whether we want to remove the last {trim_traces} time units of the trace or not. 
      For predicitons (and learning) yes, but not when evaluating the system level spec'''
    
    if not os.path.exists(f"{path}data_supervisor_{case}/traces_{index_trace}.csv"): return None , None
    data_supervisor = pd.read_csv(f"{path}data_supervisor_{case}/traces_{index_trace}.csv")

    execution, outcome = structure_traces_lead_followerd(data_supervisor, path , index_trace, trim_traces)

    return execution, outcome

def compute_variables_ranges_lead_follower(traces):

    '''Compute the range of values for each variable in the traces. : max and min values for each variable, excluding 99999.99 values.'''

    variables_ranges = []

    for var_num in range(len(traces[0])):
        if var_num == 10: #10 is the index for variable 'direction' which is a string -> skip this preprocessing step
            variables_ranges.append([0, 0])
            continue
        # consider all values exclude values 99999.99
        no_unknown = [trace[var_num][time_step] for trace in traces for time_step in range(len(trace[var_num])) if trace[var_num][time_step] != 99999.99]
       
        if len(no_unknown) == 0: #only 99999.99 values
            variables_ranges.append([0, 99999.99])
            print(f'\n\nVariable {var_num} only has 99999.99 values!')
            continue
        variables_ranges.append([min(no_unknown), max(no_unknown)])
        # Print warning if variable has the same value for all time steps in all traces
        if variables_ranges[var_num][0] == variables_ranges[var_num][1]: 
           print(f'\n\nWarning: variable {var_num} has the same value for all time steps in all traces. Value: {variables_ranges[var_num][0]}\n\n')
           
    return variables_ranges


def normalize_variables_traces_lead_follower(variables_ranges, input_traces):

    '''Normalize the values of the variables in the traces'''
    
    variables_denom = [ (variables_ranges[var_num][1] - variables_ranges[var_num][0]) for var_num in range(len(variables_ranges)) ]
    normalized_traces = copy.deepcopy(input_traces)
    for trace in normalized_traces:
        for var_num in range(len(trace)):
            if variables_denom[var_num] != 0: # Avoid division by zero
                for time_step in range(len(trace[var_num])):
                    if (var_num != 10) and (trace[var_num][time_step]!= 99999.99):
                        trace[var_num][time_step] = (trace[var_num][time_step] - variables_ranges[var_num][0])/ variables_denom[var_num]
        
    return normalized_traces

def scale_variables_formula_lead_follower(formula_string, variable_ranges):
    
    '''Scale the numerical values in the formula_string according to the variable_ranges.
    From normalized values to actual variable ranges '''

    #There is no predicate  
    if 'pi' not in formula_string: 
        print('In data_lead_follower.py, the formula does not have a predicate ??')
        return formula_string

    list_of_name_variables = []
    aux_formula_string = formula_string

    #Find indices of variables in formula_string (in the same order as in the formula_string)
    while 'pi[0]' in aux_formula_string:
        index_start = aux_formula_string.find('pi[0][')
        index_end = aux_formula_string[index_start + 6 :].find(']') + index_start + 6
        list_of_name_variables.append(int(aux_formula_string[index_start + 6 : index_end]))
        aux_formula_string = aux_formula_string[index_end + 1 :]

    scaled_string = formula_string
    #Find numerical values in formula_string
    _ , indices_removed = gramm.replace_numbers(formula_string)
    if len(indices_removed) != len(list_of_name_variables): sys.exit("Error: number of variables and number of values do not match in 'scale_variables_formula_lead_follower' function in data_lead_follower.py")
    if len(indices_removed) < 1: return formula_string
    for index_r in range(len(indices_removed)):
        index = list_of_name_variables[index_r]
        value_to_be_scaled = float(scaled_string[indices_removed[index_r][0]:indices_removed[index_r][1]+1])

        # Exclude values 99999.99 
        if value_to_be_scaled == 99999.99: continue

        new_value = (value_to_be_scaled * (variable_ranges[index][1] - variable_ranges[index][0] )) + variable_ranges[index][0]
        new_scaled_string = scaled_string[:indices_removed[index_r][0]] + str(new_value) + scaled_string[indices_removed[index_r][1] + 1 :]
        diff_char = len(scaled_string) - len(new_scaled_string)
        # Shift accoring to the difference in length of the string
        for index_update_r in range(index_r + 1, len(indices_removed)):
            indices_removed[index_update_r][0] -= diff_char
            indices_removed[index_update_r][1] -= diff_char
        scaled_string = new_scaled_string

    return scaled_string

def renormalize_traces_lead_follower(traces,  variables_to_be_changed, old_variable_ranges, new_variable_ranges):

    '''The variables in traces vary in the interval [0,1] but referring to real values that vary in old_variables_ranges.
    The function rescales the values in traces in [0,1] such that they refer to the new_variable_ranges.
      
      The variables that need to be changed are those in variables_to_be_changed,
      while those that do not need to be changed are skipped. '''

    old_variables_denom = [ (old_variable_ranges[var_num][1] - old_variable_ranges[var_num][0]) for var_num in range(len(old_variable_ranges)) ]
    new_variables_denom = [ (new_variable_ranges[var_num][1] - new_variable_ranges[var_num][0]) for var_num in range(len(new_variable_ranges)) ]
    rescaled_traces = copy.deepcopy(traces)
    for var_num in variables_to_be_changed: # Skip all the variables that do not need to be changed
        for trace in rescaled_traces:
            for time_step in range(len(trace[var_num])):
                if (var_num != 10) and (trace[var_num][time_step]!= 99999.99) \
                  and (new_variables_denom[var_num] != 0) :
                    # First: scale the value from range [0,1] to the actual value in range old_variable_ranges
                    aux = (trace[var_num][time_step] * old_variables_denom[var_num]) + old_variable_ranges[var_num][0]
                    # Second: normalize in [0,1] according to the new_variable_ranges
                    trace[var_num][time_step] = (aux - new_variable_ranges[var_num][0])/ new_variables_denom[var_num]

    return rescaled_traces


def normalize_features_tobemined_lead_follower(features_ranges, input_traces):

    '''Normalize the values of the features in the lead - follower traces'''
    
    variables_denom = [ (features_ranges[var_num][1] - features_ranges[var_num][0]) for var_num in range(6) ]
    variables_denom += [0,1] # For the categorical variable 'ind' ([6] feature)
    variables_denom += [ (features_ranges[var_num][1] - features_ranges[var_num][0]) for var_num in range(7, len(features_ranges)) ]
    normalized_traces = copy.deepcopy(input_traces)
    for trace in normalized_traces:
        for var_num in range(len(trace)):
            if (var_num != 6): # 'ind' feature is categorical (string) and should not be normalized
                for time_step in range(len(trace[var_num])):
                    trace[var_num][time_step] = (trace[var_num][time_step] - features_ranges[var_num][0])/ variables_denom[var_num]
        
    return normalized_traces




def rescale_features_lead_follower(formula_string, variable_ranges):
    
    '''Scale the numerical values in the formula_string according to the variable_ranges.
    '''

    #not a predicate node   
    if 'pi' not in formula_string: return formula_string

    list_of_name_variables = []
    aux_formula_string = formula_string

    #Find indices of variables in formula_string (in the same order as in the formula_string)
    while 'pi[0]' in aux_formula_string:
        index_start = aux_formula_string.find('pi[0][')
        index_end = aux_formula_string[index_start + 6 :].find(']') + index_start + 6
        list_of_name_variables.append(int(aux_formula_string[index_start + 6 : index_end]))
        aux_formula_string = aux_formula_string[index_end + 1 :]

    scaled_string = formula_string
    #Find numerical values in formula_string
    _ , indices_removed = gramm.replace_numbers(formula_string)
    if len(indices_removed) != len(list_of_name_variables): sys.exit("Error: number of variables and number of values do not match in 'rescale_features_lead_follower' function in data_lead_follower.py")
    if len(indices_removed) < 1: return formula_string
    for index_r in range(len(indices_removed)):
        index = list_of_name_variables[index_r]
        value_to_be_scaled = float(scaled_string[indices_removed[index_r][0]:indices_removed[index_r][1]+1])

        new_value = (value_to_be_scaled * (variable_ranges[index][1] - variable_ranges[index][0] )) + variable_ranges[index][0]
        new_scaled_string = scaled_string[:indices_removed[index_r][0]] + str(new_value) + scaled_string[indices_removed[index_r][1] + 1 :]
        diff_char = len(scaled_string) - len(new_scaled_string)
        # Shift accoring to the difference in length of the string
        for index_update_r in range(index_r + 1, len(indices_removed)):
            indices_removed[index_update_r][0] -= diff_char
            indices_removed[index_update_r][1] -= diff_char
        scaled_string = new_scaled_string

    return scaled_string


def convert_table(path, case, feature_names):

    '''Convert error table or safe table to the feature table named , respectively ,
     pos_table.csv or neg_table.csv'''

    if case == 'neg': table = pd.read_csv(f'{path}error_table.csv')
    elif case == 'pos': table = pd.read_csv(f'{path}safe_table.csv')
    
    # For every row in table
    new_table = []
    for row in table.iterrows():
        indices  = [ 1, 2, 3, 22, 23, 24, 21 ] # indices of the features in the table (ego, lead, ind)
        if row[1][21] == 0: # Left
            indices = indices + [6,7,8,9,10]
        elif row[1][21] == 1: # Straight
            indices = indices + [11, 12, 13, 14, 15]
        elif row[1][21] == 2: # Right
            indices = indices + [16, 17, 18, 19, 20]
        
        new_row = [ row[1][index] for index in indices]
        new_table.append(new_row)
    
    df = pd.DataFrame(new_table, columns = feature_names)
    df.to_csv(f'{path}{case}_table.csv', index = False)
    return 


