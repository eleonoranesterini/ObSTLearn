import os
import sys
import numpy as np
import pandas as pd
import copy
import colorsys
import random

import functions.create_grammar as gramm
import functions.syntax_eval as fun_eval
from functions.data_reader import replace_unknown_values


# Define an object called Trace with two attributes: variables and features
class Trace:
    def __init__(self, execution, features):
        self.execution = copy.deepcopy(execution)
        self.features  = copy.deepcopy(features)




def structure_data_traffic_cones(data_supervisor, path,  index_trace, trim_traces):

    input_signature = ['time','ego_x','ego_y', 'yaw', 'cone0_x', 'cone0_y', 'cone0_min_dist', 
                       'cone1_x', 'cone1_y', 'cone1_min_dist', 
                       'cone2_x', 'cone2_y', 'cone2_min_dist', 'speed_car', 'min_distance_cones']
    
    CIRCLE_CONE = 0.75
    system_level_formula = f'always ( pi[0][14] > {CIRCLE_CONE})'
    
    data_ego = pd.read_csv(f"{path}data_gaussian_cnn/traces_{index_trace}.csv")

    execution = [] #List of lists (i.e. list of variables)

    for variable_name in input_signature: # 0 - 14 
        execution.append(list(data_supervisor[f"{variable_name}"]))     
    # 15
    execution.append(list(data_ego["steering_angle"]))
    # 16 
    execution.append(list(data_ego["target_speed"]))
    mode = 'nearest_neighbor'
    unknown = 99999.99
    # 17 could be 'nan' -> in that case set it to 99999.99
    current_speed = [ 99999.99 if np.isnan(x) else x for x in list(data_ego["current_speed"])]
    execution.append(replace_unknown_values(current_speed ,unknown,  mode = mode))
    # 18 could be 'inf'-> in that case set it to 99999.99
    est_dist_cones = [ 99999.99 if np.isinf(x) else x for x in list(data_ego["estimated_dist_cones"])]
    execution.append(est_dist_cones)

    # 19
    execution.append(list(data_ego["numb_bb"]))
    # 20
    execution.append(list(data_ego["size_biggest_bb"]))
    # 21
    execution.append(list(data_ego["size_2_biggest_bb"]))
    # 22
    execution.append(list(data_ego["conf_biggest_bb"]))

    # 23
    execution.append(list(data_ego["pixel_count_color"]))
    # 24
    execution.append(list(data_ego["numb_aggl"]))
    # 25
    execution.append(list(data_ego["biggest_aggl_size"]))
    # 26
    execution.append(list(data_ego["biggest_2_aggl_size"]))

    # 27 Derivative of the size of the biggest bounding box
    size_biggest_bb = list(data_ego["size_biggest_bb"])
    execution.append([0]*10 + [ (size_biggest_bb[i+10] - size_biggest_bb[i]) \
                              for i in range(len(size_biggest_bb)-10) ])
    
    # 28 Derivative of the size of the biggest orange agglomerate
    size_biggest_aggl = list(data_ego["biggest_aggl_size"])
    execution.append([0]*10 + [ (size_biggest_aggl[i+10] - size_biggest_aggl[i]) \
                              for i in range(len(size_biggest_aggl)-10) ])


    #Cut traces to the same length
    min_length = min([len(execution[var]) for var in range(len(execution))])
    max_length = max([len(execution[var]) for var in range(len(execution))])
    if min_length != max_length: 
        # print(f'Warning: traces have different lengths. Min length: {min_length}, max length: {max_length}')
        for var in range(len(execution)): execution[var] = execution[var][:min_length]
        
    color0_rgb = list(data_supervisor["broken_car_color0"])[0]
    color1_rgb = list(data_supervisor["broken_car_color1"])[0]
    color2_rgb = list(data_supervisor["broken_car_color2"])[0]

    hue, saturation, value = colorsys.rgb_to_hsv( color0_rgb, color1_rgb, color2_rgb)
    # h_hls, l_hls, s_hls = colorsys.rgb_to_hls( color0_rgb, color1_rgb, color2_rgb)
    # y , i, q = colorsys.rgb_to_yiq( color0_rgb, color1_rgb, color2_rgb)

    hue_orange, saturation_orange, value_orange = colorsys.rgb_to_hsv( 0.941, 0.541, 0.11)
    # h_hls_orange, l_hls_orange, s_hls_orange = colorsys.rgb_to_hls( 0.941, 0.541, 0.11)
    # y_orange , i_orange, q_orange = colorsys.rgb_to_yiq(0.941, 0.541, 0.11)
    # hue_orange, saturation_orange, value_orange = colorsys.rgb_to_hsv( 0.941, 0.541, 0.11)

    # dist_from_orange_rgb = [color0_rgb - 1, color1_rgb - 0.5, color2_rgb - 0]
    # dist_from_orange_rgb = [color0_rgb - 0.941, color1_rgb - 0.541, color2_rgb - 0.11]
    dist_from_orange_hsv = [hue - hue_orange, saturation - saturation_orange, value - value_orange]
    # dist_from_orange_hls = [ h_hls - h_hls_orange, l_hls - l_hls_orange, s_hls - s_hls_orange]
    # dist_from_orange_yiq = [ y - y_orange, i - i_orange, q - q_orange]
    
    # 29 (prev 19) norma 1
    execution.append([np.linalg.norm(dist_from_orange_hsv, 1)] * min_length)
    # (prev 20) norma 2
    # execution.append([np.linalg.norm(dist_from_orange_hsv, 2)] * min_length)
    # (prev 21) norma inf
    # execution.append([np.linalg.norm(dist_from_orange_hsv, np.inf)] * min_length)
    # 30 (prev 22) hue
    execution.append([hue] * min_length)
    # 31 (prev 23) saturation
    execution.append([saturation] * min_length)
    # 32 (prev 24) value
    execution.append([value] * min_length)

    # Determine the outcome of the trace
    outcome, _ = fun_eval.evaluate_rob_boolean( [execution], system_level_formula )

    ## Cut for prediction reasons
    if min_length > trim_traces:
        for var in range(len(execution)): execution[var] = execution[var][:len(execution[var])-trim_traces]
    else:
        print(f'It was not possible to remove {trim_traces} time steps for trace {index_trace} \n')

    return execution, outcome



def CollectData(folder_name, trim_traces , indices_traces ):

    '''
    numb_pos = number of positive traces to be collected. 
    numb_neg = number of negative traces to be collected.

    In both cases : 70% of them will be used for training and 30% for testing
                    If there are less traces in the folder, all of them will be used.

    bool_trim_traces is used to say how many time units of the trace should be cut. 
      For predicitons (and learning) yes a certain amounr, but not when evaluating the system level spec -> 0 in that case'''

    
    # np.random.seed(seed)
    initial_path = f'{folder_name}/' 
   
    traces_pos, traces_neg = [] ,[]

    #Positive and Negative traces are in different folders
    for case in ['pos', 'neg' ] :
        path_supervisor = f'{initial_path}data_supervisor_{case}/'
        
        for index_trace in indices_traces:# for dataset_name in list_dataset[:max_numb]:

            if not os.path.exists(f"{path_supervisor}traces_{index_trace}.csv"): continue
            data_supervisor = pd.read_csv(f"{path_supervisor}traces_{index_trace}.csv")

            execution, outcome = structure_data_traffic_cones(data_supervisor, initial_path, index_trace, trim_traces)


            trace = execution
            
            if   outcome == 'sat': traces_pos.append(trace)
            elif outcome == 'unsat': traces_neg.append(trace) 

        
    return traces_pos, traces_neg



def CollectData_specific_trace(path, index_trace , case, trim_traces):

    '''case = 'pos' or 'neg' depending on whether the trace satisfy the system-level specification or not
     
      bool_trim_traces is used to say whether we want to remove the last {trim_traces} time units of the trace or not. 
      For predicitons (and learning) it has to be >0, but not when evaluating the system level spec (=0)'''
    

    if not os.path.exists(f"{path}data_supervisor_{case}/traces_{index_trace}.csv"): return None, None
    
    execution = []
    data_supervisor = pd.read_csv(f"{path}data_supervisor_{case}/traces_{index_trace}.csv")

    execution, outcome = structure_data_traffic_cones(data_supervisor, path,  index_trace, trim_traces)


    return execution, outcome


def compute_variables_ranges_traffic_cones(traces):

    '''Compute the range of values for each variable in the traces. : max and min values for each variable, 
    excluding 99999.99 values in 'estimated_dist_cones'.'''

    variables_ranges = []
    for var_num in range(len(traces[0])):  
        if var_num != 17 : 
            no_unknown = [trace[var_num][time_step] for trace in traces for time_step in range(len(trace[var_num]))]
        elif var_num == 17: # Remove 99999.99 values in 'estimated_dist_cones' , which is the 17th variable
            no_unknown = [trace[var_num][time_step] for trace in traces for time_step in range(len(trace[var_num])) if trace[var_num][time_step] != 99999.99]
            if len(no_unknown) == 0: #only 99999.99 values
                variables_ranges.append([0, 99999.99])
                continue # Already added the 17th variable to variables_ranges
        variables_ranges.append([min(no_unknown), max(no_unknown)])
        # Print warning if variable has the same value for all time steps in all traces
        if variables_ranges[var_num][0] == variables_ranges[var_num][1]: 
           print(f'\n\nWarning: variable {var_num} has the same value for ALL time steps in ALL traces. Value: {variables_ranges[var_num][0]}\n\n')
    return variables_ranges


def normalize_variables_traces_traffic_cones(variables_ranges, input_traces):

    '''Normalize the values of the variables in the traces except for 'estimated_dist_cones' when it is 99999.99'''
    
    variables_denom = [ (variables_ranges[var_num][1] - variables_ranges[var_num][0]) for var_num in range(len(variables_ranges)) ]
    normalized_traces = copy.deepcopy(input_traces)
    for trace in normalized_traces:
        for var_num in range(len(trace)):
            if variables_denom[var_num] != 0: # Avoid division by zero
                for time_step in range(len(trace[var_num])):
                    if (var_num != 17) or (trace[var_num][time_step]!= 99999.99): # Ok normalization for all variables except for 'estimated_dist_cones' when it is 99999.99
                        trace[var_num][time_step] = (trace[var_num][time_step] - variables_ranges[var_num][0])/ variables_denom[var_num]

    return normalized_traces

def scale_variables_formula_traffic_cones(formula_string, variable_ranges):

    '''Scale the numerical values in the formula_string according to the variable_ranges. 
    (From the value learned from the normalized traces to the actual values ranges from variable_ranges)'''

    #There is no predicate  
    if 'pi' not in formula_string: 
        print('In data_traffic_cones.py, module: scale_variables_formula_traffic_cones the formula does not have a predicate ??')
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
    if len(indices_removed) != len(list_of_name_variables): 
        sys.exit("Error: number of variables and number of values do not match in 'scale_variables_formula_traffic_cones' function in data_traffic_cones.py")
    if len(indices_removed) < 1: return formula_string

    for index_r in range(len(indices_removed)):
        index = list_of_name_variables[index_r]
        value_to_be_scaled = float(scaled_string[indices_removed[index_r][0]:indices_removed[index_r][1]+1])

        # Exclude values 99999.99 for variable 17 (estimated_dist_cones)
        if value_to_be_scaled == 99999.99: 
            if index == 17: continue
            else: # index != 17 and value_to_be_scaled == 99999.99
                print('Warning: 99999.99 value in a variable different from 17 in the formula string - data_traffic_cones.py')
            
        new_value = (value_to_be_scaled * (variable_ranges[index][1] - variable_ranges[index][0] )) + variable_ranges[index][0]
        new_scaled_string = scaled_string[:indices_removed[index_r][0]] + str(new_value) + scaled_string[indices_removed[index_r][1] + 1 :]
        diff_char = len(scaled_string) - len(new_scaled_string)
        # Shift according to the difference in length of the string
        for index_update_r in range(index_r + 1, len(indices_removed)):
            indices_removed[index_update_r][0] -= diff_char
            indices_removed[index_update_r][1] -= diff_char
        scaled_string = new_scaled_string

    return scaled_string


def renormalize_traces_traffic_cones(traces,  variables_to_be_changed, old_variable_ranges, new_variable_ranges):

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
                if ((var_num != 17) or (trace[var_num][time_step]!= 99999.99)) \
                  and (new_variables_denom[var_num] != 0) :
                    # First: scale the value from range [0,1] to the actual value in range old_variable_ranges
                    aux = (trace[var_num][time_step] * old_variables_denom[var_num]) + old_variable_ranges[var_num][0]
                    # Second: normalize in [0,1] according to the new_variable_ranges
                    trace[var_num][time_step] = (aux - new_variable_ranges[var_num][0])/ new_variables_denom[var_num]

    return rescaled_traces


def convert_table(path, case, feature_names):

    '''Convert error table or safe table to the feature table named , respectively ,
     pos_table.csv or neg_table.csv'''

    if case == 'neg': table = pd.read_csv(f'{path}error_table.csv')
    elif case == 'pos': table = pd.read_csv(f'{path}safe_table.csv')
    
    # For every row in table
    new_table = []
    for row in table.iterrows():
        indices  = [ 1, 2, 3, 4,5,6,7,8,9,10,11,12,13,14 ] # indices of the features in the table
        new_row = [ row[1][index] for index in indices]
        new_table.append(new_row)
    
    df = pd.DataFrame(new_table, columns = feature_names)
    df.to_csv(f'{path}{case}_table.csv', index = False)
    return 
