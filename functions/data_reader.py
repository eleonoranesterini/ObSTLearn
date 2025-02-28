import numpy as np
import os
import random

def read_normalize_data(name_folder_data, horizon, indices_traces, preserve_proportion = False, proportion = None):

    if 'lead_follower' in name_folder_data: case_study = 'lead_follower'
    if 'traffic_cones' in name_folder_data: case_study = 'traffic_cones'

    if case_study == 'lead_follower':
        from experiments.lead_follower.data_lead_follower import CollectData
        from experiments.lead_follower.data_lead_follower import compute_variables_ranges_lead_follower as compute_variables_ranges
        from experiments.lead_follower.data_lead_follower import normalize_variables_traces_lead_follower as normalize_variables_traces
        if proportion is None: proportion = 0.14
            
        
    elif case_study == 'traffic_cones':
        from experiments.traffic_cones.data_traffic_cones import CollectData
        from experiments.traffic_cones.data_traffic_cones import compute_variables_ranges_traffic_cones as compute_variables_ranges
        from experiments.traffic_cones.data_traffic_cones import normalize_variables_traces_traffic_cones as normalize_variables_traces
        if proportion is None: proportion = 0.05

    #Collect data
    positive_traces, negative_traces =  CollectData(f"{name_folder_data}", horizon , indices_traces )
    additional_traces_indices = None
    if preserve_proportion:
        positive_traces, negative_traces, additional_traces_indices = fun_preserve_proportion(positive_traces, negative_traces, proportion, name_folder_data, horizon, indices_traces, case_study)


    print(f'Number of positive traces: {len(positive_traces)}')
    print(f'Number of negative traces: {len(negative_traces)}')

    #Compute variables ranges for learning traces
    variables_ranges = compute_variables_ranges(positive_traces+negative_traces)
    #Normalize parameters to be mined according to variables ranges
    normalized_positive_traces = normalize_variables_traces(variables_ranges, positive_traces)
    normalized_negative_traces = normalize_variables_traces(variables_ranges, negative_traces)

    return normalized_positive_traces, normalized_negative_traces, variables_ranges, additional_traces_indices


def shuffle_and_randomly_divide_data(name_folder_data, numb_simulation, horizon, seed):
    
    '''The function is used to read data from different set of training traces, shuffle them and return them in batches of size numb_simulation'''
    
    random.seed(seed)
   
    if 'lead_follower' in name_folder_data: case_study = 'lead_follower'
    if 'traffic_cones' in name_folder_data: case_study = 'traffic_cones'

    if case_study == 'lead_follower':
        from experiments.lead_follower.data_lead_follower import CollectData_specific_trace
        from experiments.lead_follower.data_lead_follower import compute_variables_ranges_lead_follower as compute_variables_ranges
        from experiments.lead_follower.data_lead_follower import normalize_variables_traces_lead_follower as normalize_variables_traces
            
        
    elif case_study == 'traffic_cones':
        from experiments.traffic_cones.data_traffic_cones import CollectData_specific_trace
        from experiments.traffic_cones.data_traffic_cones import compute_variables_ranges_traffic_cones as compute_variables_ranges
        from experiments.traffic_cones.data_traffic_cones import normalize_variables_traces_traffic_cones as normalize_variables_traces
    
    traces_pos, traces_neg = [] ,[]
    combined_list = []
    for dataset in ['training_main', 'training_additional']:
        for index_trace in list(range(1,4500)):
            execution, outcome = CollectData_specific_trace(f'{name_folder_data}/{dataset}/', index_trace, 'pos', horizon)
            if execution is None:  execution, outcome = CollectData_specific_trace(f'{name_folder_data}/{dataset}/', index_trace, 'neg', horizon)
            if execution is  None: continue
            if   outcome == 'sat': 
                traces_pos.append((execution, f'{name_folder_data}/{dataset}/{index_trace}'))
                combined_list.append(((execution, f'{name_folder_data}/{dataset}/{index_trace}'), 'pos'))
            elif outcome == 'unsat': 
                traces_neg.append((execution, f'{name_folder_data}/{dataset}/{index_trace}'))
                combined_list.append(((execution, f'{name_folder_data}/{dataset}/{index_trace}'), 'neg'))

    # Combine both lists into one, keeping track of the origin
    #combined_list = [(item, 'pos') for item in traces_pos] + [(item, 'neg') for item in traces_neg]
    ## Shuffle the combined list to ensure randomness
    random.shuffle(combined_list)

    not_norm_batch_traces_pos, not_norm_batch_traces_neg, batch_indices = [], [], []
    for _ in range(10): #10 batches
        aux_traces_pos , aux_traces_neg , aux_indices = [] , [] , []
        for _ in range(numb_simulation):
            item = combined_list.pop(0)
            if item[1] == 'pos': aux_traces_pos.append(item[0][0]) # add the execution 
            elif item[1] == 'neg': aux_traces_neg.append(item[0][0]) # add the execution
            aux_indices.append(item[0][1]) # Store the index of the trace
        not_norm_batch_traces_pos.append(aux_traces_pos)
        not_norm_batch_traces_neg.append(aux_traces_neg)
        batch_indices.append(aux_indices)

    # Normalize each batch of data
    batch_traces_pos , batch_traces_neg, batch_var_ranges = [], [], []
    for it in range(len(not_norm_batch_traces_pos)):
        #Compute variables ranges for learning traces
        variables_ranges = compute_variables_ranges(not_norm_batch_traces_pos[it]+not_norm_batch_traces_neg[it])
        #Normalize parameters to be mined according to variables ranges
        normalized_positive_traces = normalize_variables_traces(variables_ranges, not_norm_batch_traces_pos[it])
        normalized_negative_traces = normalize_variables_traces(variables_ranges, not_norm_batch_traces_neg[it])

        batch_traces_pos.append(normalized_positive_traces)
        batch_traces_neg.append(normalized_negative_traces)
        batch_var_ranges.append(variables_ranges)


    return batch_traces_pos, batch_traces_neg, batch_var_ranges, batch_indices



def fun_preserve_proportion(positive_traces, negative_traces, proportion_neg_traces, name_folder_data, horizon, indices_traces, case_study):

    '''Preserve the proportion of positive and negative traces in the dataset while maintaining the total number of traces'''
    if case_study == 'lead_follower': from experiments.lead_follower.data_lead_follower import CollectData
    elif case_study == 'traffic_cones': from experiments.traffic_cones.data_traffic_cones import CollectData

    numb_traces_dataset = len([name for name in os.listdir(f'{name_folder_data}/data_supervisor_neg') ]) + len([name for name in os.listdir(f'{name_folder_data}/data_supervisor_pos')]) 
    bool_removed, bool_added = False, False

    additional_traces_pos = []
    additional_traces_neg = []

    index = indices_traces[-1]
    
    while True: 
        # If the proportion of negative traces is reasonable (i.e. within 1% of the desired proportion)
        #  or if some traces have been removed and some have been added (meaning we have a small dataset and we cannot achieve 1% error in the proportion)
        if (proportion_neg_traces - 0.009  < len(negative_traces)/(len(positive_traces+ negative_traces)) < proportion_neg_traces + 0.009) or (bool_removed and bool_added):
            break 
        
        # Restart from the beginning if we have reached the end of the dataset
        if index <  numb_traces_dataset: index += 1  
        else: index = 1
            
        if index in indices_traces: # Restarted and still not found -> ok stop
            break

        # If the proportion of negative traces is too high, replace one negative trace with a positive one
        if len(negative_traces)/(len(positive_traces+ negative_traces)) > proportion_neg_traces:
            new_execution, _ = CollectData(f"{name_folder_data}", horizon , [index] )
            if len(new_execution) == 0:  continue # The index corresponds to a negative trace
            bool_removed = True
            if len(negative_traces) == 1: break # If we have only one negative trace, let's not remove it and stop
            negative_traces.pop()
            positive_traces.append(new_execution[0])
            additional_traces_pos.append(index)
            

        # If the proportion of negative traces is too low, replace one positive trace with a negative one
        elif len(negative_traces)/(len(positive_traces+ negative_traces)) < proportion_neg_traces:
            
            _ ,  new_execution = CollectData(f"{name_folder_data}", horizon , [index] )
            if len(new_execution) == 0:  continue # The index corresponds to a positive trace
            bool_added = True
            if len(positive_traces) == 1: break # If we have only one positive trace, let's not remove it and stop
            positive_traces.pop()
            additional_traces_neg.append(index)
            negative_traces.append(new_execution[0])
           

    return positive_traces, negative_traces, [additional_traces_pos, additional_traces_neg]



def replace_unknown_values(variable_values, unknown, mode = 'nearest_neighbor'):

    '''Replace the unknown values (unknown) in the traces with the mean or median or mode of the known values for each variable'''
    
    if mode == 'nearest_neighbor': # Best for time series data where features are sequentially related

        new_values = []

        for i in range(len(variable_values)):
            # If the value is unknown
            if variable_values[i] == unknown:
                j = 1
                # Find the closest known value to the unknown value:
                # First backward then (if not found) forward
                # Consider interpolation ?
                while True:
                    # Check if there is a known value at distance i-j from the unknown value
                    if i-j >= 0 and variable_values[i-j] != unknown: 
                        new_values.append(variable_values[i-j])
                        break
                    # Check if there is a known value at distance i-jfrom the unknown value
                    elif i+j < len(variable_values) and variable_values[i+j] != unknown:
                        new_values.append(variable_values[i+j])
                        break
                    # Already scanned all the values -> all are unknown so add 0
                    elif j == len(variable_values):
                        new_values.append(0)
                        break
                    # Increment the distance
                    else: 
                        j += 1
                    
            # If the value is known
            else: new_values.append(variable_values[i])
        return new_values

    elif mode == 'mean':      new_value = np.mean([el for el in variable_values if el != unknown])
    elif mode == 'median':  new_value = np.median([el for el in variable_values if el != unknown])
    elif mode == 'mode':    new_value = max(set([el for el in variable_values if el != unknown]), key = variable_values.count)

    return [new_value if el == unknown else el for el in variable_values]

