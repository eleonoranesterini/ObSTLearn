import numpy as np
import sys


import functions.syntax_eval as fun_eval
import functions.metrics as metrics

       
def scale_node(case_study, formula_string, variable_ranges):

    if case_study == 'lead_follower':
            from experiments.lead_follower.data_lead_follower import scale_variables_formula_lead_follower 
            return scale_variables_formula_lead_follower(formula_string, variable_ranges)
    elif case_study == 'traffic_cones':
            from experiments.traffic_cones.data_traffic_cones import scale_variables_formula_traffic_cones 
            return scale_variables_formula_traffic_cones(formula_string, variable_ranges)




def evaluation_single_formulas(formula_string, positive_traces, negative_traces, output_file_name, case_study, bool_features = False):
    
    '''
    traces and formula_string are already in the same parameter range (either both normalized or both scaled)
    '''

    if case_study == 'lead_follower':
        if not bool_features:
            from experiments.lead_follower.data_lead_follower import scale_variables_formula_lead_follower as scale_node
            from experiments.lead_follower.data_lead_follower import normalize_variables_traces_lead_follower as normalize_variables_traces
    elif case_study == 'traffic_cones':
        if not bool_features:
            from experiments.traffic_cones.data_traffic_cones import scale_variables_formula_traffic_cones as scale_node
            from experiments.traffic_cones.data_traffic_cones import normalize_variables_traces_traffic_cones as normalize_variables_traces


   
    nb_false_positives  = 0 # Incorrectly classified as positive
    # sat sys lev spec, vio of the monitor 

    nb_true_positives = 0 # Correctly classified as positive
    # unsat sys lev spec, vio of the monitor

    nb_false_negatives= 0  # Incorrectly classified as negative
    # vio sys lev spec, sat of the monitor

    nb_true_negatives = 0 # Correctly classified as negative
    # sat sys lev spec, sat of the monitor
    
    nb_false_positives_runout , nb_false_negatives_runout = 0,0 
    vector_cost_ob , vector_cost_msc= [] ,[]
    vector_rfp, vector_rfn = [],[]
    # vector_mutation , vector_hyper_monitor, aux_mutation , aux_hyper_monitor = [] , [] , [] ,[] 


    # print('Before scaling: ', item.string)
    # formula_string = scale_node(item.string , item.variable_ranges)
    # print('After scaling: ', formula_string)
    # with open(f'{output_file_name}','a') as file:
    #     file.write(f'Pre-scaled:{item.string}')
    #     file.write(f'\nAfter-scaled:{formula_string}\n')


    # Evaluate negative traces trace by trace
    for pi in negative_traces: # traces are supposed to be normalized
        negative_result, _ = fun_eval.evaluate_rob_boolean([pi], formula_string)
        if negative_result == 'runout': nb_false_negatives_runout += 1
        elif negative_result == 'sat' : nb_false_negatives += 1
        elif negative_result == 'unsat' : nb_true_positives += 1
    
    #Evaluate positive traces trace by trace
    for pi in positive_traces:# traces are supposed to be normalized
        positive_result, _ = fun_eval.evaluate_rob_boolean([pi], formula_string )
        if positive_result == 'runout' : nb_false_positives_runout += 1
        elif positive_result == 'unsat' : nb_false_positives += 1
        elif positive_result == 'sat' : nb_true_negatives += 1

    
    with open(f'{output_file_name}','a') as file: 
        file.write(f'\n\nFormula:\n {formula_string}\n')

        # cost_ob = metrics.compute_obj_fun(positive_traces, negative_traces, formula_string, verbose = 1, mode = 'cost')
        # vector_cost_ob.append(cost_ob)
        # file.write(f'Cost obj function (on the normalized trace) = {cost_ob}')
        _ , cost_msc, rfp, rfn = metrics.compute_obj_fun(positive_traces, negative_traces, formula_string, verbose = 1, mode = 'msc')
        vector_cost_msc.append(cost_msc)
        file.write(f'\nMSC = {cost_msc}, rfp ={rfp}, rfn={rfn}\n') #MSC capital letters is the corrected version
        file.write(f'\nNumber of false positives: {nb_false_positives}')
        file.write(f'\nNumber of false negatives: {nb_false_negatives}')
        file.write(f'\nNumber of true positives : {nb_true_positives}')
        file.write(f'\nNumber of true negatives : {nb_true_negatives}')
        file.write(f'\nAccuracy: {(nb_true_positives+nb_true_negatives)/(len(positive_traces)+len(negative_traces))}')
        if nb_false_positives+nb_true_positives > 0:
            precision = nb_true_positives/(nb_false_positives+nb_true_positives)
            file.write(f'\nPrecision: {precision}')
        else: 
            precision = 0
            file.write(f'\nPrecision: denominator is 0')
        if nb_true_positives+nb_false_negatives > 0:
            recall = nb_true_positives/(nb_true_positives+nb_false_negatives)
            file.write(f'\nRecall: {recall}')
        else:
            recall = 0
            file.write(f'\nRecall: denominator is 0')
        if   (precision+recall)>0:
            file.write(f'\nF1-score: {2*(precision*recall)/(precision+recall)}')
        else:
            file.write(f'\nF1-score: precision or recall not defined')

        if len(positive_traces) >0: 
            vector_rfp.append(nb_false_positives/len(positive_traces))
            file.write(f'\nRatio false positives: {nb_false_positives/len(positive_traces)}')
        # else: vector_rfp.append(-1)
        if len(negative_traces)>0: 
            vector_rfn.append(nb_false_negatives/len(negative_traces))
            file.write(f'\nRatio false negatives: {nb_false_negatives/len(negative_traces)}')
        # else: vector_rfn.append(-1)
        
    # aux_mutation.append(item.numb_mutation)
    # aux_hyper_monitor.append(item.numb_hyper_monitor)

    # vector_mutation.append(np.mean(aux_mutation))    
    # vector_hyper_monitor.append(np.mean(aux_hyper_monitor)) 

    # with open(f'{output_file_name}','a') as file:
    #     file.write(f'\n vector_cost_ob = {vector_cost_ob}')
    #     file.write(f'\n vector_cost_msc = {vector_cost_msc}')

    #     file.write(f'\n vector_rfp = {vector_rfp}')
    #     file.write(f'\n vector_rfn = {vector_rfn}')

        
    #  AVERAGE  RATIO OF FALSE POSITIVES  (ONLY) 
    with open(f'{output_file_name}','a') as file:
        file.write(f'\n\nAverage ratio of false positives: {np.mean(vector_rfp)}')
        file.write(f'\nAverage ratio of false negatives: {np.mean(vector_rfn)}')
       

    ########################################
    #    NUMBER OF MUTATION AND MONITORS   #
    ########################################
    # with open(f'{output_file_name}','a') as file:
    #     file.write(f'\n\n\n Number of mutations: {vector_mutation}')
    #     file.write(f'\nAverage: {np.mean(vector_mutation)}+-{np.std(vector_mutation)}')
        
    #     file.write(f'\n\n\n Number of hyper_monitors: {vector_hyper_monitor}')
    #     file.write(f'\nAverage: {np.mean(vector_hyper_monitor)}+-{np.std(vector_hyper_monitor)}')
    
    

    return [nb_true_positives, nb_true_negatives, nb_false_positives, nb_false_negatives]


def evaluate_specific_ensemble(ensemble_type, positive_traces , negative_traces, learned_formulas, case_study, bool_features):
    
    '''
    

    INPUTS:

    - ensemble_type: string 
            'majority_voting'
            'conservative_voting' (unanimity)
            'total_robustness' (Sum the robustness of all the monitors and if it is greater than 0, the ensamble vote for satissfaction)
            'largest_robustness' (Highest absolute value of robustness: if it is is greater than 0, the ensemble votes for satisfaction)

    Traces are objects with two attributes: execution and features
     - positive_traces: list of traces with positive labels - not normalized
    - negative_traces: list of traces with negative labels - not normalized
   

    - learned_formulas: list of formulas learned by the mining process (they are normalized)
    Now each formula has its own attribute variables_ranges) 
    - case_study: string with the name of the case study ('lead_follower', 'traffic_cones', or 'lane_keeping')
    - bool_features: boolean to indicate whether the classification of traces is wrt features ( or not, in this case: execution )
    
    OUTPUTS: Traces object (attributes: execution and features) divided into:
    # Positive refers on whether an alert is issued or not

    - true_positives   # classified as unsat and are indeed unsat
    - true_negatives   # classified as sat   and are indeed sat 

    - false_positives  # classified as unsat but should be sat
    - false_negatives  # classified as sat but should be unsat
            
    '''

    if case_study == 'lead_follower':
        if not bool_features:
            from experiments.lead_follower.data_lead_follower import scale_variables_formula_lead_follower as scale_node
        else:
            from experiments.lead_follower.data_lead_follower import rescale_features_lead_follower as scale_node
    
    elif case_study == 'traffic_cones':
        if not bool_features:
             from experiments.traffic_cones.data_traffic_cones import scale_variables_formula_traffic_cones as scale_node
    
    elif case_study == 'lane_keeping':
        from experiments.lane_keeping.data_lane_keeping import scale_node_lane_keeping as scale_node

    true_positives = []     # classified as unsat and are indeed unsat
    true_negatives = []     # classified as sat   and are indeed sat
    false_positives = []    # classified as unsat but should be sat
    false_negatives =  []   # classified as sat but should be unsat

    if ensemble_type == 'majority_voting':
        for pi in negative_traces:
            count = 0 #count number of individual (by each monitor) mistakes 
            for index, item in enumerate(learned_formulas):
                formula_string = scale_node(item.string , item.variable_ranges) 
                if not bool_features: negative_result, _ = fun_eval.evaluate_rob_boolean([pi], formula_string)
                else: negative_result, _ = fun_eval.evaluate_rob_boolean([pi.features], formula_string)

                if negative_result == 'sat' : count += 1

                if count >= len(learned_formulas)/2:
                    false_negatives.append(pi)
                    break
                elif index == len(learned_formulas) - 1: # Here it is automatic that : count < len(learned_formulas)/2 
                    true_positives.append(pi)

        for pi in positive_traces:
            count = 0 #count number of individual errors 
            for index, item in enumerate(learned_formulas):
                formula_string = scale_node(item.string , item.variable_ranges) 
                if not bool_features: positive_result, _ = fun_eval.evaluate_rob_boolean([pi], formula_string )
                else: positive_result, _ = fun_eval.evaluate_rob_boolean([pi.features], formula_string )
            
                if positive_result == 'unsat' : count += 1

                if count >= len(learned_formulas)/2:    
                    false_positives.append(pi)
                    break
                elif index == len(learned_formulas) - 1: # Here it is automatic that : count < len(learned_formulas)/2
                    true_negatives.append(pi)

    elif ensemble_type == 'conservative_voting':
        for pi in negative_traces:
            for index, item in enumerate(learned_formulas):
                formula_string = scale_node(item.string , item.variable_ranges) 
                if not bool_features: negative_result, _ = fun_eval.evaluate_rob_boolean([pi], formula_string )
                else: negative_result, _ = fun_eval.evaluate_rob_boolean([pi.features], formula_string )

                #If one of the monitors identifies the violation -> ok, go to next trace
                if negative_result == 'unsat' : 
                    true_positives.append(pi)
                    break
                # If the loop has reached the last iteration and none has identifed violation -> FN
                elif index == len(learned_formulas) - 1: # It is automatic that: negative_result == 'sat' 
                    false_negatives.append(pi)

        for pi in positive_traces:
            for index, item in enumerate(learned_formulas):
                formula_string = scale_node(item.string , item.variable_ranges) 
        
                if not bool_features: positive_result, _ = fun_eval.evaluate_rob_boolean([pi], formula_string)
                else: positive_result, _ = fun_eval.evaluate_rob_boolean([pi.features], formula_string)

                #If one of the monitors identifies the violation -> FP , go to next trace
                if positive_result == 'unsat' : 
                    false_positives.append(pi)
                    break
                elif index == len(learned_formulas) - 1: # It is automatic that: positive_result == 'sat' 
                    true_negatives.append(pi)

    elif ensemble_type == 'total_robustness':
        for pi in negative_traces:
            total_robustness = 0 # For a fixed trace, the sum of the robustness of all the monitors
            for index, item in enumerate(learned_formulas):
                formula_string = scale_node(item.string , item.variable_ranges) 
                if not bool_features: robustness = fun_eval.evaluate_rob_quantitative([pi], formula_string )
                else: robustness = fun_eval.evaluate_rob_quantitative([pi.features], formula_string )

                total_robustness += robustness

            if total_robustness >= 0: false_negatives.append(pi)
            else: true_positives.append(pi)

        for pi in positive_traces:
            total_robustness = 0 # For a fixed trace, the sum of the robustness of all the monitors
            for index, item in enumerate(learned_formulas):
                formula_string = scale_node(item.string , item.variable_ranges) 
                if not bool_features: robustness = fun_eval.evaluate_rob_quantitative([pi], formula_string )
                else: robustness = fun_eval.evaluate_rob_quantitative([pi.features], formula_string )

                total_robustness += robustness

            if total_robustness < 0: false_positives.append(pi)
            else: true_negatives.append(pi)

    elif ensemble_type == 'largest_robustness':
        for pi in negative_traces:
            largest_robustness = 0 # For a fixed trace, the largest robustness among all the monitors
            for index, item in enumerate(learned_formulas):
                formula_string = scale_node(item.string , item.variable_ranges) 
                if not bool_features: robustness = fun_eval.evaluate_rob_quantitative([pi], formula_string )
                else: robustness = fun_eval.evaluate_rob_quantitative([pi.features], formula_string )
                # Update the largest robustness (in terms of absolute value)
                if abs(robustness) > abs(largest_robustness): largest_robustness = robustness

            if largest_robustness >= 0: false_negatives.append(pi)
            else: true_positives.append(pi)
        
        for pi in positive_traces:
            largest_robustness = 0
            for index, item in enumerate(learned_formulas):
                formula_string = scale_node(item.string , item.variable_ranges) 
                if not bool_features: robustness = fun_eval.evaluate_rob_quantitative([pi], formula_string )
                else: robustness = fun_eval.evaluate_rob_quantitative([pi.features], formula_string )
                # Update the largest robustness (in terms of absolute value)
                if abs(robustness) > abs(largest_robustness): largest_robustness = robustness

            if largest_robustness < 0: false_positives.append(pi)
            else: true_negatives.append(pi)

    else:
        print('Warning: Invalid ensemble type!')
        sys.exit()
    
    return true_positives, true_negatives, false_positives, false_negatives

def write_ensemble_result(output_file_name , number_true, number_classification ):

    nb_positive_traces , nb_negative_traces = number_true[0], number_true[1]
    nb_true_positives, nb_true_negatives, nb_false_positives , nb_false_negatives = number_classification[0], number_classification[1],number_classification[2],number_classification[3]

    with open(f'{output_file_name}','a') as file:
        file.write(f'\nNumber of positive traces in the dataset: {nb_positive_traces}')
        file.write(f'\nNumber of negative traces in the dataset: {nb_negative_traces}')
        
        file.write(f'\n\nNumber of true positives: {nb_true_positives}')
        file.write(f'\nNumber of true negatives: {nb_true_negatives}')
        file.write(f'\nNumber of false positives: {nb_false_positives}')
        file.write(f'\nNumber of false negatives: {nb_false_negatives}')
        MSC = (nb_false_negatives+nb_false_positives)/(nb_negative_traces+nb_positive_traces)
        file.write(f'\nMSC: {MSC}')
        # file.write(f'\nAccuracy: {(nb_true_positives+nb_true_negatives)/(nb_negative_traces+nb_positive_traces)}')
        if nb_false_positives+nb_true_positives > 0:
            precision = nb_true_positives/(nb_false_positives+nb_true_positives)
            file.write(f'\nPrecision: {precision}')
        else:
            precision = 0
            file.write(f'\nPrecision: denominator is 0')
        if nb_true_positives+nb_false_negatives > 0:
            recall = nb_true_positives/(nb_true_positives+nb_false_negatives)
            file.write(f'\nRecall: {recall}')
        else:
            recall = 0
            file.write(f'\nRecall: denominator is 0')
        if (precision+recall)>0:
            f1_score = 2*(precision*recall)/(precision+recall)
            file.write(f'\nF1-score: {f1_score}')
        else:
            f1_score = 0
            file.write(f'\nF1-score: precision or recall not defined')
            
        if nb_positive_traces>0: file.write(f'\n\nRatio of false positives : = {nb_false_positives/nb_positive_traces}')
        if nb_negative_traces>0: file.write(f'\nRatio of false negatives : {nb_false_negatives/nb_negative_traces}')

    return [ MSC, precision, recall, f1_score ]

def evaluate_all_ensembles(positive_traces, negative_traces, output_file_name, learned_formulas, case_study, bool_features = False):

        number_true = [len(positive_traces), len(negative_traces)]

        ########################################
        #     ENSEMBLE EVALUATION              #
        ########################################

        print('\n\n# Evaluation majority voting #')
        with open(f'{output_file_name}','a') as file:
            file.write('\n\n\n\n#############################\n')
            file.write('# Evaluation majority voting #\n')
            file.write('#############################\n')
        
        true_positives, true_negatives , false_positives, false_negatives= evaluate_specific_ensemble( 'majority_voting', positive_traces , negative_traces, learned_formulas, case_study, bool_features )
        number_classification_MV = [len(true_positives), len(true_negatives), len(false_positives), len(false_negatives)]
        
        MV = write_ensemble_result(output_file_name , number_true, number_classification_MV )

    
        #Sum the robustness of all the monitors and if it is greater than 0, the ensamble vote for satissfaction
        print('\n\n## Evaluation with average robustness ##')
        with open(f'{output_file_name}','a') as file:
            file.write('\n\n\n\n#############################\n')
            file.write('Evaluation with average robustness \n')
            file.write('#############################\n')

        true_positives, true_negatives , false_positives , false_negatives = evaluate_specific_ensemble( 'total_robustness', positive_traces, negative_traces, learned_formulas, case_study , bool_features)
        number_classification_AR = [len(true_positives), len(true_negatives), len(false_positives), len(false_negatives)]
        AR = write_ensemble_result(output_file_name , number_true, number_classification_AR )

        #Highest absolute value of robustness: if it is is greater than 0, the ensemble votes for satisfaction
        print('\n\n## Evaluation with largest robustness ##')
        with open(f'{output_file_name}','a') as file:
            file.write('\n\n\n\n#############################\n')
            file.write('Evaluation with largest robustness \n')
            file.write('#############################\n')
        
        true_positives, true_negatives , false_positives , false_negatives = evaluate_specific_ensemble( 'largest_robustness',positive_traces , negative_traces, learned_formulas,   case_study, bool_features)
        number_classification_LR = [len(true_positives), len(true_negatives), len(false_positives), len(false_negatives)]
        LR = write_ensemble_result(output_file_name , number_true, number_classification_LR )
    
        return MV, AR, LR