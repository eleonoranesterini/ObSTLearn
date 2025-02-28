#syntax_guided, create_grammar, inputs, sample_formulas , syntax_eval, syn_rules, SMT
# function_enumerate, function_mono, function_updates_print, function_create_formulas, metrics

import random 
import copy
import sys
import time
import shutil
import cProfile

import functions.enumerate_formulas as fun_enum
import functions.syntax_eval as fun_eval
import functions.apply_updates as fun_upd
import functions.create_formulas as fun_create
import functions.metrics as fun_metr
import functions.monotonicity_functions as fun_mono
# import functions.refinement as refinement
import functions.create_grammar as gramm
import functions.data_reader as data_reader
import functions.final_evaluation as final_eval

 
"""
Algorithm structure:

        Create initial solution 
        Compute cost initial solution
        Initialize best solution found so far

        Initialize temperature

        While stopping condition (good result or too many iterations) is not met:
            Create new solution
            Instantiate parameter for new solution
            Compute cost new solution

            Update best solution found so far (if better & new)
            Probabilistically update current solution
        
        Consider the best solution found so far as the final solution
"""
       
def print_formula(quantifiers, formula_string):
    
    numb_quant = len(quantifiers)
    f = ''
    for i in range(numb_quant):
        if quantifiers[i]==0: f += f'E pi[{i}] '
        elif quantifiers[i]==1: f += f'FA pi[{i}] '
    
    f = f + ': ' + formula_string
    return f


def main(name_folder_data, inputs, output_file_name = 'results.text', numb_simulation = 200 , start_trace_index = 0, bool_preserve_proportion = False, proportion = None, bool_random_dataset_training = False):
    
    #Whether to apply Syntactic rules 
    bool_apply_syntactic_rules = True 
    
    ##Initialization
    random.seed(inputs.seed)
    
    count = fun_upd.Counter()
    count.it_changes_max = 50

    # Maximum number of External Iterations proportional to the target number of formulas
    count.nb_iterations_outer = inputs.n_target * count.nb_iterations_outer 
    
    database = []#to store formulas
    # Consider to remove these: (to mine independent formulas)
    set_database_parametric = set() # To check if a formula with parameter symbols is new (after already having mined previous formula)
    set_database = set() # To check if a concrete formula is new (after already having mined previous formula)
    
    grammar = gramm.Grammar(gramm.parse_gramm(inputs.grammar_quantifiers), gramm.parse_gramm(inputs.grammar_structure) )
    # grammar_predicate will be updated during the algorithm (depending on the number of quantifiers sampled)
    # initial_grammar_predicate is how the user has provided the predicates at the beginning (fixed)
    initial_grammar_predicate = inputs.grammar_predicate
    
    # Dictionary mapping #number of quantifiers to a list with the amount of concrete predicates ordered by predicate_level
    numb_pred_dict = dict() # key : '{numb_quant}'
    # Dictionary mapping #number of quantifiers and formula length to a list with the amount of concrete predicates in each formula with these features 
    numb_form_dict = dict() # key: '{spec_level}_{numb_quant}_{length}'
    # 
    dict_par_inst = dict() # key: '{formula_string}' with {parameters}, {cost}, {iteration} 

    # Initialize some monitoring features
    monitor = fun_eval.Inputs_Monitoring()  
    
    #Set up paths for enumeration and storing formulas
    path_to_save = fun_enum.set_up_paths(inputs)
   
    tim = fun_upd.StudyComputationalTime()
    time_start = time.time()
    time_beginning = time.time()

    # Initialize parameters for the traces to be used in the learning process
    end_trace_index = numb_simulation
    if 'lead_follower'   in name_folder_data:   case_study = 'lead_follower'
    elif 'traffic_cones' in name_folder_data:   case_study = 'traffic_cones'
        

    #Loop to find a set of formulas (LOOP 1)
    while True:
        #Already collected n_target formulas --> END of the algorithm
        if len(database) >= inputs.n_target: break

        count.iter_outer_loop += 1
        #Stop after "too many" iterations --> END of the algorithm
        if count.iter_outer_loop > count.nb_iterations_outer: break

        ## READ DATA
        start_reading = time.time()
        if not bool_random_dataset_training:
            # Read database of formulas deterministically 
            traces_pos, traces_neg, var_ranges , additional_traces_indices = \
                data_reader.read_normalize_data(name_folder_data, inputs.horizon, list(range(start_trace_index, end_trace_index)), preserve_proportion = bool_preserve_proportion, proportion = proportion)
            
            with open(f'{inputs.name_output_folder}/{output_file_name}','a') as file: 
                file.write(f'\nTraces in {name_folder_data} from {start_trace_index} to {end_trace_index}\n\n')
                if additional_traces_indices is not None: file.write(f'Additional traces pos: {additional_traces_indices[0]}\n')
                if additional_traces_indices is not None: file.write(f'Additional traces neg: {additional_traces_indices[1]}\n')
                file.write(f'Positive traces:{len(traces_pos)}\n')
                file.write(f'Negative traces:{len(traces_neg)}\n')
            start_trace_index = end_trace_index
            end_trace_index += numb_simulation
        else:
            # The first time: apply the shuffling of the traces and divide them into 10 sets for learning of size numb_simulation each
            if count.iter_outer_loop == 1:
                batch_traces_pos, batch_traces_neg, batch_var_ranges, batch_indices = data_reader.shuffle_and_randomly_divide_data(name_folder_data, numb_simulation, inputs.horizon, inputs.seed)
            
            traces_pos = batch_traces_pos.pop(0)
            traces_neg = batch_traces_neg.pop(0)
            var_ranges = batch_var_ranges.pop(0)
            traces_indices = batch_indices.pop(0)
            with open(f'{inputs.name_output_folder}/{output_file_name}','a') as file: 
                file.write(f'\nTraces: {traces_indices}\n\n')
                file.write(f'Positive traces:{len(traces_pos)}\n')
                file.write(f'Negative traces:{len(traces_neg)}\n')
         
        tim.read_traces += (time.time() - start_reading)
        print('Time read traces', tim.read_traces)
        numb_var = len(traces_pos[0]) # number of variables 
        inputs.variable_ranges = copy.deepcopy(var_ranges)   
        # Create initial solution 
        formula, grammar, count, tim, numb_form_dict, numb_pred_dict = \
                fun_create.create_starting_formula(inputs, grammar, initial_grammar_predicate, numb_pred_dict, numb_form_dict, count, tim, numb_var )
        
        #Reset counter after new starting formula (count.looser_same_formula, count.strengthened_same_formula , count.mutation, count.hyper_monitor)
        count = fun_upd.reset_counter_after_new_starting_formula(count)
    
        # Instantiate the parameters in the formula 
        formula, tim, dict_par_inst = fun_metr.instantiate_parameters(inputs, formula, count, traces_pos, traces_neg, tim, dict_par_inst)
        print('Initial formula instantiated:', formula.string)

        # Compute cost initial solution
        time_start_monitoring = time.time()
        formula.cost,  msc, ratio_fp, ratio_fn  = fun_eval.cost(traces_pos, traces_neg, formula, inputs, monitor , count, mode = 'msc')
        tim.monitor += ( time.time() - time_start_monitoring )  
        # print('Initial formula cost:', formula.cost)
        store_cost = [formula.cost]
        store_msc = [msc]
        store_ratio_fp = [ratio_fp]
        store_ratio_fn = [ratio_fn]

        # Initialize best solution found so far
        formula_best = copy.deepcopy(formula)
        
        # Initialize temperature 
        temperature = fun_metr.fun_temp(inputs, 'init')
            
        while True: 

            if fun_eval.check_success_stopping_condition(inputs, formula_best.cost,  msc = msc, ratio_fp = ratio_fp, ratio_fn = ratio_fn ) \
                or (count.mutation + count.looser_same_formula + count.strengthened_same_formula) >= count.it_changes_max: # 50 is the maximum number of changes

                if  (count.mutation + count.looser_same_formula + count.strengthened_same_formula) >= count.it_changes_max:
                    print(f'\n\nMaximum number of changes reached: mutation = {count.mutation}, looser_same_formula = {count.looser_same_formula}')
                else: 
                    print('\n\nSuccess stopping condition!\n')

                ## !!! I made it independent of the database (not to check if the formula is new)
                # Currently, the following function has been modified to always output True 
                bool_is_new_formula = fun_create.check_if_formula_is_new(inputs, formula_best, set_database_parametric, set_database)
                if bool_is_new_formula:

                    #For previous implementation (only pos examples)
                    if inputs.kind_score in ['fitness', 'stl_robustness', 'efficient_monitoring', 'efficient_monitoring_stl']:
                        #For STL: Mine parameters & Replace the formula and the nodes with concrete values
                        if inputs.learning_stl:  tim, formula_best = fun_mono.mine_parameters_and_replace(count.max_number_par_symbols, formula_best, inputs, tim, traces_pos, formula_best.mono  )
                        formula_best, count, set_database_parametric = fun_create.strengthen_and_store_parametric(inputs, formula_best, grammar, monitor, count, traces_pos, bool_apply_syntactic_rules, set_database, set_database_parametric  )
                    elif inputs.kind_score in ['obj_function']: 
                        set_database_parametric = set_database_parametric.union(set([print_formula(formula_best.quantifiers, formula_best.string_param)]))
                    
                    '''
                    ## Remine parameteres with much lower tolerance = smaller error (and update the formula if cost is better)
                    # Instantiate the parameters in the formula
                    new_formula, tim, dict_par_inst = fun_metr.instantiate_parameters(inputs, formula_best, count, traces_pos, traces_neg, tim, dict_par_inst, max_fun_dir = 20, xtol = 0.03, ftol = 0.05)
                    #Compute new cost
                    time_start_monitoring = time.time()
                    new_formula.cost = fun_eval.cost(traces_pos, traces_neg, new_formula, inputs, monitor, count)
                    tim.monitor += (time.time() - time_start_monitoring)
                    formula_best = fun_upd.update_best_formula(inputs.kind_score, formula_best, new_formula)
                    with open(f'{inputs.name_output_folder}/{output_file_name}','a') as file: file.write(f'Formula after remine: {formula_best.string} with cost: {formula_best.cost}')
                    '''

                    # ACTIVE LEARNING (FALSIFICATION)
                    # print('Formula before active learning:', formula_best.string, 'Nodes:', gramm.tree_to_formula(formula_best.nodes), 'Cost:', formula_best.cost)
                    # formula_best, tim, count, numb_form_dict, inputs = refinement.active_loop(formula_best,  traces_pos, traces_neg, inputs, monitor, count, tim, numb_form_dict, numb_pred_dict, grammar)
                    # print('Formula after active learning: ', formula_best.string, 'Nodes: ', gramm.tree_to_formula(formula_best.nodes), 'Cost:', formula_best.cost)

                    #Store new formula
                    set_database , database = fun_create.store_new_formula(formula_best, inputs, count, set_database, database, path_to_save, time_start)
                    # Write new formula in the text output file
                    with open(f'{inputs.name_output_folder}/{output_file_name}','a') as file: file.write(f'\nFormula {len(database)}:\n {print_formula(database[-1].quantifiers, database[-1].string)}\n\n')
                    _ = final_eval.evaluation_single_formulas(database[-1].string, traces_pos, traces_neg, f'{inputs.name_output_folder}/{output_file_name}', case_study, bool_features = False)
                        
                    with open(f'{inputs.name_output_folder}/{output_file_name}','a') as file:
                        file.write(f'Pre-scaled:{database[-1].string}')
                        file.write(f'\nAfter-scaled:{final_eval.scale_node(case_study, database[-1].string, database[-1].variable_ranges)}\n')
                        file.write(f'\n\ncost_store:{store_cost}\n')
                        file.write(f'\nmsc_store:{store_msc}\n')
                        file.write(f'\nratio_fp_store:{store_ratio_fp}\n')
                        file.write(f'\nratio_fn_store:{store_ratio_fn}\n')

                    # Update time and count
                    tim = fun_upd.update_time_after_success(tim)  
                    count = fun_upd.update_count_after_success(count) # (count.hyper_monitor_tot , count.already_stored)
                    time_start = time.time()  
                 
                #If the formula is not new and is already in the dataset   
                else:  count.already_stored += 1
                     
                break #Exit the loop for mutating the formula - the loop is meant for violated formulas that, through mutations, need to be transformed into satisfied ones     
            
            numb_it_same_temperate = 0 #Number of iterations in which the temperature is the same    
            numb_it_same_temperate_to_do = fun_metr.fun_temp(inputs, 'numb_it', temperature)

            while (numb_it_same_temperate < numb_it_same_temperate_to_do) and (count.mutation + count.looser_same_formula + count.strengthened_same_formula) < count.it_changes_max:

                numb_it_same_temperate += 1 # Increase the number of iterations in which the temperature is the same
                bool_applied_rule, count, tim, numb_form_dict, new_formula    = \
                        fun_create.apply_changes(bool_apply_syntactic_rules, formula,grammar, inputs , count, tim,numb_form_dict, numb_pred_dict, numb_var, temperature) 
                print('New formula:', new_formula.string_param)
                # Instantiate the parameters in the formula
                new_formula, tim, dict_par_inst = fun_metr.instantiate_parameters(inputs, new_formula, count, traces_pos, traces_neg, tim, dict_par_inst)
                
                #Compute new cost
                time_start_monitoring = time.time()
                new_formula.cost,  msc, ratio_fp, ratio_fn = fun_eval.cost(traces_pos, traces_neg, new_formula, inputs, monitor, count, mode = 'msc')
                tim.monitor += (time.time() - time_start_monitoring)
                store_cost.append(new_formula.cost)
                store_msc.append(msc)
                store_ratio_fp.append(ratio_fp)
                store_ratio_fn.append(ratio_fn)

                with open(f'{inputs.name_output_folder}/{output_file_name}','a') as file:
                    file.write(f'\n\nFormula {len(database)+1}:\n  Iteration {count.mutation + count.looser_same_formula + count.strengthened_same_formula}\n')
                    file.write(f'{print_formula(new_formula.quantifiers, new_formula.string)}\n')
                    file.write(f'\nCost: {new_formula.cost}, msc = {msc}, ratio_fp ={ratio_fp}, ratio_fn = {ratio_fn}\n')
                    file.write(f'Time so far: {time.time() - time_beginning}\n')
                    file.write(f'Time for this formula: {time.time() - time_start}\n\n')

                # For the statistics
                # if fun_eval.check_success_stopping_condition(inputs, new_formula.cost, msc = msc, ratio_fp = ratio_fp, ratio_fn = ratio_fn ): 
                #         if bool_applied_rule: count.looser_formula_succ += 1 # If loosening the formula produced a SATISFIED formula:
                #         elif not bool_applied_rule: count.mutation_succ += 1 # If mutating the formula produced a SATISFIED formula:

                # The candidate formula is updated (or not) according to the Metropolis-Hastings acceptance ratio
                formula = fun_upd.update_candidate_formula(inputs, formula, new_formula, temperature)
                
                ## !!! Made it independent of the database (not to check if the formula is new)
                # Currently, the following function has been modified to always output True 
                bool_new_formula_is_new = fun_create.check_if_formula_is_new(inputs, new_formula, set_database_parametric, set_database)
                # If the new formula is new, update the best formula found so far (if better cost)
                if bool_new_formula_is_new: formula_best = fun_upd.update_best_formula(inputs.kind_score, formula_best, new_formula)
                
                if fun_eval.check_success_stopping_condition(inputs, formula_best.cost, msc = msc, ratio_fp = ratio_fp, ratio_fn = ratio_fn ):
                    break # Exit the loop with same temperature
            
            temperature = fun_metr.fun_temp(inputs, 'update', temperature)

        #JUST EXITED LOOP 2.1 (Loop of mutations to find a satisfied formula)
            
    #JUST EXITED LOOP 1 (Loop to find a set of satisfied formulas)
    
    #Order database according to the cost value
    if not inputs.store_enumeration_formulas: shutil.rmtree(f'{inputs.name_output_folder}/grammar')

    fun_upd.write_final_metrics(f'{inputs.name_output_folder}/{output_file_name}', tim, count)

    return database, inputs

