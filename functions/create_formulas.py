import os
import sys
import random
import copy
import time
import math
import pickle
import numpy as np
import itertools

import functions.enumerate_formulas as fun_enum
import functions.sample_formulas as fun_sample
import functions.create_grammar as gramm
import functions.monotonicity_functions as fun_mono
import functions.syn_rules as syn

def print_formula(quantifiers, formula_string):
    
    numb_quant = len(quantifiers)
    f = ''
    for i in range(numb_quant):
        if quantifiers[i]==0: f += f'E pi[{i}] '
        elif quantifiers[i]==1: f += f'FA pi[{i}] '
    
    f = f + ': ' + formula_string
    return f


def create_syntactically_valid_starting_formula(inputs, grammar, numb_quant, numb_var, length, spec_level_start, count, tim, numb_form_dict, numb_pred_dict):
    
    ''' The function creates a formula that respect the given grammar - i.e., syntactically valid'''
    
    ## REJECTION SAMPLING FOR NOT WELL-FORMED FORMULA
    # not well-formed means:
    # 1) Trivial (when there are not temporal operators, this is checked with Z3)
    # 2) The formula is missing elements that need to be present, specified by the user
    # 3) Not all trace variables named by the quantifiers appear in the internal formula 
    # 4) For STL formulas, the formula is not monotonic in all its parameters

    initial_length = length
    bool_new_min_length = False #True if the length of the formula has been increased
    count.num_rej = 0
        
    ##Loop to find a valid starting formula (LOOP 2.1)
    while True:  #REJECTION SAMPLING to find a VALID STARTING formula (SYNTACTICALLY)
            count.num_rej += 1  
            #If formulas with given {spec_level_start} and {length} have not been enumerated yet , do so
            if not os.path.exists(f'{inputs.name_output_folder}/grammar'): os.mkdir(f'{inputs.name_output_folder}/grammar')
            if not os.path.exists(f'{inputs.name_output_folder}/grammar/spec_level{spec_level_start}'): os.mkdir(f'{inputs.name_output_folder}/grammar/spec_level{spec_level_start}')
            if not os.path.exists(f'{inputs.name_output_folder}/grammar/spec_level{spec_level_start}/length_{length}'): 
                time_start_enumeration = time.time()
                fun_enum.enumerate_formulas(length, spec_level_start, grammar.structure, inputs )
                tim.enumerating_formulas += (time.time() - time_start_enumeration) 
            
            time_start_enumeration = time.time()   
            
            #Update the dictionary - used for sampling
            if f'{spec_level_start}_{numb_quant}_{length}' not in numb_form_dict: 
                    numb_form_dict[f'{spec_level_start}_{numb_quant}_{length}'] = (fun_enum.compute_number_formulas(spec_level_start, numb_quant, length, numb_pred_dict, inputs)).copy()
            tim.enumerating_formulas += (time.time() - time_start_enumeration)
                            
            time_start_gen = time.time()  
            
            #Create the formula
            if len(os.listdir(f'{inputs.name_output_folder}/grammar/spec_level{spec_level_start}/length_{length}')) != 0:
                
                list_formulas = os.listdir(f'{inputs.name_output_folder}/grammar/spec_level{spec_level_start}/length_{length}')
                list_formulas.sort()
                #In the following, weights are used to weight formulas without concrete predicates depending on the number of the corresponding number of concrete formulas they can generate
                sampled_f = random.choices(list_formulas, weights = numb_form_dict[f'{spec_level_start}_{numb_quant}_{length}'] , k = 1)[0]
                with open(f'{inputs.name_output_folder}/grammar/spec_level{spec_level_start}/length_{length}/{sampled_f}', 'rb') as f: nodes = pickle.load(f)
                
                #Sample the predicate and instantiate a concrete formula
                nodes = fun_sample.sample_predicate(grammar.predicate, nodes, inputs.bool_different_variables,  inputs.second_variable, numb_quant, inputs)
                string_param = gramm.tree_to_formula(nodes)
                
                #Check the presence of the predicates that need to be present (either user-specified or all trace variables named by the quantifiers)
                if inputs.to_be_present is not None : to_be_present = copy.deepcopy(inputs.to_be_present)
                else: to_be_present = [ [f'pi[{j}]' for j in range(numb_quant) ] , numb_quant]

                if sum([to_be_present[0][i] in string_param for i in range(len(to_be_present[0]))])>=to_be_present[1]:
                    mono = None      
                        
                    if inputs.learning_stl and not inputs.kind_score in ['obj_function']:
                        #Check monotonicity for stl.
                        # If monotonic, store the kind of monotoniity for each parameter. Otherwise rejection sampling
                        mono = fun_mono.check_monotonicity(nodes)
                        if mono != False: # ok monotonic
                            tim.gen_formula += (time.time() - time_start_gen)
                            break

                    else: #no computation of mono in this case
                        tim.gen_formula += (time.time() - time_start_gen)
                        break
                            
                #Too many rejected -> sample new length for the formula        
                if count.num_rej > count.numb_rej_max + 1:
                    
                    #Change length if not forced with one user-specified length or if it is forced to a length but stuck
                    if inputs.length is None or count.num_rej > 50 * (count.numb_rej_max): 
                        length = random.randint(2*math.ceil(numb_quant/2)-1,inputs.max_length) # sample the length of the formula
                        count.num_rej = 0
                    
                    tim.gen_formula += (time.time() - time_start_gen)
            
            #If there are no formulas with that length: 
            # first increase formula length; if restarted with length =1, then stop when reaching initial length (those checks have already been done)
            elif length < inputs.max_length and (not bool_new_min_length or length < initial_length): 
                length += 1
                tim.gen_formula += (time.time() - time_start_gen)
               
            elif length == inputs.max_length and not bool_new_min_length:
                length = 1 #reset the length to the minimum
                bool_new_min_length = True
                tim.gen_formula += (time.time() - time_start_gen)
            
            else:
                print('No valid initial formula found within the length bounds.\n Change the length bounds or the grammar!\n')
                sys.exit()

    return  count, tim, numb_form_dict, length , nodes , string_param, mono 



def create_starting_formula(inputs, grammar, initial_grammar_predicate, numb_pred_dict, numb_form_dict, count, tim, numb_var ):
    
    ''' The function initializes the parameters for the creation of a formula, 
        enumerates predicates and formulas, 
        and creates a syntactically valid starting formula (calling the function create_syntactically_valid_starting_formula).
        '''
    
    ##INITIALIZATION
    #Sample the quantifiers
    quantifiers , quantifier_levels,  spec_level_start = fun_sample.sample_quantifiers(grammar.quantifiers, inputs.numb_quantifiers[0], inputs.numb_quantifiers[1])
    # quantifier_levels = [#num_quantifiers , ['string quant 1 ', level_in_gramm  ], ['string quant 2', level_in_gramm] , [,]]
    # e.g. [4, [' E', 0], [' A A ', 0], [' E ', 0], [' phi0', 0]]
    
    numb_quant = len(quantifiers)
    # print(f'Number of quantifiers = {numb_quant}')
    
    # Enumerate the admissible tuple of traces
    if not inputs.learning_stl: 
        grammar.predicate = copy.deepcopy(initial_grammar_predicate)
        #Replace 'alltuples' in the grammar.predicate
        for index_gr in range(len(grammar.predicate)):
            if 'all tuples' in initial_grammar_predicate[index_gr][0][-1]: grammar.predicate[index_gr][0][-1] = [ item for item in itertools.permutations(range(numb_quant), min(numb_quant,2))]
        
    elif inputs.learning_stl: #For STL, predicates are assumed to be already enumerated
    #Depending on the number of quantifiers samples, allow only predicates that involve trace variables <= than the number of quantifiers
        grammar.predicate = []
        for index_gr in range(len(initial_grammar_predicate)):
            aux = []
            for item_gr in initial_grammar_predicate[index_gr]:
                if not any([i_trace in item_gr for i_trace in [f'pi[{numb_quant}]',f'pi[{numb_quant+1}]', f'pi[{numb_quant+2}]', f'pi[{numb_quant+3}]']]): aux.append(item_gr)
            grammar.predicate.append(aux)
    
    # Enumerate and store all the predicates        
    fun_enum.enumerate_predicates(grammar.predicate,  inputs, numb_quant)
    
    #Update dictionary with key (#quantifiers) and value: list of #concrete predicates per grammar level 
    if f'{numb_quant}' not in numb_pred_dict: numb_pred_dict[f'{numb_quant}'] = (fun_enum.compute_number_predicates(numb_quant, inputs)).copy()
    #Set the length of formula structure
    if inputs.length is not None:  length = inputs.length
    else:  # Sample the length for the formula structure 
        length = random.randint(2*math.ceil(numb_quant/2)-1, inputs.max_length) # length of the formula
        ### !!! Fitness does not accept length = 1 (?)
        if inputs.kind_score == 'fitness' and length == 1: length = 2
    

    count, tim, numb_form_dict, length , nodes , string_param, mono = create_syntactically_valid_starting_formula(inputs, grammar,numb_quant, numb_var, length, spec_level_start, count, tim, numb_form_dict, numb_pred_dict)

    formula = gramm.Formula(quantifiers, quantifier_levels, string_param, nodes, length, mono)   #for stl, nodes is NOT instantiated , nor string_param it is
         
    return formula , grammar, count, tim, numb_form_dict, numb_pred_dict


def replace_subtree(formula, indices_old_node, length_node, grammar, inputs, count, tim, numb_form_dict, numb_pred_dict):

    ''''The function replaces the subformula (individuated by the indices indices_old_nodes in the formula.nodes list) 
    of the formula with a new subtree (randomly generated such that the final formula has a max length and belongs to the grammar).'''

    time_start_changes = time.time()

    # Number of quantifiers
    numb_quant = len(formula.quantifiers)

    #Select grammar level of the node to be deleted
    spec_level_change = formula.nodes[indices_old_node[0]].level_grammar

    #Loop to find a valid new formula
    # (LEVEL 3.1) - SYNTACTICALLY VALID MUTATED FORMULA - same criteria as loop 2.1
    count.num_rej = 0

    while True:

        count.num_rej += 1
        # After too many iterations -> give up and do not apply changes        
        if count.num_rej > count.numb_rej_max :
            print('Could not find a valid mutated formula. Mutations are not applied for this iteration.\n')
            new_string_param = formula.string_param
            new_nodes = copy.deepcopy(formula.nodes)
            new_mono = formula.mono
            break

        # Sample the length of the new subtree (such as the final lenght of the new formula is not greater than the maximum length)
        admissible_new_lengths = range(1, inputs.max_length - (formula.length - length_node) + 1)
        length_new_subtree = random.choices(admissible_new_lengths,  k = 1)[0]
        
        # If the formulas with corresponding {spec_level_change} and {length_new_subtree} have not been enumerated yet, do so
        if not os.path.exists(f'{inputs.name_output_folder}/grammar/spec_level{spec_level_change}'): os.mkdir(f'{inputs.name_output_folder}/grammar/spec_level{spec_level_change}')
        if not os.path.exists(f'{inputs.name_output_folder}/grammar/spec_level{spec_level_change}/length_{length_new_subtree}'): 
            tim.apply_changes += (time.time() - time_start_changes)
            time_start_enumeration = time.time()
            fun_enum.enumerate_formulas(length_new_subtree, spec_level_change, grammar.structure, inputs)
            tim.enumerating_formulas += (time.time() - time_start_enumeration)
        time_start_enumeration = time.time()
        #Update the dictionary (for storing the monitored subformulas):
        if f'{spec_level_change}_{numb_quant}_{length_new_subtree}' not in numb_form_dict:  
            numb_form_dict[f'{spec_level_change}_{numb_quant}_{length_new_subtree}'] = (fun_enum.compute_number_formulas(spec_level_change, numb_quant, length_new_subtree, numb_pred_dict, inputs)).copy()   
        tim.enumerating_formulas += (time.time() - time_start_enumeration)

        time_start_changes = time.time()
        list_changed_formulas = os.listdir(f'{inputs.name_output_folder}/grammar/spec_level{spec_level_change}/length_{length_new_subtree}')
        if len(list_changed_formulas) == 0: continue #go to next iteration of the loop (different length)

        list_changed_formulas.sort()
        sampled_f = random.choices(list_changed_formulas,weights = numb_form_dict[f'{spec_level_change}_{numb_quant}_{length_new_subtree}'], k = 1)[0]
        with open(f'{inputs.name_output_folder}/grammar/spec_level{spec_level_change}/length_{length_new_subtree}/{sampled_f}', 'rb') as f:   new_node = pickle.load(f)
            
        new_node = fun_sample.sample_predicate(grammar.predicate, new_node, inputs.bool_different_variables, inputs.second_variable, numb_quant , inputs)

        #Embed the new nodes inside the tree of the original formula
        new_nodes = fun_sample.change_node(formula.nodes, indices_old_node, new_node)
            
        #Transform the new tree into a string
        new_string_param = gramm.tree_to_formula(new_nodes)
        
        #Check the presence of the predicates that need to be present (either user-specified or all trace variables named by the quantifiers)
        if inputs.to_be_present is not None : to_be_present = copy.deepcopy(inputs.to_be_present)
        else: to_be_present = [ [f'pi[{j}]' for j in range(numb_quant) ] , numb_quant]
            
        if formula.string_param != new_string_param and \
            sum([to_be_present[0][i] in new_string_param for i in range(len(to_be_present[0]))]) >= to_be_present[1]: #2 <-> numb_quant
            
            if not inputs.learning_stl or inputs.kind_score in ['obj_function']:
                new_mono = None # do not check for monotonicity
                break
            else:# CHECK MONOTONICITY FOR STL
                new_mono = fun_mono.check_monotonicity(new_nodes)
                if new_mono != False: break  # ok monotonic

    new_formula = gramm.Formula(formula.quantifiers, formula.quantifier_levels, new_string_param, new_nodes, len(new_nodes) , mono = new_mono , string = new_string_param)
    tim.apply_changes += (time.time() - time_start_changes)

    return new_formula, count, tim, numb_form_dict


def apply_changes(bool_apply_syntactic_rules, formula, grammar, inputs , count, tim, numb_form_dict, numb_pred_dict, numb_var, temperature):
    
    
    '''
    The function applies changes to the formula structure or to the quantifiers.
    The changes are applied according to the following rules:
    - first, syntactic rules are applied to loosen the formula
    - second, if syntactic rules are not applied, random changes are applied to the formula structure or to the quantifiers with probability 0.5
    The function returns the new formula and the updated counters.
    Note: the function does not check whether the new formula is satisfied or not.
    The function returns None as new_formula if the maximum number of iterations is reached (in this case the main funciton will start with another formula)
    '''
    # Start time for changes
    time_start_changes = time.time()   

    # Random number to determine whether to apply syntactic rules or not (0.1 prob for former, 0.9 for latter)
    prob_kind_change = random.uniform(0,1) 

    #Initialize the boolean variable that indicates whether syntactic rules have been applied
    bool_applied_rule = False

    ### NO SYNTACTIC RULES APPLIED
    #If allowed (bool_apply_syntactic_rules) and with probability 0.1, apply syntactic rules
    if bool_apply_syntactic_rules and (prob_kind_change < 0): ### NO SYNTACTIC RULES APPLIED
        loosening_or_strengthening = random.uniform(0,1) # To decide whether to apply loosening or strengthening
        if loosening_or_strengthening < 0.5:
            #print('Apply loosening')
            bool_applied_rule, new_formula = syn.looser(formula, grammar, inputs)
            #if Syntactic rules have been applied -> update the counter
            if bool_applied_rule: count.looser_same_formula += 1
        elif loosening_or_strengthening >= 0.5:
            #print('Apply strengthening')
            bool_applied_rule, new_formula = syn.strengthen(formula, grammar, inputs)
            #if Syntactic rules have been applied -> update the counter
            if bool_applied_rule: count.strengthened_same_formula += 1

    #if Syntactic rules not applied -> apply random changes
    # either because it is not allowed to apply syntactic rules or because the random number is greater than 0.1
    if not bool_applied_rule:
        #print('Apply mutation to ', formula.string_param)
        
        # With probability 0.5, the changes are applied to the formula structure
        # The changes are applied to the formula structure also if the user explictly violates the mutation of quantifiers by setting bool_mutation_quantifiers to False
        if not inputs.bool_mutation_quantifiers or random.uniform(0,1) >= 0.5: 
            
            #Select the nodes that will be deleted
            selected_index = fun_sample.select_node(formula.nodes, temperature)
            indices_old_node, length_node = fun_sample.find_subtree(formula.nodes, selected_index)
            tim.apply_changes += ( time.time() - time_start_changes )

            #Replace the subtree with a new subtree
            new_formula, count, tim, numb_form_dict = replace_subtree(formula, indices_old_node, length_node, grammar, inputs, count, tim, numb_form_dict, numb_pred_dict)
            time_start_changes = time.time()
            
            
        #Otherwise, changes are applied to the formula's quantifiers
        else: 
            new_quantifiers, new_quantifier_levels = fun_sample.sample_new_quantifiers(formula , grammar.quantifiers)
            new_formula = gramm.Formula(new_quantifiers, new_quantifier_levels, formula.string_param, formula.nodes, len(formula.nodes) , mono = formula.mono, string = formula.string_param) 
       
        count.mutation_tot += 1
        count.mutation += 1           

    tim.apply_changes += ( time.time() - time_start_changes )

    return bool_applied_rule, count, tim, numb_form_dict, new_formula  



def check_if_formula_is_new(inputs, formula, set_database_parametric, set_database):
    
    '''
    The function checks whether the formula is new or not (whether it is already in the dataset or not).
    The function returns True if the formula is new, False otherwise.

    '''
    ## !!!

    #In case of STL for the previous version of the algorithm (satisfaction only)
    if inputs.kind_score in ['fitness', 'stl_robustness', 'efficient_monitoring', 'efficient_monitoring_stl'] \
        and inputs.learning_stl:
    #if inputs.learning_stl:
            # formula.string_param = gramm.tree_to_formula(formula.nodes)#parametric formula
            # If the parameteric formulas has not been used yet to instantiate a valid formula     
            if not set([print_formula(formula.quantifiers, formula.string_param)]).issubset(set_database_parametric):
                bool_is_new_formula = True
            # If the parameteric formulas has already been used to instantiate a valid formula     
            else:  bool_is_new_formula = False
    
    #In case of NOT STL (e.g., LTL)
    # or in case of objective function
    else:
        #Check if the formula is already in the database:
        if  set([print_formula(formula.quantifiers,  formula.string)]).issubset(set_database): bool_is_new_formula = False
        else: bool_is_new_formula = True

    return True #bool_is_new_formula #, formula


    
def strengthen_and_store_parametric(inputs, formula, grammar, monitor, count, traces_pos, bool_apply_syntactic_rules, set_database,set_database_parametric  ):
    
    '''USED FOR FORMULA_BEST -> make it less general'''

    #Check whether to apply changes to strengten the formula according to the syntactic rules
    bool_applied_rule = False

    if inputs.kind_score in ['fitness', 'stl_robustness', 'efficient_monitoring', 'efficient_monitoring_stl'] \
      and bool_apply_syntactic_rules: 
        # The counter for number of attempted stricter formulas is inside the following function
        bool_applied_rule, formula = syn.strengthen(formula, grammar , inputs, bool_pos_ex_only = True , monitor = monitor, count = count, traces_pos = traces_pos, set_database= set_database)
        if bool_applied_rule : count.strengthened_formulas_succ +=1
        # If changes have been applied and parametric stl-> add the structure of the new formula to set_database_parametric
        if bool_applied_rule and inputs.learning_stl: set_database_parametric = set_database_parametric.union(set([print_formula(formula.quantifiers, gramm.tree_to_formula(formula.nodes_param) )]))
        
    # If changes have NOT been applied -> add the structure of the original formula to set_database_parametric
    if not bool_applied_rule and inputs.learning_stl: set_database_parametric = set_database_parametric.union(set([print_formula(formula.quantifiers, formula.string_param)]))
    
    return formula, count, set_database_parametric



def store_new_formula(formula, inputs, count, set_database, database, path_to_save, time_start):
    
    ''' USED FOR FORMULA BEST
    The function stores the new formula in the database and updates the counters.
    The function returns the updated counters and the updated database.
    '''              

    time_end = time.time()  

    #Store the new learned formula
    formula.time = time_end - time_start
    formula.numb_mutation = count.mutation
    formula.numb_hyper_monitor = count.hyper_monitor
    formula.variable_ranges = copy.deepcopy(inputs.variable_ranges)
    
    #Update database
    set_database = set_database.union(set([print_formula(formula.quantifiers,  formula.string)]))
    database.append(formula)
    print(f'\n\nFormula {len(database)}: {print_formula(formula.quantifiers, formula.string)}')#'; numb_mutation = {count.mutation}, numb_hyper_monitor  = {count.hyper_monitor}, length={formula.length}, time = {time_end-time_start}\n')

    #Store only if explicitely asked
    if inputs.store_formulas:
        with open(f'{path_to_save}formula_to_be_stored{len(database)}.obj', 'wb') as f: pickle.dump(formula, f)
                       
    return set_database , database

