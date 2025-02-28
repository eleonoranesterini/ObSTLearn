import os
import pickle
import itertools
import sys
import numpy as np
import shutil

import functions.create_grammar as gramm

def set_up_paths(inputs):
    # Eliminate previous grammar
    if os.path.exists(f'{inputs.name_output_folder}') and os.path.exists(f'{inputs.name_output_folder}/grammar'): shutil.rmtree(f'{inputs.name_output_folder}/grammar')

    if not os.path.exists(f'{inputs.name_output_folder}'):  os.mkdir(f'{inputs.name_output_folder}')
    if not os.path.exists(f'{inputs.name_output_folder}/grammar'): os.mkdir(f'{inputs.name_output_folder}/grammar')
    
    
    if inputs.store_formulas and not os.path.exists(f'{inputs.name_output_folder}/Learned_formulas'): os.mkdir(f'{inputs.name_output_folder}/Learned_formulas')
    path_to_save = f'{inputs.name_output_folder}/Learned_formulas/learned_formula_seed{inputs.seed}/' 
    if inputs.store_formulas and not os.path.exists(path_to_save): os.mkdir(path_to_save)
    
    return path_to_save

def list_tuples_sum(num_addend,value_sum):
    
    ''' The function computes all possible tuples of num_addend elements whose sum is equal to value_sum
    INPUTS: 
        - num_addend : number of elements that are added
        - value_sum : result of the sum
    '''
    
    tuples = [] #1,value_sum-num_addend
    
    for comb in itertools.product(range(1, value_sum-num_addend+2), repeat =num_addend):
        if sum(comb) == value_sum: tuples.append(comb)
    return tuples



def enumerate_formulas(length, spec_level, grammar_structure, inputs ):
                 
       # bool_same_number
    ''' 
    The function enumerates and stores all the formulas 
    with given length and with the given spec_level,
    together with all the other sets of formulas 
    required for this enumeration (smaller lengths and/or greater spec level)'''
    
    if not os.path.exists(f'{inputs.name_output_folder}/grammar'): os.mkdir(f'{inputs.name_output_folder}/grammar')
    if not os.path.exists(f'{inputs.name_output_folder}/grammar/spec_level{spec_level}'): os.mkdir(f'{inputs.name_output_folder}/grammar/spec_level{spec_level}')
    
    if not os.path.exists(f'{inputs.name_output_folder}/grammar/spec_level{spec_level}/length_{length}'): os.mkdir(f'{inputs.name_output_folder}/grammar/spec_level{spec_level}/length_{length}')
    elif len(os.listdir(f'{inputs.name_output_folder}/grammar/spec_level{spec_level}/length_{length}')) == 0 : 
         # print(f'\nNo formula possible with spec_level = {spec_level} and length = {length}\n')
         return
    else: return # Already enumerated 
    
    path = f'{inputs.name_output_folder}/grammar/spec_level{spec_level}/length_{length}'
    len_database = 0
    
    list_number_predicates = []
    
    current_spec =  grammar_structure[spec_level]
    for index, operator in enumerate(current_spec):
        
        if length >= current_spec[index].count('phi') + 1:
        
            if 'predicate' in operator: 
              
                if length == 1: 
                     grammar_level = int(operator[operator.index('predicate') + 9 : ])
                     
                     current_node = [gramm.BinaryTreeNode(operator)] #'predicate'
                     current_node[-1].level_grammar = spec_level
                     # current_node[-1].location = [spec_level, length, len_database]
                     current_node[-1].start_subformula = [-1, -1] # no subformula for the predicates 
                     
                     ## !!!
                     aux = [0] * 10 #10 as max number of grammar_predicates
                     aux[grammar_level] = 1
                     current_node[-1].predicates = aux.copy()
                    
                     with open(f'{path}/nodes_{len_database}.obj', 'wb') as f: 
                            list_number_predicates.append(current_node[-1].predicates)
                            pickle.dump(current_node, f)
                            len_database += 1
                             
            #Unary operators
            elif'not' in operator or \
                'always' in operator or 'eventually' in operator or 'historically' in operator or 'once' in operator or \
                's_next' in operator  or 'next' in operator or 's_prev' in operator or 'prev' in operator :
                
                next_spec = int(operator[operator.index('phi') + 3])
                
                current_node = gramm.BinaryTreeNode(operator.replace(f' phi{next_spec}',''))
                current_node.level_grammar = spec_level
                
                enumerate_formulas(length - 1 , next_spec , grammar_structure, inputs)
                list_formulas = os.listdir(f'{inputs.name_output_folder}/grammar/spec_level{next_spec}/length_{length - 1}')
                list_formulas.sort()
                
                for i in range(len(list_formulas)):
                    with open(f'{inputs.name_output_folder}/grammar/spec_level{next_spec}/length_{length - 1}/{list_formulas[i]}', 'rb') as f: prev_nodes = pickle.load(f)
                    nodes = [current_node] + prev_nodes # concatenation of lists 
                    nodes[1].parent = nodes[0]
                    nodes[0].leftchild = nodes[1]
                    
                    nodes[0].predicates = (nodes[1].predicates).copy()
                    # nodes[0].location = [spec_level , length, len_database]
                    nodes[0].start_subformula = [1, -1] # subformula1 starting in 1, subformula2 not existing 
                    
                    if not('not' in operator) or not('not' in nodes[1].data):  #Avoid double negation
                        # Store
                        with open(f'{path}/nodes_{len_database}.obj', 'wb') as f:
                            list_number_predicates.append(nodes[0].predicates)
                            pickle.dump(nodes, f)
                            len_database += 1
                    
            #Binary operators
            else: 
                tuples_of_length = list_tuples_sum( 2 , length-1 ) # all possibilities
                next_spec1 = int(operator[operator.index('phi') + 3 ])
                next_spec2 = int(operator[operator.index('phi', operator.index('phi') + 3 ) + 3 ])
                
                for tuples in tuples_of_length:
                    
                    if '<->' in operator or   'implies' in operator or \
                        'until' in operator or  'since' in operator or  'weak_u' in operator or\
                        'or' in operator or  'and' in operator: 
                          
                        current_spec_copy = operator.replace(f'phi{next_spec1} ','')
                        current_spec_copy = current_spec_copy.replace(f' phi{next_spec2}','')
                        current_node = gramm.BinaryTreeNode(current_spec_copy)
                        current_node.level_grammar = spec_level
                        
                        enumerate_formulas(tuples[0] , next_spec1, grammar_structure, inputs)         
                        list_formulas_left = os.listdir(f'{inputs.name_output_folder}/grammar/spec_level{next_spec1}/length_{tuples[0]}')
                        list_formulas_left.sort()
                        
                        database_left = []
                        for i in range(len(list_formulas_left)):
                            with open(f'{inputs.name_output_folder}/grammar/spec_level{next_spec1}/length_{tuples[0]}/{list_formulas_left[i]}', 'rb') as f: 
                                database_left.append(pickle.load(f))
                        
                        enumerate_formulas(tuples[1] , next_spec2, grammar_structure, inputs)   
                        list_formulas_right = os.listdir(f'{inputs.name_output_folder}/grammar/spec_level{next_spec2}/length_{tuples[1]}')
                        list_formulas_right.sort()
                                
                        database_right = []
                        for j in range(len(list_formulas_right)):
                            with open(f'{inputs.name_output_folder}/grammar/spec_level{next_spec2}/length_{tuples[1]}/{list_formulas_right[j]}', 'rb') as f: 
                                database_right.append(pickle.load(f))
                        
                    
                        for item_left in database_left:
                            for item_right in database_right:
                        
                                nodes = [current_node] + item_left
                                nodes[1].parent = nodes[0]
                                nodes[0].leftchild = nodes[1]
                                
                                index_right = len(nodes)
                                nodes = nodes + item_right
                                nodes[index_right].parent = nodes[0]
                                nodes[0].rightchild = nodes[index_right]
                                
                                nodes[0].predicates = [ a + b for (a,b) in zip(nodes[index_right].predicates, nodes[1].predicates)]
                                # nodes[0].location = [spec_level , length, len_database]
                                nodes[0].start_subformula = [1, index_right] #subformula1 starting in 1, subformula2 starting in index_right
                                
                                with open(f'{path}/nodes_{len_database}.obj', 'wb') as f:
                                    list_number_predicates.append(nodes[0].predicates)
                                    pickle.dump(nodes, f)
                                    len_database += 1
                    else:  print('Used a non-existing operator: \n\nADD NEW OPERATOR!!\n\n')
    
    with open(f'{inputs.name_output_folder}/grammar/spec_level{spec_level}/predicates_length{length}', 'wb') as f:  pickle.dump(list_number_predicates, f)
    
    return
    
    
def enumerate_predicates(grammar_predicate, inputs, m):
    
    '''The function is used to enumerate all predicates. 
        
    INPUTS:
        
        - grammar_predicate: list of predicates levels = [ predicate0, predicate1, predicate2, ... ]
                              where predicatei = [[rel_i_0, var_i_0, pair_traces_i_0], [rel_i_1, var_i_1, pair_traces_i_1],...], where
                                  rel : relational signs admitted
                                  var : admitted variables
                                  pair_traces: pair of traces admitted
                              Example of need for different levels for the predicates:  ['(predicate1 Until predicate2) or (always predicate1)'] for the weak until
                                        
        - bool_different_variables: True if a predicate can involve different variables (e.g. pi(x) = pi(y));
                   
        - second_variable: list 'true' if the second variable should be True,
                            'actions' if the second variable should be a number representing a certain action (see dining philophers)
                            'variable' for any other variable
        
    '''
    
    if not os.path.exists(f'{inputs.name_output_folder}/grammar'): os.mkdir(f'{inputs.name_output_folder}/grammar')
    if not os.path.exists(f'{inputs.name_output_folder}/grammar/predicates'): os.mkdir(f'{inputs.name_output_folder}/grammar/predicates')
    if not os.path.exists(f'{inputs.name_output_folder}/grammar/predicates/{m}_quant'): os.mkdir(f'{inputs.name_output_folder}/grammar/predicates/{m}_quant')
    
    if not inputs.learning_stl :
        for predicate_level , item_level in enumerate(grammar_predicate):
            len_database = 0
            
            path = f'{inputs.name_output_folder}/predicates/{m}_quant/predicate_level{predicate_level}/'
            if not os.path.exists(f'{path}'): os.mkdir(f'{path}')
            elif len(os.listdir(f'{path}')) == 0 : 
                 # print(f'\nNo formula possible with predicate_level = {predicate_level}\n')
                 continue
            else: continue # Already enumerated 
        
            #Different options in the same level
            for sublevel, item_sublevel in enumerate(grammar_predicate[predicate_level]): 
                # Enumerate the signs symbol in the grammar level '[level]' and sub-level '[sublevel]'
                signs_enum = grammar_predicate[predicate_level][sublevel][0]
                
                #Adjust the tuples of traces
                #Enumerate the tuples of trace variables
                if inputs.second_variable[predicate_level] == 'variable': trace_variables_enum = grammar_predicate[predicate_level][sublevel][2]
                else: trace_variables_enum = list(set([el[0] for el in grammar_predicate[predicate_level][sublevel][2]]))
                
                # Adjust the variables
                if inputs.second_variable[predicate_level] == 'variable':
                    # Enumerate the pair of variables
                    if inputs.bool_different_variables: #Different variables admitted in the predicates
                        variables_enum = [ el for el in itertools.product(grammar_predicate[predicate_level][sublevel][1], repeat =2)]
                    else:  # No different variables admitted in the predicates
                        variables_enum = [(el,el) for el in grammar_predicate[predicate_level][sublevel][1]]
                
                elif inputs.second_variable[predicate_level] == 'true': variables_enum = grammar_predicate[predicate_level][sublevel][1]
                else: variables_enum = [(a,b) for a in grammar_predicate[predicate_level][sublevel][1] for b in inputs.second_variable[predicate_level]]
                
                for sign in signs_enum:
                    for trace_variables in trace_variables_enum:
                        for variables in variables_enum:
                            if  inputs.second_variable[predicate_level] == 'true': 
                                value = f'pi[{trace_variables}][{variables}] {sign} True '
                            elif inputs.second_variable[predicate_level] == 'variable': 
                                if len(trace_variables) < 2: value = f'pi[{trace_variables[0]}][{variables[0]}] {sign} pi[{trace_variables[0]}][{variables[1]}] '
                                else: value = f'pi[{trace_variables[0]}][{variables[0]}] {sign} pi[{trace_variables[1]}][{variables[1]}] '
                            else:
                                value = f'pi[{trace_variables}][{variables[0]}] {sign} {variables[1]} '
                            
                            current_node = [gramm.BinaryTreeNode(value)] 
                            
                            with open(f'{path}/nodes_{len_database}.obj', 'wb') as f: 
                                    pickle.dump(current_node, f)
                                    len_database += 1
    #Learning STL                                
    else: 
        for predicate_level , item_level in enumerate(grammar_predicate):
            
            path = f'{inputs.name_output_folder}/grammar/predicates/{m}_quant/predicate_level{predicate_level}/'
            if not os.path.exists(f'{path}'): os.mkdir(f'{path}')
            elif len(os.listdir(f'{path}')) == 0 : 
                 # print(f'\nNo formula possible with predicate_level = {predicate_level}\n')
                 continue
            else: continue # Already enumerated 
            
            len_database = 0
            
            for value in item_level:
                    current_node = [gramm.BinaryTreeNode(value)] 
                    with open(f'{path}/nodes_{len_database}.obj', 'wb') as f: 
                                    pickle.dump(current_node, f)
                                    len_database += 1
       
    return 


def compute_number_predicates(m, inputs):
    
    ''' Function enumerate_predicates needs to be run before this function is called.
    m is the number of quantifiers. '''

    path = f'{inputs.name_output_folder}/grammar/predicates/{m}_quant'
    
    levels = len(os.listdir(path))
    list_number = []
    
    for predicate_level in range(levels): list_number.append(len(os.listdir(f'{path}/predicate_level{predicate_level}')))
    
    return list_number



def compute_number_formulas(spec_level, m, length, numb_pred_dict, inputs ):
    
    path = f'{inputs.name_output_folder}/grammar/spec_level{spec_level}/length_{length}'
    with open(f'{inputs.name_output_folder}/grammar/spec_level{spec_level}/predicates_length{length}', 'rb') as f:  list_number_predicates = pickle.load(f)
    
    
    numb_template_formulas = len(os.listdir(f'{path}'))
    
    list_number = []
    
    for index in range(numb_template_formulas):
        el = [ b**a for (a,b) in zip( list_number_predicates[index] , numb_pred_dict[f'{m}' ])]        
        list_number.append(np.prod(el))         
                 
    return list_number
