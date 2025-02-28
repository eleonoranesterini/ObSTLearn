import random
import copy
import os
import pickle

def sample_predicate(grammar_predicate, nodes, bool_different_variables, second_variable, m, inputs):
   
    '''The function is used to sample predicates in case the user specifies if some 
    predicates need to be equal or they need to be different by imposing for examples:
        
        ['(predicate1 Until predicate2) or (always predicate1)'] for the weak until
        
    INPUTS:
        - grammar_predicate: list of predicates levels = [ predicate0, predicate1, predicate2, ... ]
                              where predicatei = [[rel_i_0, var_i_0, pair_traces_i_0], [rel_i_1, var_i_1, pair_traces_i_1],...], where
                                  rel : relational signs admitted
                                  var : admitted variables
                                  pair_traces: pair of traces admitted
                                                         
            
        - nodes: nodes (tree-structured) representing the formula;
            
        - bool_different_variables: True if a predicate can involve different variables (e.g. pi(x) = pi(y));
                
        #-bool_same_number: True if whenever the same predicate name is used (e.g. 'predicate2'), all
                            nodes with the same name has to be equal;
                            BUT the presence of two different names (e.g. 'predicate1' and 'predicate2')
                            does not necessarily implies they are different.
        - second_variable: 'true' if the second variable should be True,
                            'actions' if the second variable should be a number representing a certain action (see dining philophers)
                            'variable' for any other variable
        
    '''
    
    # list_bool = [False]*len(nodes) # if it is already apperead
    # list_values = ['1']*len(nodes) 
    
    for index_node, node in enumerate(nodes): # Loop over the different nodes to modify all the strings 'predicate..'
        for predicate_level in range(len(grammar_predicate)):#Loop over grammar levels to find the number next to 'predicate' (e.g. 'predicate1')
            
          if f'predicate{predicate_level}' in node.data: #If the node is a predicate
                  
                # #If assigning same predicate to same name and the current predicate name has already been assigned to a concrete predicate
                # if  bool_same_number and list_bool[index_node] == True: node.data = list_values[index_node] #just copy a previous predicate
               
                # else: 
                    
                      list_predicates = os.listdir(f'{inputs.name_output_folder}/grammar/predicates/{m}_quant/predicate_level{predicate_level}')
                      list_predicates.sort()
                      sampled_pred = random.sample(list_predicates,1)[0]
                      with open(f'{inputs.name_output_folder}/grammar/predicates/{m}_quant/predicate_level{predicate_level}/{sampled_pred}', 'rb') as f: sampled_nodes = pickle.load(f)
                        
                      node.data = sampled_nodes[0].data
                      
                      # list_values[index_node] = node.data
                      # list_bool[index_node] = True
            
    return nodes


def sample_quantifiers(grammar_quantifiers, min_nb_quantifiers, max_nb_quantifiers):
    
    '''Standard: grammar_quantifiers = [ ['E psi0', 'A psi0', 'phi0' ]]
    
    Second output (word_quantifiers) named as quantifier_levels in syntax_guided.py represents the following:
        
    quantifier_levels = [#num_quantifiers , ['string quant 1 ', level_in_gramm  ], ['string quant 2', level_in_gramm] , [,]]
    e.g. [4, [' E', 0], [' A A ', 0], [' E ', 0], [' phi0', 0]]
    
    '''
    
    
    # For quantifiers: 
    # symbol 0 corresponds to EXISTS
    # symbol 1 corresponds to FOR ALL
    
    if min_nb_quantifiers is None: min_nb_quantifiers = 2
    if max_nb_quantifiers is None: max_nb_quantifiers = 10
    
        
    list_possible_strings = []
    current_formula = [ 0 ]
    level_grammar = 0
        
    list_possible_formulas = build_quantifiers(grammar_quantifiers, level_grammar, current_formula, list_possible_strings, min_nb_quantifiers, max_nb_quantifiers)
    
    if len(list_possible_formulas)==0: print('\nIt was not possible to find a string of quantifiers with admissible length!\n')
    
    word_quantifiers = random.sample(list_possible_formulas, k=1)[0]
    
    quantifiers = []
    #The loop starts from 1, because at index 0 there is only the number of quantifiers of the formula
    for i in range(1, len(word_quantifiers)): 
        for item in  word_quantifiers[i][0]: #contains the quantifiers
            if item == 'E': quantifiers.append(0)
            elif item == 'A': quantifiers.append(1)
    
    ##!!! Max number of spec_level is 9 (because of the use of -1 in the next line)   
    spec_level = int(word_quantifiers[-1][0][-1]) # the level in the spec grammar where the selected quantifiers point to
    
    # word_quantifiers[1:] is outputted because it will be used when changing the quantifiers; it indicates the level in the
    # quantifiers grammar for each quantifier
    
    #spec_level indicates the level in the grammar_spec that has to be called when using this quantifier 
    return quantifiers, word_quantifiers, spec_level

def build_quantifiers(grammar_quantifiers, level_grammar, current_formula, list_possible_formulas, min_nb_quantifiers, max_nb_quantifiers):
    
    '''Given a grammar, the function constructs all the strings of quantifiers that derive from the grammar 
    and whose length is within the given limits. 
    
    INPUTS:
        - grammar_quantifiers : the grammar that defines the quantifiers given as a list. 
            The upper level is defined by the the first element of the list. 
            If  Hyper-LTL is considered: grammar_quantifiers = [ ['E psi0', 'A psi0', 'phi0' ]]
            
        - level_grammar : the level of the grammar, indicated as index of the list grammar_quantifiers
        
        - current_formula: the current quantifiers in a list of pairs the form: [[quantifier, quantifier_level], [], []]
        
        - list_possible_formulas: list of admissible formulas find so far having the form of list of current_formulas
        
        - min_nb_quantifiers: minimum number of quantifiers 
        
        - max_nb_quantifiers: maximum number of quantifiers   
    '''
   
    #For all possible options in the current grammar level
    for index, item in enumerate( grammar_quantifiers[level_grammar]): 
        
        current_quantifiers = current_formula.copy()
        
        if 'phi' in item: #phi indicates the END of the recursion over the quantifiers
        
            #If the number of quantifiers is acceptable
            if (min_nb_quantifiers <= item.count('A')+ item.count('E') + current_quantifiers[0] <= max_nb_quantifiers): #ok to consider this string as acceptable
                # spec_level = int(item[(item.index('phi')+3)])#level of specification in grammar_structure that is required by these quantifiers
                current_quantifiers[0] += item.count('A')+ item.count('E')
                current_quantifiers.append([item, level_grammar])
                list_possible_formulas.append(current_quantifiers)
                # print(current_quantifiers)
                
        else:      
            
            if item.count('A')+ item.count('E') + current_quantifiers[0] <= max_nb_quantifiers: # try another iteration on the quantifier
                
                current_quantifiers[0] += item.count('A')+ item.count('E')
                spec_level = int(item[(item.index('psi')+3)])
                current_quantifiers.append([item[:-4], level_grammar]) # -4 because we keep only 'A' and 'E' from item (= eliminate psi* ) that corresponds to 4 characters
                list_possible_formulas = build_quantifiers(grammar_quantifiers,spec_level , 
                   current_quantifiers,
                  list_possible_formulas,min_nb_quantifiers, 
                   max_nb_quantifiers )
        
    return list_possible_formulas
                



def sample_new_quantifiers(formula, grammar_quantifiers):
    
    '''The function samples new quantifiers from the given ones, such that:
        
        - the final number of new_quantifiers is the same as the original ones;
        - the final new_quantifiers point at the same grammar level as the original quantifiers '''
    
    quantifiers = formula.quantifiers
    quantifiers_levels = formula.quantifier_levels
    
    list_possible_formulas = []
    current_formula = [0 ]
    
    selected_index = random.randrange(1, len(quantifiers_levels)) #selet an index from where to apply changes 
    level_grammar = quantifiers_levels[selected_index][1] # the second component stores the quantifier level of the current quantifier
    
    # Take into account cases in qwhich one elements contain multilple quantifiers [[ 'A',0] , ['EE',0]]
    length_new_quant = 0 #length of the new string of quantifiers
    
    for i in range(selected_index, len(quantifiers_levels)):
        length_new_quant +=  (quantifiers_levels[i][0].count('A') + quantifiers_levels[i][0].count('E'))
    
    list_possible_formulas = build_quantifiers(grammar_quantifiers, level_grammar, 
            current_formula, list_possible_formulas, length_new_quant, length_new_quant)
    
    
    while True: 
        word_quantifiers = random.sample(list_possible_formulas, k=1)[0]
        #If the new quantifiers point to the same spec level 
        if int(word_quantifiers[-1][0][-1]) == int(quantifiers_levels[-1][0][-1]): 
            break
            
    
    #Generation of the new quantifiers and new quantifier levels 
      
    new_quantifiers = quantifiers[:selected_index - 1]
    new_quantifiers_levels = quantifiers_levels[:selected_index] 
    
    
    for i in range(1, len(word_quantifiers)): 
        new_quantifiers_levels.append(word_quantifiers[i])
        for item in  word_quantifiers[i][0]: #contains the quantifiers
            if item == 'E': new_quantifiers.append(0)
            elif item == 'A': new_quantifiers.append(1)
         
    return new_quantifiers, new_quantifiers_levels

def count_depth_all_nodes(nodes):
    """
    Counts the depth of each node in the given list of nodes.

    Args:
        nodes (list): A list of nodes.

    Returns:
        list: A list containing the depth of each node.
    """
    list_depth = [None] * len(nodes)

    for ind, node in enumerate(nodes):
        list_depth[ind] = count_depth(nodes, ind)

    return list_depth

def count_depth(nodes, ind):
    '''The function counts the depth of the node at ind (i.e. of nodes[ind]) in the tree nodes'''
    max_depth = 0
    if nodes[ind].leftchild is not None:
        child_depth = count_depth(nodes, nodes.index(nodes[ind].leftchild))
        max_depth = max(max_depth, child_depth)
    if nodes[ind].rightchild is not None:
        child_depth = count_depth(nodes, nodes.index(nodes[ind].rightchild))
        max_depth = max(max_depth, child_depth)
    return max_depth + 1


def select_node(nodes, temperature):
    
    '''The function uniformly randomly selects a node in a formula.
    
        INPUT: 
            - nodes: formula expressed as a tree (list of nodes)
            - temperature (used in Simulated Annealing); if None, it is not used
            
        OUTPUTS:
            - index of the selected node (its subtree will be replaced)
    '''

    if len(nodes) == 1: selected_index = 0
    else: 
        # List of the depth of each node
        list_depth = count_depth_all_nodes(nodes)

        # If not using Simulated Annealing, just sample a subtree depth
        if temperature is None: subtree_depth = random.choice(list(range(1,6)))
        # When using simulated annealing, the subtree depth would (ideally) be higher when the temperature is higher and reduced to 1 when the temperature is close to zero
        else: 
            max_depth = max(list_depth)
            
            if temperature < 3000   : subtree_depth = 1

            elif temperature < 6000 : subtree_depth   = random.choice(list(range(1, max( 2, int((max_depth + 1) * 0.3 ))   ))) # 19 (->11/13 in practice)

            elif temperature < 8000 : subtree_depth  = random.choice(list(range(1, max( 2, int((max_depth + 1) * 0.5 ))   ))) # 15/16

            elif temperature < 10000 : subtree_depth = random.choice(list(range(1, max( 2, int((max_depth + 1) * 0.8))   ))) # 12/13

            else: subtree_depth = random.choice(list(range(1, max_depth + 1))) # 10
        
        list_of_indices = [ i for i, _ in enumerate(list_depth) if list_depth[i] == (subtree_depth)] 
        
        while len(list_of_indices)==0:  
            subtree_depth -= 1
            list_of_indices = [ i for i, _ in enumerate(list_depth) if list_depth[i] == (subtree_depth)] 
        
        selected_index = random.choice(list_of_indices)

    return selected_index

def find_subtree(nodes, selected_index):

    '''The function finds the indices of the nodes that are descendant of nodes[selected_index] (and hence the indices of the nodes that will be replaced).
      Moreover, the function returns the length of this chain of descendants of nodes[selected_index]'''     
       
    #old_node stores the indices of the nodes that will be replaced
    indices_old_node = [selected_index]

    #Initialize length of the chain of descendants of nodes[selected_index]
    length_node = 1
     
    to_be_examined = [selected_index]
    
    while True:
        if len(to_be_examined)==0: break
        
        if nodes[to_be_examined[0]].leftchild is not None:
            indices_old_node.append(nodes.index(nodes[to_be_examined[0]].leftchild))
            to_be_examined.append(nodes.index(nodes[to_be_examined[0]].leftchild))
            length_node +=1
        
        if nodes[to_be_examined[0]].rightchild is not None:
            indices_old_node.append(nodes.index(nodes[to_be_examined[0]].rightchild))
            to_be_examined.append(nodes.index(nodes[to_be_examined[0]].rightchild))
            length_node +=1
            
        to_be_examined.pop(0)
        
    return indices_old_node, length_node



def change_node(nodes, indices_old_node, new_node):
    
    '''The function replaces the nodes corresponding to indices_old_nodes in
    original_nodes with new_node'''
    
    right_child_bool = False
    
    is_root = True
    
    new_nodes = copy.deepcopy(nodes)
    
    ## REMOVE the bound from the selected node and its parent
    #if the node selected to be removed (together with all its descendants), it is not the root: 
    if new_nodes[indices_old_node[0]].parent is not None:
        
        # if the node randomly selected was a right child
        if new_nodes[indices_old_node[0]].parent.rightchild == new_nodes[indices_old_node[0]]:
            #remove the bound: the selected node is no more the right child of its parent
            new_nodes[indices_old_node[0]].parent.rightchild = None
            right_child_bool = True
            
        # if the node randomly selected was a left child
        elif new_nodes[indices_old_node[0]].parent.leftchild == new_nodes[indices_old_node[0]]:
            #remove the bound: the selected node is no more the left child of its parent
            new_nodes[indices_old_node[0]].parent.leftchild = None
            
        new_parent = new_nodes.index(new_nodes[indices_old_node[0]].parent)
        is_root = False
    
    # So far, indices_old_node[0] is where the subformula to be removed starts (and consequently where the new one should be put) 
        
    ## REMOVE all elements having index in indices_old_node from original_nodes
    new_indices_old_node = sorted(indices_old_node, reverse=True)
    for item in new_indices_old_node: new_nodes.pop(item)
    
    # Now, new_indices_old_node has been reversed, so consider new_indices_old_node[-1]
    
    ## ADD elements in new_node to original_nodes
    for i, item in enumerate(new_node): 
        if is_root :  new_nodes.append(item)
        else: new_nodes.insert(new_indices_old_node[-1] + i, item)
        
        if i == 0 and (is_root==False): 
            if right_child_bool: new_nodes[new_parent].rightchild = new_nodes[new_indices_old_node[-1]]
            else: new_nodes[new_parent].leftchild = new_nodes[new_indices_old_node[-1]]
            new_nodes[new_indices_old_node[-1]].parent = new_nodes[new_parent]
    
    return new_nodes

    