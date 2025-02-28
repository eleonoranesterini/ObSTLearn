import os
import pickle
import copy
from turtle import left


class Grammar:
    
    def __init__(self, quantifiers, structure, predicate = None):
        
        self.quantifiers = quantifiers
        self.structure   = structure
        self.predicate   = predicate

class Formula:
    
    def __init__(self, quantifiers, quantifier_levels, string_param, nodes, length ,mono = None, cost = None , string = None ):
        
        self.quantifiers = copy.deepcopy(quantifiers)
        self.quantifier_levels = quantifier_levels
        self.string_param = string_param
        self.nodes = copy.deepcopy(nodes) # STL nodes are always parametric
        self.length = length

        self.mono = mono # Monotonicity of the parameters
        self.cost = cost
        
        self.string = string # For STL, it is the version of string with the concrete values for parameters in the predicates

        self.variable_ranges = None # Just before storing the formula, we store the ranges of the variables in the formula because they appear normalized

        #Following line commented because for STL nodes are always parametric
        #self.nodes_param = None # For STL, it it the version of nodes with the parameters symbols in the predicates
        
        #The following attributes are used for storing once found a satisfied one:
        self.time = None #Time to learned the formula from the last satisfied one
        self.numb_mutation = None # Number of mutation before hitting a satisfied formula
        self.numb_hyper_monitor = None # Number of hyperformulas to be monitored before hitting a satisfied formula
        # In general, we expect number_hyper_monitor < numb_mutation because repeated formulas are monitored only once
        

class BinaryTreeNode:
    def __init__(self, data):
        
        self.data = data       # string indicating the operator or the predicate expressed by the node
        
        self.leftchild = None  # node that is the left child of the current node
        self.rightchild = None # node that is the right child of the current node
        self.parent = None     # node that is the parent of the current node
        
        #E.g. a until b with nodes = [node0, node1, node2] being node0.data = a, node1.data = until , node2,data = b
        # node1.leftchild  = node0
        # node1.rightchild = node2
        # node0.parent = node1
        # node2.parent = node1
        
        # Index of the nodes where the subformula(s) start: 
        # It is used in efficient monitoring in the case in whcih we want to store the outcome of the monitor on subformulas
        # The values are in the reference system of the current node: so the global position would be node_index + index_subformula1, node_index+ index_subformula2
        self.start_subformula = None # [index_subformula1, index_subformula2] ; negative values indicate the absence of subformula_i
        
        self.level_grammar = None # number reffering to the level of the node in the grammar definition (spec_level)
        
        # The node in position 0 (or in a different position but it was in position 0 when leading a subformul) 
        # in a formula contain the info [ number of 'predicates0', '#predicates1',.. ] in the whole formula
        self.predicates = None #Needed for enumeration of possible formulas # only used in enumerate_formulas
        
        # The node indicates where to find the corresponding formula in terms of 'spec_level', 'length', and 'number_node'
        # self.location = None # [spec_level , length, number_node ]
        
        self.polarity = None # Used to check monotonicity of parameters. The rest of the time is set to None. 
        
def parse_gramm(grammar_sec):
    
    '''The function transforms the grammar (either of the quantifiers or of the quantifier-free formula structure)
    given as a string with different levels in the format needed by the algorithm.'''
    
    #Different levels divided by the new line
    levels = grammar_sec.split('\n')
    grammar_structure = []
    
    for level in levels:
        line = []
        
        if '='  in level: #Remove first and last line because they are empty
            
            right = level.split('==')
            if '|' not in right[1]: #right[1] is the right  part of the equation
                line = [right[1]]
            else: 
                item = right[1].split('|')
                for i in range(len(item)):
                    if ' |' in item[i]: item[i] = item[i].replace(' |', '')
                    if '|' in item[i]: item[i] = item[i].replace('|', '')
                    if 'forall' in item[i]: item[i] = item[i].replace('forall', 'A')
                    if 'exists' in item[i]: item[i] = item[i].replace('exists', 'E')
                    line.append(item[i])
            grammar_structure.append(line)
    return grammar_structure 


def print_tree(tree_formula):
    for i, node in enumerate(tree_formula):
        print(f'\n\n\nindex={i}, node={node.data}')
        if node.rightchild is not None: print(f'rightchild={tree_formula.index(node.rightchild)}')
        if node.leftchild is not None: print(f'leftchild={tree_formula.index(node.leftchild)}')
        if node.parent is not None: 
            print(f'parent={tree_formula.index(node.parent)}')
       
            
def tree_to_formula(nodes):
    
    ## HYPOTHESIS:
    ## nodes[0] = root
    
    start = '' 
    formula_string = f'{translate_node(start, nodes,  0 )}'
    
    return formula_string


    
def translate_node(formula, nodes, i):
    
    if  'not' in nodes[i].data or  'next' in nodes[i].data or 's_next' in nodes[i].data or\
         'prev' in nodes[i].data or 's_prev' in nodes[i].data or  'historically' in nodes[i].data \
          or 'once' in nodes[i].data or 'eventually' in nodes[i].data or 'always'  in nodes[i].data :
        return  f'{formula}{nodes[i].data}({translate_node(formula, nodes, nodes.index(nodes[i].leftchild))} )'
   
    elif  'until' in nodes[i].data   \
            or 'since' in nodes[i].data or '<->' in nodes[i].data or 'implies' in nodes[i].data \
            or 'or' in nodes[i].data or 'and' in nodes[i].data :
        return f'{formula}( {translate_node(formula,nodes, nodes.index(nodes[i].leftchild))}){nodes[i].data}({translate_node(formula,nodes, nodes.index(nodes[i].rightchild))} )'
    
    elif 'weak_u' in nodes[i].data:
        return f'{formula}( ({translate_node(formula,nodes, nodes.index(nodes[i].leftchild))}){nodes[i].data}({translate_node(formula,nodes, nodes.index(nodes[i].rightchild))}) or always({translate_node(formula,nodes, nodes.index(nodes[i].leftchild))}))'

    else: #PREDICATE
        return nodes[i].data
    

def enumerates_equivalent_formula_strings(nodes): 
    
    '''Heuristics to enumerate  formulas that are equivalent to the given one by swapping the order of the children of the commutative binary operators.
    There is always a tree structure: so (A and B) and C will swap A with B , (A and B) with C, but never A with C (or B with C).
    So the enumeration is clearly only partial!
    '''

    ## HYPOTHESIS:
    ## nodes[0] = root

    start = ''
    all_formulas = []
    
    # Start the recursion with the root node
    generate_formulas(start, nodes, 0, all_formulas)
    # for formula in all_formulas:
    #     print('eq formula:', formula)

    return all_formulas


def generate_formulas(formula, nodes, i, all_formulas):
    """
    Recursively builds formulas and appends all valid permutations
    by swapping the children of binary operators.
    """
    
    # If the node is a unary operator
    if  'not' in nodes[i].data or 'next' in nodes[i].data or 's_next' in nodes[i].data or\
         'prev' in nodes[i].data or 's_prev' in nodes[i].data or 'historically' in nodes[i].data or\
         'once' in nodes[i].data or 'eventually' in nodes[i].data or 'always' in nodes[i].data:
        
        generated_formulas = generate_formulas(formula, nodes, nodes.index(nodes[i].leftchild), all_formulas)
        all_subformulas = []
        for el in generated_formulas:
            # Recursively build formula with unary operator
            subformula = f'{formula}{nodes[i].data}({el})'
            all_subformulas.append(subformula)
            if i == 0: all_formulas.append(subformula)

        return all_subformulas
    
    # If the node is a binary operator COMMUTATIVE
    elif   '<->' in nodes[i].data or 'or' in nodes[i].data or 'and' in nodes[i].data:
        all_subformulas = []
        # Get left and right children formulas
        left_formulas = generate_formulas(formula, nodes, nodes.index(nodes[i].leftchild), all_formulas)
        right_formulas = generate_formulas(formula, nodes, nodes.index(nodes[i].rightchild), all_formulas)
        
        for left_formula in left_formulas:
            for right_formula in right_formulas:
                # Add original order (left-right)
                original_formula = f'({left_formula}) {nodes[i].data} ({right_formula})'
                if i ==0: all_formulas.append(original_formula)
                all_subformulas.append(original_formula)
                
                # Add swapped order (right-left)
                swapped_formula = f'({right_formula}) {nodes[i].data} ({left_formula})'
                if i ==0: all_formulas.append(swapped_formula)
                all_subformulas.append(swapped_formula)
        
        return all_subformulas 
    
    elif 'until' in nodes[i].data or 'since' in nodes[i].data or 'implies' in nodes[i].data:
        all_subformulas = []

        # Get left and right children formulas
        left_formulas = generate_formulas(formula, nodes, nodes.index(nodes[i].leftchild), all_formulas)
        right_formulas = generate_formulas(formula, nodes, nodes.index(nodes[i].rightchild), all_formulas)
        
        for left_formula in left_formulas:
            for right_formula in right_formulas:
                original_formula = f'({left_formula}) {nodes[i].data} ({right_formula})'
                if i ==0: all_formulas.append(original_formula)
                all_subformulas.append(original_formula)
        
        return all_subformulas  # Return the original formula for recursion
    
    # If the node is 'weak_u'
    elif 'weak_u' in nodes[i].data:
        all_subformulas = []
        # Get left and right children formulas
        left_formulas = generate_formulas(formula, nodes, nodes.index(nodes[i].leftchild), all_formulas)
        right_formulas = generate_formulas(formula, nodes, nodes.index(nodes[i].rightchild), all_formulas)
        for left_formula in left_formulas:
            for right_formula in right_formulas:
                # Original weak_u formula
                weak_u_formula = f'(({left_formula}) {nodes[i].data} ({right_formula}) or always({left_formula}))'
                if i == 0: all_formulas.append(weak_u_formula)
                all_subformulas.append(weak_u_formula)
        
        return all_subformulas

    else:  # If it's a predicate (leaf node)
        return [nodes[i].data]





def instantiate_subformula_same_predicate(nodes, subformula_nodes, start_index):
    
    ''' subformula_nodes is the tree structure of the subformula in nodes starting at index start_index'''
    
    for i in range(len(subformula_nodes)):
        
        if 'predicate' in subformula_nodes[i].data:
            subformula_nodes[i].data = nodes[ i + start_index].data
            
    return subformula_nodes
        
    
    
def find_traces_subformulas(formula, ind):

    ''' ind = tuple of traces that have to replace trace variable pi[i] in the evaluation '''   
    
    indices_start =  [i for i in range(len(formula)) if formula.startswith('pi[', i)]
                     
    indices_end = [i for i in range(len(formula)) if formula.startswith('][', i)]
    
    trace_variables_present = [int( formula[indices_start[i] + 3 : indices_end[i]] )  for i in range(len(indices_start))]
    
    new_ind = tuple()
    
    for i, el in enumerate(ind):
        if i in trace_variables_present: new_ind = new_ind + (el,)
        else: new_ind = new_ind + (-1,)
        
    return new_ind



def check_if_in_gramm(nodes, grammar, inputs):
    
    '''The function checks whether the nodes can be generated by the given grammar.'''
    
    # Let's forget about quantifiers for the moment.
    #       - The quantifiers need to be checked on their own 
    #       - Define the admissible_levels for the first element in the recursion
    
    # Start with the root node
    # Need to check for each node:
        # - if it is an operator, that the operator exists at that level
        #                         that its children have the level admitted by the grammar
        # - if they are predicate: they are in the list of predicates corresponding to their level
    
    # Find the index of the root node
    for index, node in enumerate(nodes):
        if node.parent is None:
            start_index = index
            break
        
    return check_if_in_grammar_recursion(start_index, [0], nodes, grammar,inputs)

     

def check_if_in_grammar_recursion(start_subformula, admissible_levels, nodes, grammar, inputs):
    
    '''Consider the node nodes[start_subformula]. 
       Check if its level belongs to the list of admissible_levels in the grammar'''
    
    numb_quant = 1 # Number of quantifiers in the formula

    current_node = nodes[start_subformula]
    
    # Check if the node is in at an admissible grammar level  
    if current_node.level_grammar not in admissible_levels: return False
    
    # If the node is a unary operator
    if any(un_op in current_node.data for un_op in ['not', 'always', 'eventually' , 'historically', 'once', 's_next' , 'next', 's_prev' , 'prev']):
        #Vector of booleans indicating whether current node is in one of strings in the indicated grammar level
        bool_inclusion = [ current_node.data.replace(" ",'') in item for item in grammar.structure[current_node.level_grammar]]
        #Check if the node is in the corresponding grammar level
        if not any(bool_inclusion): return False
        # Else 
        future_admissible = []
        # At least one string in the grammar level contains the current node 
        for index, bool_value in enumerate(bool_inclusion):
            if bool_value: # Add the spec level of the child indicated by the grammar 
                string_in_grammar = grammar.structure[current_node.level_grammar][index] #string in the grammar corresponding to the current node 
                future_admissible.append(int(string_in_grammar[string_in_grammar.find('phi') + 3 ]))
    
        return check_if_in_grammar_recursion(nodes.index(current_node.leftchild) , future_admissible , nodes, grammar ,inputs)
        
        
    # If the node is a binary operator 
    elif any(bin_op in current_node.data for bin_op in ['<->' , 'implies', 'until', 'since', 'weak_u', 'or', 'and']):
        #Vector of booleans indicating whether current node is in one of strings in the indicated grammar level
        bool_inclusion = [ current_node.data.replace(" ",'') in item for item in grammar.structure[current_node.level_grammar]]
        #Check if the node is in the corresponding grammar level
        
        if not any(bool_inclusion): return False
        #Else
        future_admissible_left = []
        future_admissible_right = []
        # At least one string in the grammar level contains the current node 
        for index, bool_value in enumerate(bool_inclusion):
            if bool_value: 
                # Add the spec level of the child indicated by the grammar  : first encountered is left
                string_in_grammar = grammar.structure[current_node.level_grammar][index] #string in the grammar corresponding to the current node
                future_admissible_left.append(int(string_in_grammar[string_in_grammar.find('phi') + 3 ]))
                future_admissible_right.append(int(string_in_grammar[string_in_grammar.find('phi', 4) + 3 ]))
    
        bool_left = check_if_in_grammar_recursion(nodes.index(current_node.leftchild), future_admissible_left, nodes, grammar, inputs)
        if not bool_left: return False
        #Keep only the right admissible corresponding to the actual level of the left child
        new_future_admissible_right = []
        for i in range(len(future_admissible_right)):
           if current_node.leftchild.level_grammar == future_admissible_left[i]: new_future_admissible_right.append(future_admissible_right[i])
       
        bool_right = check_if_in_grammar_recursion(nodes.index(current_node.rightchild) , new_future_admissible_right, nodes, grammar, inputs)
        
        if not bool_right: return False
        else: return True
        
        
    # Else: the node is a predicate
    else: 
        predicate_levels = [] # List of predicate levels in the grammar matching with the current node
        for index_level, item_level in enumerate(grammar.predicate):
            
            # Need to check for STL predicates that in current_node.data may be instantiated, and not in item_level
            if any(['epsilon' in el for el in item_level]):
                
                #Replace numbers not in square brackets in current_node.data
                without_numbers, indices_removed = replace_numbers(current_node.data)
                
                #Replace 'epsilon{i}-' in el in item_level and check whether there is matching with without_numbers

                for el in item_level: 
                    new_item , indices_epsilon = replace_epsilon_string(el)
                    #If there is a matching without numbers and epsilon
                    if without_numbers in new_item and check_matching_numerical_values(current_node.data, indices_removed, el , indices_epsilon, inputs.par_bounds):
                        predicate_levels.append(index_level)
                        break
            # No STL or STL but with no numerical values to be mined (fixed predicates)
            # If index_level already added to predicate_levels, no need to check again
            if len(predicate_levels) > 0 and predicate_levels[-1] == index_level: continue
            else:
                # path to the enumeration of the predicates in the current grammar level
                path = f'{inputs.name_output_folder}/grammar/predicates/{numb_quant}_quant/predicate_level{index_level}/'
                # Load the list of predicates in the current grammar level
                if not os.path.exists(f'{path}'):
                    print(f'{path}\nPath for predicates not found') 
                    continue
                for index_load in range(len(os.listdir(path))):
                    with open(f'{path}/nodes_{index_load}.obj', 'rb') as f: el = pickle.load(f)
                    if current_node.data in el[0].data: 
                        predicate_levels.append(index_level)
                        break
    
        # Check if the string predicate{i} is in the corresponding grammar level
        if any([f'predicate{index}' in el_gramm for el_gramm in grammar.structure[current_node.level_grammar] for index in predicate_levels] ): return True
        else: return False


def extract_subnodes(nodes , selected_index):

    '''Extract the subnodes of the formula at the given index'''

    to_be_examined = [selected_index]
    to_be_examined_subformula = [0]
    
    subnodes = [BinaryTreeNode(nodes[selected_index].data)]

    while True:

        if len(to_be_examined)==0: break

        subnodes[to_be_examined_subformula[0]].level_grammar = nodes[to_be_examined[0]].level_grammar

        if nodes[to_be_examined[0]].leftchild is not None:
            to_be_examined.append(nodes.index(nodes[to_be_examined[0]].leftchild))
            subnodes.append(BinaryTreeNode(nodes[to_be_examined[0]].leftchild.data))
            to_be_examined_subformula.append(len(subnodes) - 1)

            subnodes[to_be_examined_subformula[0]].leftchild = subnodes[-1]
            subnodes[-1].parent = subnodes[to_be_examined_subformula[0]]
        
            subnodes[to_be_examined_subformula[0]].start_subformula = [ len(subnodes) - 1 - to_be_examined_subformula[0]]
        else:
            subnodes[to_be_examined_subformula[0]].start_subformula = [-1]

        if nodes[to_be_examined[0]].rightchild is not None:
            to_be_examined.append(nodes.index(nodes[to_be_examined[0]].rightchild))
            subnodes.append(BinaryTreeNode(nodes[to_be_examined[0]].rightchild.data))
            to_be_examined_subformula.append(len(subnodes) - 1)

            subnodes[to_be_examined_subformula[0]].rightchild = subnodes[-1]
            subnodes[-1].parent = subnodes[to_be_examined_subformula[0]]
            
            subnodes[to_be_examined_subformula[0]].start_subformula.append(len(subnodes) - 1 - to_be_examined_subformula[0])
        else:
            subnodes[to_be_examined_subformula[0]].start_subformula.append(-1)
            
        to_be_examined.pop(0)
        to_be_examined_subformula.pop(0)

        
    return subnodes


def replace_numbers(data):
    
    ''' Data is a string that possibly contains numbers.'''

    predicate_operations = ['<', '>', '=', '!']

    indices = [] # Indices of data to be removed
    inside_brackets = False
    # Eliminate numbers except inside square brackets (because they are variables numbers)
    for index, char in enumerate(data):
        if char == '[': inside_brackets = True
        elif char == ']': inside_brackets = False
        # Allow also dot for float numbers  and the exponential representation e-{number}
        elif not inside_brackets and (char.isdigit() or char == '.' \
                                      or (char == 'e' and len(data) > index+2 and data[index+1] == '-' and data[index+2].isdigit()) \
                                      or (char == '-' and len(data) > index+1 and data[index+1].isdigit()) ): 
            indices.append(index)
            continue  # Skip numbers outside square brackets (i.e. variables index in the predicates)
        
    # Group consecutive numbers in the variable indices    
    indices_removed = []
    start = None
    
    for i in range(len(indices)):
        if i == 0 or indices[i] != indices[i-1] + 1:
            if start is not None:
                indices_removed.append([start, indices[i-1]])
            start = indices[i]
    if start is not None:
        indices_removed.append([start, indices[-1]])

    # Do not eliminate costant number values in the predicate
    # eg. If there are numbers in the predicate (ex. (pi[0][1] - 3) < 5) -> 5 has to be replaced but 3 not
    refined_indices = []
    for i_indices, el in enumerate(indices_removed):
        # If the two elements before and after the number are not predicate operations -> Remove it
        if ( (el[0]-1 < 0) or data[el[0]-1] not in predicate_operations) and \
         ((el[0]-2 < 0) or data[el[0]-2] not in predicate_operations) and \
         (el[1] + 1 >= len(data) or data[el[1]+1] not in predicate_operations) and \
         (el[1] + 2 >= len(data) or data[el[1]+2] not in predicate_operations):
            # Remove i-th element from indices_removed
            refined_indices.append(i_indices)
    
    indices_removed = [ el for i, el in enumerate(indices_removed) if i not in refined_indices]
    
    # ELiminate from data the elements in refined_indices_removed
    new_data = ''
    for index, char in enumerate(data):
        bool_to_be_added = True # True that we can add char to new data
        for el in indices_removed:
            if el[0] <= index <= el[1]: 
                bool_to_be_added = False # False because char's index is in the list of indices to be removed
                break
        if bool_to_be_added: new_data += char

    # Eliminate black spaces
    return new_data.replace(" ", ""), indices_removed

def replace_epsilon_string(data):
    
    ''' data is a string that possibly contains substrings 'epsilon{i}-'
    The function returns a string without all such substrings and the strart-end indices of them in data'''
    
    indices_epsilon = []
    
    # Collect indices of all epsilon{}- : where the string starts and where it ends
    index = 0 
    while (data[index:].find('epsilon') >= 0):
        start_index = data[index:].find('epsilon') + index
        end_index = data[start_index:].find('-') + start_index
        indices_epsilon.append([start_index, end_index])
        index = end_index
     # If no epsilons were in data   
    if len(indices_epsilon) == 0: return data.replace(" ", ""), indices_epsilon
    
    # Create new string without all the espilon{}- substrings
    index = 0 
    while (index < len(indices_epsilon)):
        #Initial substring
        if index == 0: new_data = data[:indices_epsilon[index][0]]
        # Substring between two epsilon
        else: new_data = new_data + data[indices_epsilon[index -1 ][1]+1 : indices_epsilon[index][0]]
        index += 1
    # Add final part
    new_data = new_data + data[indices_epsilon[-1][1] + 1 : ] 
    new_data = new_data.replace(" ", "")# Eliminate black spaces 
    return new_data, indices_epsilon

def check_matching_numerical_values(data, indices_removed, item_grammar, indices_epsilon, par_bounds):

    '''The function checks whether the numerical values in data are among those allowed 
     by inputs.par_bounds for the epsilon{i}- in item_grammar'''
    
    for index in range(len(indices_removed)):
        #Value of i for epsilon{i} in the index-th element of item_grammar
        epsilon_number = int(item_grammar[indices_epsilon[index][0]+  7 : indices_epsilon[index][1] ])
        if par_bounds[epsilon_number][0] <= float(data[indices_removed[index][0]: indices_removed[index][1]+1]) <= par_bounds[epsilon_number][1]:
            continue
        else: return False
    
    return True


def insert_leftchild_unary_node_in_list_nodes(nodes, new_node_data, new_node_index):
    
    '''The function insert a new node in an existing list of nodes by updating 
    relation of childhood and parenthood with the existing list of node.
    
    !! New_node copies the level of grammar from its parent
    
    Unary node because the start_subformula of the new node contains only leftchild and it is also a LEFTCHILD
    
    INPUTS: 
        - nodes (list of nodes)
        - new_node_data 
        - new_node_index: index that new_node_data has to assume in the list new_nodes
    
    '''
    
    new_nodes = copy.deepcopy(nodes)
    
    #Select node in the list that will be parent of the newly added node
    node_parent_to_be = new_nodes[new_node_index - 1]
    
    #Update the attribute .start_subformula of the existing nodes
    
    for i in range(new_node_index): #For nodes with indices coming before the index where the new_node will be added
        
        # If it has a leftchild that will be after the newly added node
        # (Not in case of the parent_to_be node, because the new node will be the parent of its leftchild)
        if  i!= new_node_index -1 and \
            new_nodes[i].leftchild is not None and new_nodes.index(new_nodes[i].leftchild) >= new_node_index : new_nodes[i].start_subformula[0] += 1
        # If it has a rightchild that will be after the newly added node
        if new_nodes[i].rightchild is not None and new_nodes.index(new_nodes[i].rightchild)>= new_node_index : new_nodes[i].start_subformula[1] += 1
        
    
    #Insert the new node
    new_nodes.insert(new_node_index, BinaryTreeNode(new_node_data))
    
    #Update relations of the original leftchild of the current node
    node_parent_to_be.leftchild.parent = new_nodes[new_node_index]
    
    #Impose relations of the newly added node
    new_nodes[new_node_index].leftchild =  node_parent_to_be.leftchild
    new_nodes[new_node_index].parent =  node_parent_to_be
    new_nodes[new_node_index].start_subformula = [ node_parent_to_be.start_subformula[0] , -1]
    new_nodes[new_node_index].level_grammar =  node_parent_to_be.level_grammar
    
    #Update relations index_node
    node_parent_to_be.leftchild = new_nodes[new_node_index]
    node_parent_to_be.start_subformula = [1, node_parent_to_be.start_subformula[1]] # The new node is its leftchild and it is added immediatly after him
    
    # node.level_grammar = node.level_grammar #Remains unchanged

    return new_nodes




# #Definition of the grammar for the quantifiers
# grammar_quantifiers = '''
# psi0 == forall forall psi0 | phi0
# '''
# #Definition of the grammar for the formula structure        
# grammar_structure = '''
# phi0 == always phi1
# phi1 == phi2 implies phi2
# phi2 == eventually phi2 | always phi2 | phi2 until phi2 | phi2 implies phi2 | phi2 and phi2 | phi2 or phi2 | not phi2 | predicate0
# '''
# #Definition of the grammar for predicates
# grammar_predicate = [['( abs( pi[0][0] - pi[1][0] ) < epsilon0- )', '( abs( pi[0][1] - pi[1][1] ) < epsilon1- )']]


# grammar = Grammar()
# grammar.quantifiers = parse_gramm(grammar_quantifiers)
# grammar.structure = parse_gramm(grammar_structure)
# grammar.predicate = grammar_predicate 



# nodes = [gramm.BinaryTreeNode('or'), gramm.BinaryTreeNode('eventually'), gramm.BinaryTreeNode('implies'), gramm.BinaryTreeNode('predicate1'), gramm.BinaryTreeNode('predicate2'), gramm.BinaryTreeNode('always'),gramm.BinaryTreeNode('and'),gramm.BinaryTreeNode('predicate3'), gramm.BinaryTreeNode('predicate4')]
# nodes = [BinaryTreeNode('or'),BinaryTreeNode('eventually'), BinaryTreeNode('implies'), BinaryTreeNode('predicate1'), BinaryTreeNode('predicate2'),BinaryTreeNode('always'),BinaryTreeNode('and'),BinaryTreeNode('predicate3'), BinaryTreeNode('predicate4')]

# nodes[0].leftchild = nodes[1]
# nodes[0].rightchild = nodes[5]
# nodes[1].leftchild = nodes[2]
# nodes[2].leftchild = nodes[3]
# nodes[2].rightchild = nodes[4]
# nodes[5].leftchild = nodes[6]
# nodes[6].leftchild = nodes[7]
# nodes[6].rightchild = nodes[8]
# nodes[1].parent = nodes[0]
# nodes[2].parent = nodes[1]
# nodes[3].parent = nodes[2]
# nodes[4].parent = nodes[2]
# nodes[5].parent = nodes[0]
# nodes[6].parent = nodes[5]
# nodes[7].parent = nodes[6]
# nodes[8].parent = nodes[6]
# for node in nodes: node.level_grammar = 0
# nodes[0].start_subformula = [1, 5]
# nodes[1].start_subformula = [1, -1]
# nodes[2].start_subformula = [1, 2]
# nodes[3].start_subformula = [-1, -1]
# nodes[4].start_subformula = [-1, -1]
# nodes[5].start_subformula = [1, -1]
# nodes[6].start_subformula = [1, 2]
# nodes[7].start_subformula = [-1, -1]
# nodes[8].start_subformula = [-1, -1]
# print(tree_to_formula(nodes))

# # new_nodes = insert_leftchild_unary_node_in_list_nodes(nodes, 'once', 1)
# print(check_if_in_gramm(nodes, grammar))


def obtain_numerical_value_substring_from_original_string(subformula_string_param,formula_string):
        
        # Obtain the subformula string without epsilon string
        substring_no_numbers , indices_epsilon_removed = replace_epsilon_string(subformula_string_param)

        # If the subformula has no numerical values, skip the evaluation
        if len(indices_epsilon_removed) == 0: 
            print('No numerical values in the subformula - refinement.py')
            return None
        
        # Obtain the original formula string without numerical values
        string_no_numbers, indices_number_removed = replace_numbers(formula_string)
        # Compute where the substring starts in the string without numbers
        occurrences = [i for i in range(len(string_no_numbers)) if string_no_numbers.startswith(substring_no_numbers, i)]
        if len(occurrences) == 0: 
            print('Subformula not found in the formula')
            return None
        
        if len(occurrences) > 1: print('Subformula found more than once in the formula - refinement.py. Formula:', string_no_numbers, 'substring_no_numbers',substring_no_numbers)
        index_occurrence = occurrences[0] # Take the first occurrence

        # Slide indices in both strings to find which intervals of values in indices_number_removed correspond to the subformula
        index_original = 0
        index_without_numbers = 0
        interval_skipped = 0
        while index_original < len(formula_string):
            if index_without_numbers == index_occurrence: break
            # If the characters are the same -> next step
            if string_no_numbers[index_without_numbers] == formula_string[index_original]:
                index_without_numbers += 1
                index_original += 1
            elif formula_string[index_original] == ' ': index_original += 1
            else:
                index_original = indices_number_removed[interval_skipped][1] + 1
                interval_skipped += 1

        aux = subformula_string_param
        index_subformula = 0   
        # Replace the epsilon values with the numerical values in the original formula
        while 'epsilon' in aux:
            start_epsilon = aux.find('epsilon') + 7
            end_epsilon = aux.find('-', start_epsilon)
            index_epsilon = int(aux[start_epsilon : end_epsilon])
            aux = aux.replace(f'epsilon{index_epsilon}-', formula_string[\
                indices_number_removed[index_subformula + interval_skipped][0] : indices_number_removed[index_subformula + interval_skipped][1] + 1])
            index_subformula += 1

        return aux