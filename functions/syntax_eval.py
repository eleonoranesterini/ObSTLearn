import sys
# sys.path.insert(0, 'rtamt/')

import itertools
import numpy as np
import math

import rtamt
from rtamt.spec.stl.discrete_time.specification import Semantics

import functions.create_grammar as gramm
import functions.metrics as metrics

# # Functions:
#     - evaluate_rob_boolean
#     - evaluate_rob_quantitative
#     - compute_fitness_positive (call evaluate_rob_boolean)
#     - compute_fitness_negative (call evaluate_rob_boolean)
#     - compute_fitness (call compute_fitness_positive and compute_fitness_negative)
#     - efficient_monitoring (call evaluate_rob_boolean)
#     - compute_robustness_STL (call evaluate_rob_quantitative)
#     - lookup_table_uniform_distance
#     - monitor_hyperSTL_correctness (call lookup_table_uniform_distance and evaluate_rob_quantitative )

class Inputs_Monitoring:
    
    def __init__(self ):
        
        self.set_violated_formulas = set()
        self.set_satisfied_formulas = set()
        self.print_counter = None

        
def print_formula(quantifiers, formula):
    
    m = len(quantifiers)
    
    f = ''
    
    for i in range(m):
        if quantifiers[i]==0: f += f'E pi[{i}] '
        elif quantifiers[i]==1: f += f'FA pi[{i}] '
    
    f = f + ': ' + formula
    return f

def round_timing_bound(formula):

    '''Change the formula to address the RTAMT error:
     RTAMT Exception: The operator bound must be a multiple of the sampling period  '''

    new_formula = formula
    sub_formula = new_formula   
    characters_removed = 0  #Counts the number of characters removed

    while True:
        # sub_formula is a fraction of new_formula, at each iteration with a pair of [] less
        start = sub_formula.find('[')

        #Stop when there are no more temporal parameters
        if start < 0 : return new_formula 

        end = sub_formula.find(']')
        bounds = sub_formula[start+1 : end].split(':')
        
        # Round the first temporal parameter
        sep = sub_formula.find(':')
        new_formula = new_formula[:characters_removed + start + 1]+ f'{int(round(float(bounds[0])))}' + new_formula[characters_removed + sep:]
        sub_formula_1 = sub_formula[: start + 1]+ f'{int(round(float(bounds[0])))}' + sub_formula[sep:]
         
        #Round the second temporal parameter   
        sep = sub_formula_1.find(':') #updated cause the formula has changed
        end = sub_formula_1.find(']') #updated cause the formula has changed
        new_formula = new_formula[:characters_removed + sep + 1]+ f'{int(round(float(bounds[1])))}' + new_formula[characters_removed + end:]
        sub_formula_2 = sub_formula_1[: sep + 1]+ f'{int(round(float(bounds[1])))}' + sub_formula_1[end:]
        end = sub_formula_2.find(']')

        #Count charatcters removed from sub_formula_2 to the end of the formula (by eliminating the part of the formula till the first ']')
        characters_removed += len(sub_formula_2[:end+1])
        # Sub_formuala for the next iteration: the remaining part of the formula after the first ']'
        sub_formula = sub_formula_2[end+1:]


def evaluate_rob_boolean(pi, formula, bool_interface_aware = False, input_variables = None , output_variables = None):
    
    '''
    For LTL in which we only want satisfaction or violation
    
    Computes the boolean satisfaction/violation of a (quantifier-free) formula 
    with respect to a specific tuple of traces. 
    It uses the RTAMT tool with boolean satisfaction: outputs are only +inf or -inf
    
    -bool_interface_aware: boolean, whether to use or not the prior knowledge of input-output variables
    -input_variables: None if bool_interface_aware is False, else the list of input variables
    -output_variables: None if bool_interface_aware is False, else the list of output variables
    
    '''
    # Transform pi[i][j] into pi_i_j because the tool does not support the brackets
    
    new_formula = formula.replace('][','_')
    new_formula = new_formula.replace('pi[','pi_')

    # If the formula does not have the value of end_trace specified (eg if traces in the dataset have different lengths),
    # replace it with the actual length of the trace
    new_formula = new_formula.replace('end_trace', str(len(pi[0][0])) )
    
    # Indices of where the 'pi' start
    variable_indices = [i for i in range(len(new_formula)) if new_formula.startswith('pi', i)]
    
    
    spec = rtamt.STLSpecification(language=rtamt.Language.PYTHON,semantics=rtamt.Semantics.OUTPUT_ROBUSTNESS)
    # spec.semantics = Semantics.STANDARD
    
    spec.name = 'STL Discrete-time Offline monitor'
    
    dataSet = { 'time': list(np.arange(len(pi[0][0])))}
    
    already_declared = []
    
    for item in  variable_indices:
        indx1 = new_formula[item + 3]# +3 because it is the length of the string 'pi_'
        end = 4 # the index has at least length 1 ( = 4-3 )
        while True : # find the end of first index 
            if new_formula[item + end] == '_': break #the first index ends when '_' appears
            end += 1
        indx1 = int(new_formula[item + 3: item + end]  )
        
        start = item + end + 1 #from the position of '_' + 1 (start of second index)
        end = start + 1 
        while True: # find the end of second index 
            if new_formula[end] == ']' or new_formula[end] == ')': break #the second index ends when ' ' or ')' appear
            end += 1
        
        indx2 = int(new_formula[start : end])
        new_formula = new_formula[:end]+ ' ' + new_formula[end +1:]
        #Avoid already declared pairs of indices
        if f'{indx1}{indx2}' not in already_declared:
        
            spec.declare_var(f'pi_{indx1}_{indx2}', 'float')
            
            if bool_interface_aware == False: spec.set_var_io_type(f'pi_{indx1}_{indx2}', 'input')
            else:
               
                if indx2 in input_variables :    spec.set_var_io_type(f'pi_{indx1}_{indx2}', 'input')
                elif indx2 in output_variables:  spec.set_var_io_type(f'pi_{indx1}_{indx2}', 'output')
                else:
                    print(f'\nVariable {indx2} not specified either as input or output variable!\n')
                    sys.exit()
                    
            dataSet.update({f'pi_{indx1}_{indx2}' : pi[int(indx1)][int(indx2)]})
            
            already_declared.append(f'{indx1}{indx2}')
    
    if 'True' in new_formula:
        spec.declare_const('True', 'int', 1)
        spec.set_var_io_type('True', 'input')
        
    if 'False' in new_formula:
        spec.declare_const('False', 'int', 0)
        spec.set_var_io_type('False', 'input')

    if 'UNKNOWN' in new_formula:
        spec.declare_const('UNKNOWN', 'float', 99999.99)

    new_formula = round_timing_bound(new_formula)    
                
    spec.spec = new_formula
    
    try:
        spec.parse()
        
    except rtamt.STLParseException as err:
        print(new_formula)
        print('STL Parse Exception: {}'.format(err))
        sys.exit()
        
    rho = spec.evaluate(dataSet)
    
   
    # JUST CHECK SATISFACTION VS VIOLATION
    if bool_interface_aware == False:
        
        # if rho[0][1] == 0: print('\n\n\n\n\n\n\n\n\n\n\n\n\nROB = 0 !!!!\n\n\\n\n\n\n\n\n\n\n\n\n\n')
    
        if rho[0][1] >= 0: return 'sat', [rho[i][1] for i in range(len(rho)) ] 
        
        else: return 'unsat' , [rho[i][1] for i in range(len(rho)) ] 
        
    # #CHECK NON - VACOUSLY SATISFACTION    
    elif bool_interface_aware == True:
        if rho[0][1] == np.inf: 
    #         # print('skipped')
            return 'skipped' , [rho[i][1] for i in range(len(rho)) ] 
    
        elif rho[0][1] > 0:
    #         # print('sat')
            return 'sat' , [rho[i][1] for i in range(len(rho)) ] 
        
        elif rho[0][1]< 0: return 'unsat', [rho[i][1] for i in range(len(rho)) ] 
        elif rho[0][1] == 0: 
            # print('che fare?')
            return 'unsat', [rho[i][1] for i in range(len(rho)) ] 
           



def evaluate_rob_quantitative(pi, formula):
    
    
    '''Computes the QUANTITATIVE satisfaction/violation of a (quantifier-free) formula 
    with respect to a specific tuple of traces. 
    It uses the RTAMT tool with quantitative robustness: outputs is a real number
     '''
    
    # Transform pi[1][2] into pi_1_2 because the tool does not support the brackets

    new_formula = formula.replace('][','_')
    new_formula = new_formula.replace('pi[','pi_')
    
    new_formula = new_formula.replace('end_trace', str(len(pi[0][0])) )
    
    # Indices of where the 'pi' start
    variable_indices = [i for i in range(len(new_formula)) if new_formula.startswith('pi', i)]
    
    spec = rtamt.STLSpecification(language=rtamt.Language.PYTHON)
    #spec =  rtamt.StlDenseTimeSpecification(language=rtamt.Language.PYTHON)
    spec.name = 'STL Discrete-time Offline monitor'
    
    dataSet = {'time': list(np.arange(len(pi[0][0])))}
    
    already_declared = []
    
    for item in  variable_indices:
        indx1 = new_formula[item + 3] # +3 because it is the length of the string 'pi_'
        end = 4 # the index has at least length 1 ( = 4-3 )
        while True : # find the end of first index 
            if new_formula[item + end] == '_': break #the first index ends when '_' appears
            end += 1
        indx1 = int(new_formula[item + 3: item + end]  )
        
        start = item + end + 1 #from the position of '_' + 1 (start of second index)
        end = start + 1 
        while True: # find the end of second index 
            if new_formula[end] == ']' or new_formula[end] == ')': break #the second index ends when ' ' or ')' appear
            end += 1
        
        indx2 = int(new_formula[start : end])
        new_formula = new_formula[:end]+ ' ' + new_formula[end +1:]
        
        #Avoid already declared pairs of indices
        if f'{indx1}{indx2}' not in already_declared:
            spec.declare_var(f'pi_{indx1}_{indx2}', 'float')
            dataSet.update({f'pi_{indx1}_{indx2}' : pi[int(indx1)][int(indx2)]})
            already_declared.append(f'{indx1}{indx2}')

    if 'UNKNOWN' in new_formula:
        spec.declare_const('UNKNOWN', 'float', 99999.99)

    # Round the temporal parameters
    new_formula = round_timing_bound(new_formula)   
    
    spec.spec = new_formula
    spec.semantics = Semantics.STANDARD
    
    try:
        spec.parse()
    except rtamt.STLParseException as err:
        print(new_formula)
        print('STL Parse Exception: {}'.format(err))
        sys.exit()
        
    rho = spec.evaluate(dataSet)
    return rho[0][1]
    


def compute_fitness_positive(traces, formula, quantifiers, bool_temporal_operators):
    
    '''The function comuputes the POSITIVE fitness of a given formula.
    
    INPUTS:
        - traces: set of traces w.r.t which the score has to be evaluated.
        - formula: formula of the formula of which the score has to be computed
        - quantifiers: quantofoers to be applied to the formulas's trace variables
        - bool_temporal operators: 
            If bool_temporal_operators == True: there are temporal operators in the grammar --> use RTAMT
            If bool_temporal_operators == False: there are NOT temporal operators in the grammar --> use python function evaluate
       
    '''
    
    m = len(quantifiers)
    numb_traces = len(traces)
    
    results = []
    
    ## REMIND TO CHANGE ALSO INSIDE THE LAST LOOP OF THE FUNCTION!!!
    ##Compare pi also with itself
    # indices_traces_combo = [item for item in itertools.product(range(numb_traces), repeat=m)]
    
    ##Do not compare pi  with itself
    indices_traces_combo = [item for item in itertools.permutations(range(numb_traces), m)]
    
    ##STUDY SATISFACTION/VIOLATION OF EACH COMBINATION OF TRACES
    for ind in indices_traces_combo:
        
        pi = [traces[item] for item in ind] #pi is used inside eval(formula)
        
        #Positive --> Counts the number of satisfaction 
        
        # if SMT.evaluate_satisfaction(z3formula, pi) == 'sat': results.append(1) #satisfaction
        
        if bool_temporal_operators == False: 
            if eval(formula): results.append(1) #satisfaction
            else: results.append(0) #violation
        #Use RTAMT
        elif bool_temporal_operators == True:
            bool_output, _  = evaluate_rob_boolean(pi, formula, bool_interface_aware= False, input_variables= None, output_variables=None ) 
            if bool_output == 'sat': results.append(1) #satisfaction
            else: results.append(0) #violation
    
    sat = results.copy()
      
    ## EVALUATE FORMULA WITH QUANTIFIERS
    for i in range(0,m): #loop over the quantifiers
        aux = []
        
        #POSITIVE --> counts the number of satisfaction 
        # Answer the question: 
        # Which is the minimum number of combinations (of m traces) outcomes that I need to change to get a violation?
           
        # If the evaluation with the formula itself is not admitted
        for j in range(0, int(math.factorial(numb_traces)/math.factorial(numb_traces - (m-i-1)))):
            #EXISTS
            if quantifiers[-i-1]==0:  aux.append(sum( sat[j*(numb_traces-(m - i - 1)): (j+1)*(numb_traces-(m - i - 1))]) ) #(#traces - #left quantifiers)
            #FORALL
            elif quantifiers[-i-1]== 1: aux.append(min( sat[j*(numb_traces-(m - i - 1)): (j+1)*(numb_traces-(m - i - 1))]) )
        
        # If the evaluation with the formula itself is admitted
        # for j in range(0, numb_traces**(m-i-1)):
            # if quantifiers[-i-1]==0:  aux.append(sum( sat[j*numb_traces: (j+1)*numb_traces]) )
            # elif quantifiers[-i-1]== 1: aux.append(min( sat[j*numb_traces: (j+1)*numb_traces]) )
          
        sat = aux.copy()
    
    return sat[0]
    

def compute_fitness_negative(traces, formula, quantifiers, bool_temporal_operators):
    
    '''The function comuputes the NEGATIVE fitness of a given formula.
    
    INPUTS:
        - traces: set of traces w.r.t which the score has to be evaluated.
        - formula:  formula of which the score has to be computed
        - quantifiers: quantifiers to be applied to the formulas's trace variables
        - bool_temporal operators: 
            If bool_temporal_operators == True: there are temporal operators in the grammar --> use RTAMT
            If bool_temporal_operators == False: there are NOT temporal operators in the grammar --> use python function evaluate
       
    '''
    
    m = len(quantifiers)
    numb_traces = len(traces)
    
    results = []
    
    ## REMIND TO CHANGE ALSO INSIDE THE LAST LOOP OF THE FUNCTION!!!
    #Compare pi also with itself
    # indices_traces_combo = [item for item in itertools.product(range(numb_traces), repeat=m)]
    
    ##Do not compare pi  with itself
    indices_traces_combo = [item for item in itertools.permutations(range(numb_traces), m)]
    
    ##STUDY SATISFACTION/VIOLATION OF EACH COMBINATION OF TRACES
    for ind in indices_traces_combo:
        
        pi = [traces[item] for item in ind] #pi is used inside eval(formula)
        
        if bool_temporal_operators == False:
            if eval(formula): results.append(0) #satisfaction
            else: results.append(-1) #violation
            
        elif bool_temporal_operators == True:
            bool_output, _ = evaluate_rob_boolean(pi, formula, bool_interface_aware= False, input_variables= None, output_variables=None)
            if bool_output == 'sat': results.append(0) #satisfaction
            else: results.append(-1) #violation
    
    sat = results.copy()
      
    ## EVALUATE FORMULA WITH QUANTIFIERS
    for i in range(0,m): #loop over the quantifiers
        
        aux = []
        
        #NEGATIVE --> counts the number of violations 
        # Answer the question: 
        # Which is the minimum number of combinations (of m traces) outcomes that I need to change to get satisfaction?
         
        # If the evaluation with the trace itself is not admitted
        for j in range(0, int(math.factorial(numb_traces)/math.factorial(numb_traces - (m-i-1)))):
            #EXISTS
            if quantifiers[-i-1]==0:  aux.append(max( sat[j*(numb_traces-(m - i - 1)): (j+1)*(numb_traces-(m - i - 1))]) ) # change: every time we study (numb_traces -1)
            #FORALL
            elif quantifiers[-i-1]== 1: aux.append(sum( sat[j*(numb_traces-(m - i - 1)): (j+1)*(numb_traces-(m - i - 1))]) )
        
        #If the evaluation with the trace itself is admitted   
        # for j in range(0, numb_traces**(m-i-1)):
            # if quantifiers[-i-1]==0:  aux.append(max( sat[j*numb_traces: (j+1)*numb_traces]) ) # change: every time we study (numb_traces -1)
            # elif quantifiers[-i-1]== 1: aux.append(sum( sat[j*numb_traces: (j+1)*numb_traces]) )
            
            
        sat = aux.copy()
    
    return sat[0]
    

def compute_fitness(traces, formula, quantifiers, bool_temporal_operators):
    
    '''The function computes the positive cost if the most internal quantifier is EXISTS,
       and the negative cost if it is FOR ALL.
       In case the previous result is 0, the computation is repeated with the other measure of cost.
       
       If bool_temporal_operators == True: there are temporal operators in the grammar --> use RTAMT
       If bool_temporal_operators == False: there are NOT temporal operators in the grammar --> use python function evaluate
       '''
    
    if quantifiers[-1] == 0: # if the most internal quantifier is EXISTS
    
        cost = compute_fitness_positive(traces, formula, quantifiers, bool_temporal_operators)    
        if cost == 0: cost = compute_fitness_negative(traces, formula, quantifiers, bool_temporal_operators)
    
    elif quantifiers[-1] == 1: # if the most internal quantifier is FOR ALL
    
        cost = compute_fitness_negative(traces, formula, quantifiers, bool_temporal_operators)    
        if cost == 0: cost = compute_fitness_positive(traces, formula, quantifiers, bool_temporal_operators)
    
    return cost


def efficient_monitoring(traces, formula, quantifiers, bool_temporal_operators = True, print_counter = None ,  nodes = [], bool_interface_aware= False, input_variables= None, output_variables=None):
    
    '''The function computes whether the set of traces satisfies the formula with the given quantifiers.
    In this case, the cost is not computed: but only the boolean satisfaction of the hyperformula.
    In this case, the forall is stopped as soon as one tuple is violated, 
    and the exists is stopped as soon as one tuple is satisfied.
    
    OUTPUT: +1 (satisfied) or -1 (violated). With n_subset = False, OUTPUT: +-1 , counter (number of calls to ltl monitor)'''
    
    if print_counter is False:  bool_return_counter = True
    else: bool_return_counter = False
    
    
    #!!! Indices are tought for not evaluating the formula with the same trace repeated more than once
    m = len(quantifiers)
    numb_traces = len(traces)
    
    ##Do not compare pi  with itself
    indices_traces_combo = [item for item in itertools.permutations(range(numb_traces), m)]
    
    #Compare pi also with itself
    # indices_traces_combo = [item for item in itertools.product(range(numb_traces), repeat=m)]
    
    
    ind = [item for item in range(m)]
    
    index_to_skip = None #index to skip in case a satisfied/violated tuple has been found (E/F, respectively)
    value_to_skip  = None
    sat_skip = None #either 0 or 1
    
    counter = 0
    results = [[]] * m #Inizialization of the vector of the results. Each component refers to a quantifiers
    
    ## STUDY SATISFACTION/VIOLATION OF EACH COMBINATION OF TRACES
    #For each combinaion of traces
    for ind in indices_traces_combo:
        
        #If we can skip (due to already satisfied exists or already violated forall)
        if index_to_skip is not None: 
            #If we are still on the index to skip
            if ind[index_to_skip] == value_to_skip: 
                results[-1] = results[-1] + [sat_skip]
            #If we have skipped the index to skip    
            elif ind[index_to_skip] != value_to_skip:
                 index_to_skip = None
                 
        #If there are NOT already satisfied-exists or already violated-forall  
        if index_to_skip is None:
            
            #Evaluate the satisfaction/violation of the current tuple of traces
            pi = [traces[item] for item in ind] #pi is used inside eval(formula)
            
            if bool_temporal_operators == False: 
                if eval(formula): results[-1] = results[-1] + [1] #satisfaction
                else: results[-1] = results[-1] + [0] #violation
            #Use RTAMT
            elif bool_temporal_operators == True:
                counter += 1
                
                if counter > 200000: 
                    #print('Runout')
                    if bool_return_counter : return 'runout', counter #-1#
                    else: return 'runout' 
                
                bool_output, vec_output = evaluate_rob_boolean(pi, formula , bool_interface_aware, input_variables , output_variables)
                
                if  bool_output == 'sat': results[-1] = results[-1] + [1] #satisfaction
                elif bool_output == 'unsat': results[-1] = results[-1] + [0] #violation
                else:  results[-1] = results[-1] + ['skipped']
            
                
        #Loop over quantifiers starting from the most internal one
        for i in range( m-1 , 0 , -1): 
            # If the list is completed
            if len(results[i]) == numb_traces - i: ###! Change if same trace can be repeated (== numb_traces)
            
                if quantifiers[i] == 0: #EXISTS
                    if 1 in results[i]: results[i-1] = results[i-1] + [1] #satisfied
                    else: results[i-1] = results[i-1] + [0] #violated
                
                elif quantifiers[i] == 1: #FORALL 
                    if 0 in results[i]: results[i-1] = results[i-1] + [0] #violated
                    else: results[i-1] = results[i-1] + [1] #satisfied
                
                results[i] = []
                
            # If the list can be completed automatically with success (EXISTS) 
            elif (index_to_skip is None or index_to_skip > i-1) and quantifiers[i] == 0 and 1 in results[i]:  
                sat_skip = 1
                index_to_skip = i - 1
                value_to_skip = ind[i - 1]
            
            # If the list can be completed automatically with insuccess (FORALL) 
            elif (index_to_skip is None or index_to_skip > i-1) and quantifiers[i] == 1 and 0 in results[i]:  
                sat_skip = 0
                index_to_skip = i - 1
                value_to_skip = ind[i - 1]
            
            else: #Continue normally
                break
        #Check on the most external quantifier (early stop for EXISTS)
        if quantifiers[0] == 0 and 1 in results[0]: 
            if bool_return_counter: return 1, counter
            else: return 1 #satisfaction
        #Check on the most external quantifier (early stop for FORALL)
        elif quantifiers[0] == 1 and 0 in results[0]: 
            if bool_return_counter: return -1, counter
            else: return -1 #violation
            
    if quantifiers[0] == 0 :#EXISTS
        
        if 1 in results[0]: 
            if bool_return_counter: return 1, counter
            else: return 1 #satisfaction
        else: 
             if bool_return_counter: return -1, counter
             else: return -1 #violation
    
    elif quantifiers[0] == 1 : #FORALL
        if 0 in results[0]: 
             if bool_return_counter: return -1, counter
             else: return -1 #violation
        elif bool_interface_aware and 1 not in results[0]:
            if bool_return_counter: return -1, counter
            else: return -1  #violation
        else: 
             if bool_return_counter: return 1, counter
             else: return 1  #satisfaction
   
def compute_robustness_STL(traces, formula, quantifiers):
    
    '''The function computes the robustness of the pair (quantifiers, formula) with respect to the current traces'''
    
    m = len(quantifiers)
    numb_traces = len(traces)
    
    results = []
    
    ## REMIND TO CHANGE ALSO INSIDE THE LAST LOOP OF THE FUNCTION!!!
    ## Compare pi also with itself
    # indices_traces_combo = [item for item in itertools.product(range(numb_traces), repeat=m)]
    
    ## Do not compare pi  with itself
    indices_traces_combo = [item for item in itertools.permutations(range(numb_traces), m)]
    
    ##STUDY SATISFACTION/VIOLATION OF EACH COMBINATION OF TRACES
    for ind in indices_traces_combo:
        pi = [traces[item] for item in ind]
        #Use RTAMT
        results.append(evaluate_rob_quantitative(pi, formula)) #append the computed robustness
        
    sat = results.copy()
      
    ## EVALUATE FORMULA WITH QUANTIFIERS
    for i in range(0,m): #loop over the quantifiers
        
        aux = []
        
        # If the evaluation with the formula itself is not admitted
        for j in range(0, int(math.factorial(numb_traces)/math.factorial(numb_traces - (m-i-1)))):
            #EXISTS
            if quantifiers[-i-1]==0:  aux.append(max( sat[j*(numb_traces-(m - i - 1)): (j+1)*(numb_traces-(m - i - 1))]) ) #(#traces - #left quantifiers)
            #FORALL
            elif quantifiers[-i-1]== 1: aux.append(min( sat[j*(numb_traces-(m - i - 1)): (j+1)*(numb_traces-(m - i - 1))]) )
        
        # If the evaluation with the formula itself is admitted
        # for j in range(0, numb_traces**(m-i-1)):
            #EXISTS
            # if quantifiers[-i-1]==0:  aux.append(max( sat[j*numb_traces: (j+1)*numb_traces]) )
            #FORALL
            # elif quantifiers[-i-1]== 1: aux.append(min( sat[j*numb_traces: (j+1)*numb_traces]) )
          
        sat = aux.copy()
    
    return sat[0]

def lookup_table_uniform_distance(traces):
    
    '''The function computes the lookup table that associate each pair of
        traces with their uniform distancee.
        Thee matrix is symmetric with diagonal of zeros'''
        
    numb_traces = len(traces)
    table = np.zeros((numb_traces, numb_traces))
    
    for i in range(numb_traces):
        matrix_trace_i = np.matrix(traces[i])
        
        for j in range(i+1, numb_traces):
           table[i,j] = np.linalg.norm(matrix_trace_i-np.matrix(traces[j]), ord = np.inf ) #uniform norm . multidimensional?
           table[j,i] = table[i,j]
    
    return table


def monitor_hyperSTL_correctness(traces, formula, quantifiers):
       
    '''The function computes the robustness of the pair (quantifiers, formula) with respect to the current traces
    using the correctness theorem for the quantitative semantics of STL:
        
    If w1 satsifies phi with robustness rho and |w1-w2|<rho then w2 satisfies phi as well'''
    
    
    ### !!!! SET TO TRUE TO RUN stl_monitoring.py for salability results
    bool_return_counter = False
    
    m = len(quantifiers)
    numb_traces = len(traces) 
    # print(formula)
    
    #Table of distances between pairs of traces
    lookuptable = lookup_table_uniform_distance(traces)
    
    #Table of the robustness
    rob_ind_table = [ ] # indices of the tuples of traces
    rob_rho_table = [ ] # indices of values of robustness
    
    ##Do not compare pi  with itself
    indices_traces_combo = [item for item in itertools.permutations(range(numb_traces), m)]
    
    #Compare pi also with itself
    # indices_traces_combo = [item for item in itertools.product(range(numb_traces), repeat=m)]
    
    ind = [item for item in range(m)]
    
    index_to_skip = None #index to skip in case a satisfied/violated tuple has been found (E/F, respectively)
    value_to_skip  = None
    sat_skip = None #either 0 or 1
    
    counter_monitor = 0
    counter_skip_correctness =0
    results = [[]] * m #Inizialization of the vector of the results. Each component refers to a quantifiers
    
    ## STUDY SATISFACTION/VIOLATION OF EACH COMBINATION OF TRACES
    #For each combinaion of traces
    for ind in indices_traces_combo:
        
        #If we can skip (due to already satisfied exists or already violated forall)
        if index_to_skip is not None: 
            #If we are still on the index to skip
            if ind[index_to_skip] == value_to_skip: 
                results[-1] = results[-1] + [sat_skip]
            #If we have skipped the index to skip    
            elif ind[index_to_skip] != value_to_skip:
                 index_to_skip = None
                 
        #If there are NOT already satisfied-exists or already violated-forall  
        if index_to_skip is None:
            
            
            skip_rob = False #if true: the evaluation of the robustness can be skipped,
            # because the correctness theorem can be applied
             
            #Compare the current indices with the indices of previously computed robustness
            for index_corr, item_corr in enumerate(rob_ind_table):
                
                #Indices of components with different values
                different_components = [i for i, (a, b) in enumerate(zip(item_corr, ind)) if a != b]
                
                #If there is a tuple of traces with all but one equal components
                if len(different_components) == 1:
                    
                   # Distance between the two traces corresponding to the index that differs in the tuple
                   distance = lookuptable[ item_corr[different_components[0]] ,ind[different_components[0]]]
                    
                   #If the distance is smaller than the abs value of the robustness
                   if distance < abs(rob_rho_table[index_corr]):
                       
                       if rob_rho_table[index_corr] >=0: results[-1] = results[-1] + [1] #satisfaction
                       elif rob_rho_table[index_corr]<0: results[-1] = results[-1] + [0] #violation
                       #skip the computation of the robustness for the current tuple of traces
                       skip_rob = True
                       counter_skip_correctness += 1
                       break
                
            #(Correctness theorem cannot be applied) The robustness needs to be computed
            if skip_rob == False:
                
                #Evaluate the satisfaction/violation of the current tuple of traces
                pi = [traces[item] for item in ind] #pi is used inside eval(formula)
                counter_monitor += 1
                if counter_monitor > 200000: 
                    if bool_return_counter : return 'runout', counter_monitor, counter_skip_correctness
                    else: return 'runout' #-1#
                
                rho = evaluate_rob_quantitative(pi, formula)
                
                #Update table of the robustness
                rob_ind_table.append(ind)
                rob_rho_table.append(rho)
                
                if  rho >= 0: results[-1] = results[-1] + [1] #satisfaction
                elif rho < 0: results[-1] = results[-1] + [0] #violation
            
            #If the current tuple has NOT been sampled to be evaluated -> skip the monitor  
            # else:  results[-1] = results[-1] + ['skipped']
                
        #Loop over quantifiers starting from the most internal one
        for i in range(m-1 , 0 , -1): 
            # If the list is completed
            if len(results[i]) == numb_traces - i: ###! Change if same trace can be repeated (== numb_traces)
            
                if quantifiers[i] == 0: #EXISTS
                    if 1 in results[i]: results[i-1] = results[i-1] + [1] #satisfied
                    else: results[i-1] = results[i-1] + [0] #violated
                
                elif quantifiers[i] == 1: #FORALL 
                    if 0 in results[i]: results[i-1] = results[i-1] + [0] #violated
                    else: results[i-1] = results[i-1] + [1] #satisfied
                
                results[i] = []
                
            # If the list can be completed automatically with success (EXISTS) 
            elif (index_to_skip is None or index_to_skip > i-1) and quantifiers[i] == 0 and 1 in results[i]:  
                sat_skip = 1
                index_to_skip = i - 1
                value_to_skip = ind[i - 1]
            
            # If the list can be completed automatically with insuccess (FORALL) 
            elif (index_to_skip is None or index_to_skip > i-1) and quantifiers[i] == 1 and 0 in results[i]:  
                sat_skip = 0
                index_to_skip = i - 1
                value_to_skip = ind[i - 1]
            
            else: #Continue normally
                break
        #Check on the most external quantifier (early stop for EXISTS)
        if quantifiers[0] == 0 and 1 in results[0]: 
            if bool_return_counter :return  1 , counter_monitor, counter_skip_correctness
            else: return 1 #satisfaction
        #Check on the most external quantifier (early stop for FORALL)
        elif quantifiers[0] == 1 and 0 in results[0]: 
            if bool_return_counter : return -1, counter_monitor, counter_skip_correctness
            else: return -1 #violation
            
    if quantifiers[0] == 0 :#EXISTS
        
        if 1 in results[0]: 
            if bool_return_counter : return 1, counter_monitor, counter_skip_correctness
            else:return 1 #satisfaction
        else: 
            if bool_return_counter : return -1, counter_monitor, counter_skip_correctness
            else: return -1 #violation
    
    elif quantifiers[0] == 1 :#FORALL
        if 0 in results[0]: 
             if bool_return_counter : return  -1, counter_monitor, counter_skip_correctness
             else: return -1 #violation
        else: 
             if bool_return_counter :  return 1, counter_monitor, counter_skip_correctness
             else: return 1 #satisfaction



def check_success_stopping_condition(inputs, cost, msc = None, ratio_fp = None, ratio_fn = None ):
    """
    Check the SUCCESS stopping criterion based on the inputs and cost.

    Args:
        inputs: The inputs of the mining algorithm.
        cost: The cost value to evaluate.

    Returns:
        bool: True if the stopping criterion is met, False otherwise.
    """
    # Stopping condition for the first version of the mining algorithm: just satisfaction
    if inputs.kind_score in ['fitness', 'stl_robustness', 'efficient_monitoring', 'efficient_monitoring_stl']:
        if cost == 'runout' or cost < 0: return False
        elif cost > 0: return True
    # Stopping condition for the new (second) version of the algorithm: objective function below a threshold epsilon_stop
    elif inputs.kind_score == 'obj_function':
        if cost < inputs.epsilon_stop: 
            print('Stopping condition met for cost = ', cost)
            return True
        elif msc is not None and msc < 0.01: 
            print('Stopping condition met for msc = ', msc)
            return True
        elif (ratio_fp is not None and ratio_fp < 0.05)\
            and (ratio_fn is not None and ratio_fn < 0.05):
            print('Stopping condition met for ratio_fp = ', ratio_fp, ' and ratio_fn = ', ratio_fn)
        else: return False

        



def cost(traces_pos, traces_neg, formula, inputs, inputs_mon, count, mode = 'cost'):

    '''mode is used only in case of obj_function and states whether the cost is computed, or msc'''

    #Efficient_monitoring and efficient_monitoring_stl are the two boolean fitness function implemented.    
    
    #Avoid recomputing violated formulas
    if ( inputs.kind_score == 'efficient_monitoring' or inputs.kind_score == 'efficient_monitoring_stl') \
        and set([print_formula(formula.quantifiers, formula.string)]).issubset(inputs_mon.set_violated_formulas):
            cost = -1
    #Avoid recomputing satisfied formulas              
    elif ( inputs.kind_score == 'efficient_monitoring' or inputs.kind_score == 'efficient_monitoring_stl')\
        and set([print_formula(formula.quantifiers, formula.string)]).issubset(inputs_mon.set_satisfied_formulas) :
                cost = 1
                    
    else:
        count.hyper_monitor += 1
        count.hyper_monitor_tot += 1
        
        if inputs.kind_score == 'fitness': 
            cost = compute_fitness(traces_pos, formula.string, formula.quantifiers, inputs.bool_temporal_operators)
        
        elif inputs.kind_score == 'stl_robustness' : 
              cost = compute_robustness_STL(traces_pos, formula.string, formula.quantifiers)
                    
        elif inputs.kind_score == 'efficient_monitoring': 
              cost = efficient_monitoring(traces_pos, formula.string, formula.quantifiers, inputs.bool_temporal_operators, inputs_mon.print_counter, 
                                                         formula.nodes, inputs.bool_interface_aware, inputs.input_variables, inputs.output_variables)

        elif inputs.kind_score == 'efficient_monitoring_stl': 
            cost = monitor_hyperSTL_correctness(traces_pos, formula.string, formula.quantifiers)

        elif inputs.kind_score == 'obj_function':
            cost = metrics.compute_obj_fun(traces_pos, traces_neg, formula.string, verbose = -1, mode = mode)

    if inputs.kind_score != 'obj_function':
        # Add to data structures recording satisfied and violated formulas        
        if check_success_stopping_condition(inputs, cost): inputs_mon.set_satisfied_formulas = inputs_mon.set_satisfied_formulas.union(set([print_formula(formula.quantifiers, formula.string)]))
        else:  inputs_mon.set_violated_formulas = inputs_mon.set_violated_formulas.union(set([print_formula(formula.quantifiers, formula.string)]))
     
    return cost
    
    
    
                 