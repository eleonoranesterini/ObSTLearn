class InputData():
    
    def __init__(self, seed, numb_quantifiers, grammar_quantifiers, grammar_structure, grammar_predicate, max_length, \
                 length = None, learning_stl = False, par_bounds = None, variable_ranges = None, fraction = 10, kind_score = 'efficient_monitoring',
                   epsilon_stop = 0.05, horizon = 0):
         
        # seed : number representing the seed
        self.seed = seed 
        
        
        '''GRAMMAR DEFINITION - for further details on the grammar definition see below'''
        
        '''Quantifiers'''
        
        # numb_quantifiers :[min_numb_quantifiers , max_numb_quantifiers] interval of admissible numbers of quantifiers
        # e.g. numb_quantifiers = [2,3] to only allow for 2 or 3 quantifiers
        self.numb_quantifiers = numb_quantifiers 
        
        # grammar_quantifiers : string describing the grammar that defines the quantifiers
        # See below for a detailed description.
        self.grammar_quantifiers = grammar_quantifiers
        
        
        '''Formula structure'''
        
        # grammar_structure : string describing the grammar that defines the quantifier-free formula structure.
        # Max 10 levels of specification levels
        # See below for a detailed description.
        self.grammar_structure = grammar_structure
        
        
        '''Predicate'''
        
        # grammar_predicate : list describing the grammar that defines the admissible predicates.
        # See below for a detailed description. 
        # Further options for the definition of the predicates can be found in the 'Additional Options' section.
        self.grammar_predicate = grammar_predicate
        
        
        '''Formula length'''
        
        # max_length : positive integer number representing the maximum allowed length for the learned formulas
        self.max_length =  max_length
        
        # l : positive integer number representing the EXACT LENGTH of the learned formula
        # if l is specified, max_length is not taken into account
        self.length = length #None by default
        
        
        '''STL options'''
        
        # learning_stl : boolean variable to indicate whether STL/HyperSTL is learned (True) or not (False)
        # This indication is needed for the specification of the type of predicate and the learning of the parameters
        # By default, it is set to False
        self.learning_stl = learning_stl #False
        
        # par_bounds : list of intervals indicating the numerical bounds of the predicates. To be specified only when learning_stl == True.
        # e.g. par_bounds = [[1,2], [-1.5,90]] indicates the first parameter varies in the range [1,2], while the second one in [-1.5,90]
        self.par_bounds = par_bounds #None #if learning_stl, then par_bounds

        self.variable_ranges = variable_ranges #List of lists with actual min and max values for each variable.
        
        # fraction is used to regulate the threshold fo the uncertainty region area in the parameter synthesis phase.
        # The threshold is set as (total_parameter_area/fraction) and the itertive procedure of mining parameters stops when the uncertanty region has an area below the threshold.
        # By default, fraction is set to 10. This means that the parameter synthesis stops when the uncertanty region has an area below one tenth of the total area of the parameter space.
        self.fraction = fraction #10
        
        
        '''Score'''
        
        # kind_score : string indicating the score to be used during the learning process.
        # 4 options:
        #   - 'efficient_monitoring' : efficient monitoring technique
        #   - 'fitness' : fitness function described in the paper
        #   - 'stl_robustness' : quantitative semantics of STL exentded to HyperSTL. To be used only when learning_stl == True
        #   - 'efficient_monitoring_stl' : efficient monitoring technique improved with the correctness property of STL. To be used only when learning_stl == True
        #By default, it is set the option 'efficient_monitoring'
        self.kind_score =  kind_score # 'efficient_monitoring' #one among: 'fitness','stl_robustness',, 

        # For kind_score == objective function, this value determines the stopping condition for the learning process.
        # Namely, the learning process stops when the objective value is below the threshold.
        self.epsilon_stop = epsilon_stop #0.001 
        
        
        '''Additional options'''
     
        # bool_interface aware : Boolean variable that indicates whether the user specifies the nature of the variables (input/output) - True - or not - False.
        # E.g., bool_interface aware should be set to True if the user wants to avoid vacuously satisfied implications.
        # By default, it is set to False
        self.bool_interface_aware = False
        
        
        # If bool_interface =+ True, input_variables : list of input variables
        # e.g. input_variables = [0,3,4] indicates that the 0th, 3rd and 4th variables are inputs.
        self.input_variables = None
        
        
        # If bool_interface =+ True, output_variables : list of output variables
        # e.g. output_variables = [1,2,5] indicates that the 1st, 2nd and 5th variables are outputs.
        self.output_variables = None
        
        
        # bool_temporal operators: Boolean variable that indicates whether temporal operators are present in the grammar (True) or not (False).
        # By default, it is set to True. It should be specified as False, if no temporal operators are present.
        self.bool_temporal_operators = True 
        
        
        # bool_mutation_quantifiers : Boolean variable that indicates whether the quantifiers should be mutated during the learning process (True) or not (False).
        # By default, it is set to True.
        # When the quantifiers are fixed (e.g., forall forall), bool_mutation_quantifiers should be set to False to speed up the learning process.
        self.bool_mutation_quantifiers = True
        
        
        # bool_different_variables: Boolean variable that indicates whether different variables are admitted in the same predicates (True) or not (False).
        # e.g., in the trigonometric function case study, the predicates should involve different variables (e.g., x > cosine ), so this variable was set to True.
        # e.g., in the autonomous car parking case study, the predicates should involve only one variable (e.g., pi_1(velocity) > pi_2(velocity) or pi_1(distance) > pi_2(distance), so this variable was set to False.
        # By default, it is set to False
        self.bool_different_variables = False 
        
        
        # bool_same_number : Boolean variable that indicates whether the same predicate name in the grammar_predicate refers to the exact same predicate (True) or not (False).
        # If bool_same_number == True, every time 'predicate0' is used, it is always associated with a concrete predicate (e.g., pi_1(x) == p_2(x) )
        # If bool_same_number == False, 'predicate0' refers in general to the grammar defined for predicate0 (e.g., p_i(k) == p_j(k) with i,j in {1,2,3} and k variable in {x,y,z}).
        # In all our case studies, this variable was set to False
        #By default, it is set to False
        #self.bool_same_number =  False 
        
        
        #second_variable : string (or list of numbers) indicating the kind of terms admitted in the predicates.
        # 3 options:
        #    - 'variable' : if both the two terms in a predicate can be one among the admissible terms from the grammar_predicates, such as pi_1(x) == p_2(x).
        #    - 'true'     : if one of the two terms in the predicates should be the constant True, such as pi_1(x) == True (this is used for the temporal testers case study)
        #    - [0, 1 ,2,..] : for logs, if one of the two terms in the predicates should be a number representing a certain action, such as pi_1(x)==1 (this is used for the dining philosophers case study)
        #By default, it is set to 'variable'
        self.second_variable = [['variable']] * len(grammar_predicate)
        
        
        #to_be_present : list of two elements to specify expressions that need to be present in the formula candidate - in the predicates - (If absent, the formula is rejected)
        #The first element is the list of strings that have to be included in the predicates , the second element is the minimum number of elements in to_be_present[0] that have to appear in the formula
        # For example, in the temporal testers with template: to_be_present = [ ['pi[0][0]' , 'pi[0][1]'], 2] i.e., two different variables involved in the formula
        # For example, in the dining philosohers: to_be_present = [[f'pi[0][{agent}] == {action}' for agent in range(n) for action in input_data.second_variable ], 2 ] ; i.e., at the least 2 different predicates should be included
        self.to_be_present = None
        
        
        # store_numeration_formulas : Boolean variable that indicates whether the list of possible formulas derived from the grammar should be stored in a file (True) or not (False).
        # For each level of grammar_structure and number of quantifiers, it is created a folder named 'spec_{spec_level}_{numb_quantifier}'. 
        # This contains the enumeration of formulas from the grammar organized per length. 
        # Different lengths are added during the execution of the program and not all at once at the beginning of the run.
        # By default, it is set to False. In this case, the existing folder 'grammar' is deleted.
        self.store_enumeration_formulas = False
        
        
        # store_formulas : Boolean variable that indicates whether the learned formulas should be stored in a file (True) or not (False).
        # The formulas are stored as objects whose attributes are: quantifiers, nodes of the syntax tree, length, fitness value and time needed to learn it.
        # To transform the syntax tree into a readable string, use the function:  tree.NATURAL_tree_to_formula
        # To print the hyperproperty, use the function: syntax_guided.print_formula
        # By default, store_formulas is set to False
        self.store_formulas = False
        
        
        # name_set_mined_formulas is used to give a name to the folder containing the set of mined formulas.
        # The final name of such set will be: {name_set_mined_formula}learned_formula_seed{seed}
        # To use this option, the option store_formulas has to be set to True
        self.name_output_folder = 'Results'
        
        
        # n_target : positive integer value representing the target number of formulas to be mined.
        # The final number of mined formulas can be less than n_target (maximum number of iterations reached before reaching the target), but it can never be greater than n_target.
        # By default, it is set to 50.
        self.n_target = 50
        
        # The horizon for the prediction. It must be an integer value indicating the number of time steps ahead to predict.
        self.horizon = horizon
        
        
   
'''GRAMMAR DEFINITION'''   

'''Definition of the grammar for the quantifiers'''

## grammar_quantifiers is a string that defines the grammar for the admissible sequences of quantifiers.
## It is organized in different levels, one for each line. 
## Each line starts with: 'psi_{level_number} ==  ' (level_number is a sequential number starting from 0)
## Different options in the same level are separated with the vertical line |
## The term 'phi' refers to the quantifier-free part of the formula whose grammar is defined in the grammar_structure. The number after phi refers to the grammar level in grammar_structure. 
## The forall quantifier is specified with the term 'forall', the exists quantifier with the term 'exists'

## Let us consider some examples:
    
## Only the forall quantifier is admitted (no restrictions on the number of the forall quantifiers):

# grammar_quantifiers = '''
# psi0 == forall psi0 | phi0
# '''



## Exactly two exists quantifiers:
    
# grammar_quantifiers = '''
# psi0 == exists exists phi0
# '''



## The first quantifier has to be a forall, then any sequence of quantifier is admitted:
    
# grammar_quantifiers = '''
# psi0 == forall psi1 
# psi1 == forall psi1 | exists psi1 | phi0
# '''



## Only an even number of quantifiers is admitted:
    
# grammar_quantifiers = '''   
# psi0 == forall forall psi0 | forall exists psi0 | exists forall psi0 | exists exists psi0 | phi0
# '''    



## The forall quantifier should be used when using the 0th level of the quantifier-free fromula grammar,
## and the exists quantifier should be used when using the 1st level of the quantifier-free fromula grammar.

# grammar_quantifiers = '''
# psi0 ==  psi1 | psi2
# psi1 == forall psi1 | phi0
# psi2 == exists psi2 | phi1
# '''



## Important: If 'phi' is not present in at least one level, it is impossible to exit the loop of adding quantifiers


'''Definition of the grammar for the quantifier-free formula structure'''

## grammar_structure is a string that defines the grammar for the quantifier-free formula structure.
## It is organized in different levels, one for each line. 
## Each line starts with: 'phi_{level_number} ==  ' (level_number is a sequential number starting from 0)
## Different alternatives in the same level are separated with the vertical line | 
## Each option has to contain ONLY ONE operator or predicate.
## The term 'predicate' is followed by a number referring to the level of the grammar in the definition of the admissible predicates - grammar_predicate.

## Let us consider some examples:
    
##Example of STL grammar
# grammar_structure = '''
# phi0 ==  eventually phi0 | always phi0 | phi0 until phi0 | phi0 implies phi0 | phi0 and phi0 | phi0 or phi0 | not phi0 | predicate0
# '''

##Example of grammar for the formula: predicate0 implies predicate0
# grammar_structure = '''
# phi0 = phi1 implies phi1
# phi1 = predicate0
# '''



##Admissible operators for NON-TEMPORAL properties/Hyperproperties:
    
# and
# or
# not 
# implies
# <->
# predicate    



##Admissible operators for LTL/HyperLTL: 
    
# eventually   
# always   
# until
# weak_u    # for the weak version of the until operator
# next      # for the weak version of the next operator             
# s_next    # for the strong version of the next opertor        
# prev      # for the weak version of the previous operator
# s_prev    # for the strong version of the previous operator
# once
# historically
# since
# and 
# or
# not 
# implies   # for the implication 
# <->     # for the equivalence 
# predicate
 


##Admissible operators for STL/HyperSTL: 
    
# eventually
# always 
# until 
# and
# or 
# not  
# implies
# predicate
 
## In case of PARAMETRIC temporal bounds, indicate them in square brackets. 
## The parameters have to be indicated as 'espilon{number}-' (e.g., 'always[epsilon0-,epsilon1-]')
## {number} refers to a sequential integer number (starting from 0) that identifies a particular parameter symbol.
##  All parameter symbols having the same number are identified as the same parameter (i.e.,in the end, they all will assume the same value)
##  Parameter symbols and parameter values can be used in the same operator, e.g., 'eventually[0, epsilon3-]' 


## IMPORTANT!
## Each alternative of each line has to contain EXACTLY ONE OPERATOR OR ONE PREDICATE:
## e.g.,  it is NOT possible to have 'always(predicate0 implies predicate1)' - this can be rewritten as:
# grammar_structure = '''
# phi0 ==  always phi1
# phi1 ==  phi2 implies phi3
# phi2 ==  predicate0
# phi3 ==  predicate1
# '''
## and it is NOT possible to have 'phi2 and phi3 and phi4' - this can be rewritten as:
# grammar_structure = '''
# phi0 ==  phi1 and phi2
# phi1 ==  phi3 and phi4
# phi2 ==  ...
# phi3 ==  ...
# phi4 ==  ...
# '''



'''Definition of the grammar for predicates : non-STL/HyperSTL formulas'''

## If learning_stl == False, grammar_predicates is a list of elements representing sets of admissible predicates.
## Each set of admissible predicates is indicated as predicate{number}, where {number} is a sequential positive integer number (starting from 0).

## Each predicate{number} is a list of three elements:
##     - admissible relational symbols (e.g., '==', '>','>=', '<=', '<')
##     - admissible variables (e.g., [0,1,2] to indicate the 0th, 1st and 2nd variable)
##     - admissible tuples of trace variables (e.g., [(0,1),(1,0)] to indicate (p_0,p_1) and (p_1,p_0))

## So, predicate{number} =  [ [admissible_symbols], [admissible_variables], [admissible_tuples_trace_variables]]
## Insead of [the list of admissible_tuples_trace_variables] the user can use 'all tuples' if all the pairs of trace variables are admitted

## E.g. predicate0 = [ ['>'], [2,3], [(0,1), (0,1)] ] 
## It admits the following 4 predicates: 
##          pi_0[2] > pi_1[2]  
##          pi_0[3] > pi_1[3] 
##          pi_1[2] > pi_0[2],
##          pi_1[3] > pi_0[3]



## Example with n variables admitted and all pairs of trace variables 

# grammar_predicate = []
# predicate0 = [['=='], list(range(0,n)), 'all tuples' ]
# grammar_predicate.append([predicate0])



## Example of two sets of admissble predicates

# grammar_predicate = []
# predicate0 = [['=='], [0,2], [ (0,1)(1,0)] ]# admssible predicates: pi_0[0]==pi_1[0], pi_0[2]==pi_1[2], pi_1[0]==pi_0[0], pi_1[2]==pi_0[2].
# grammar_predicate.append([predicate0])
# predicate1 = [['>'], [1], [ (0,2),(2,0) ] ] # admssible predicates: pi_0[1]>pi_2[1], pi_2[1]>pi_0[1]
# grammar_predicate.append([predicate1])

 

'''Definition of the grammar for predicates : STL/HyperSTL formulas '''

## If learning_stl == True, the grammar for predicates is a list of the admissible predicates.
## The predicates can be PARAMETRIZED. The parameters symbols are indicated as 'epsilon{number}-', 
## {number} refers to a sequential integer number (starting from 0) that identifies a particular parameter symbol.
## All parameter symbols having the same number are identified as the same parameter (i.e.,in the end, they all will assume the same value)
## There is no distinction between temporal and magnitude parameters, so, if some parameter numbers have already been used for temporal operators (e.g., 0,1,2),
## the parameters in grammar_predicate should start from the first parameter number  which is free (3 in the example).

## Traces variables are indicated as pi[0], pi[1], pi[2] ... The second component of each of these lists represents the value of the variable in the trace.
## For example: 
##          pi[0]:  one generic trace
##                  pi[0][0] values of variable x in the trace pi[0]
##                  pi[0][1] values of variable y in the trace pi[0]
##                  pi[0][2] values of variable z in the trace pi[0]
##
##          pi[1]:  another generic trace
##                  pi[1][0] values of variable x in the trace pi[1]
##                  pi[1][1] values of variable y in the trace pi[1]
##                  pi[1][2] values of variable z in the trace pi[1]


## Example of grammar_predicate for STL( --> 1 quantifier --> 1 trace variable --> pi[0] only ):
# grammar_predicate = ['( pi[0][0] < epsilon2- )',   '( pi[0][1] > epsilon3- )' ]

## Example of grammar_predicate for HyperSTL with 2 quantifiers (--> 2 trace variables --> pi[0] and pi[1]):
# grammar_predicate = [   '( abs( pi[0][0] - pi[1][0] ) < epsilon3- )', '( abs( pi[0][1] - pi[1][1] ) < epsilon4- )']



'''EXAMPLE: HyperLTL grammar'''

# grammar_quantifiers = '''
# psi0 == forall psi0 | exists psi0 | phi0
# '''

# grammar_structure = '''
# phi0 = always phi0 | eventually phi0 | phi0 until phi0 | s_next phi0 | s_prev phi0 | phi0 and phi0 | phi0 or phi0 | phi0 implies phi0 | phi0 <-> phi0 | not phi0 | predicate0
# '''

# grammar_predicate = []
# predicate0 = [['=='], list(range(0,n)), [ item for item in itertools.permutations(range(m), min(m,2))] ]
# grammar_predicate.append([predicate0])



'''EXAMPLE: Observational Determinism template'''

# grammar_quantifiers = '''
# psi0 == forall forall phi0
# '''

# grammar_structure = '''
# phi0 == phi1 implies phi3
# phi1 == always phi2
# phi2 == phi2 and phi2 | predicate0
# phi3 == always phi4
# phi4 == phi4 and phi4 | predicate1
# '''

# grammar_predicate = []
# predicate0 = [['=='], list(range(0,n-1)), [ item for item in itertools.permutations(range(m),  min(m,2))] ] #involving variables from 0th to  (n-2)th
# grammar_predicate.append([predicate0])
# predicate1 = [['=='], list(range(n-1,n)), [ item for item in itertools.permutations(range(m),  min(m,2))] ] #involving the variable (n-1)th
# grammar_predicate.append([predicate1])

#bool_interface_aware = True #for non vacuously satisfied formulas
#input_variables = list(range(0,n-1)) # the first n-1 variables are input
#output_variables = [n-1] # the last variable is the output



'''EXAMPLE: STL grammar'''

# grammar_quantifiers = '''
# psi0 == forall phi0
# '''

##Temporal operators can be either bounded or unbounded
# grammar_structure = '''
# phi0 = always[epsilon0-,epsilon1-] phi0 | eventually[epsilon0-,epsilon1-] phi0 | phi0 until phi0 | phi0 and phi0 | phi0 or phi0 | phi0 implies phi0 | not phi0 | predicate0
# '''

#grammar_predicate = [ '( pi[0][0] < epsilon2- )',   '( pi[0][1] > epsilon3- )'  ]
#learning_stl = True
# par_bounds = [   [0,100],  #for epsilon0
#                 [100,300], #for epsilon1
#                 [0.1, 50], #for epsilon2
#                 [0.1, 50]] #for epsilon3



'''EXAMPLE: HyperSTL grammar for autonomous valet parking case study'''

## 2 quantifiers
# grammar_quantifiers = '''
# psi0 == forall exists phi0 | exists  forall phi0 | forall forall phi0 | exists exists phi0
# '''

# grammar_structure = '''
# phi0 = always phi0 | eventually phi0 | phi0 until phi0 | phi0 and phi0 | phi0 or phi0 | phi0 implies phi0 | not phi0 | predicate0
# '''

# grammar_predicate = [  
# #       Velocity
#   '( abs(pi[0][0] - pi[1][0] ) < epsilon0- )',
# #    Distance to the pedestrian
#   '( abs(pi[0][1] - pi[1][1] ) < epsilon1- )']

#learning_stl = True 
#par_bounds = [[0.1,50], [0.1,50]]


