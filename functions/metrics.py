from typing import List
from scipy.optimize import minimize as scipy_minimize
# from pymoo.core.problem import ElementwiseProblem
# from pymoo.optimize import minimize as pymoo_minimize
# from pymoo.visualization.scatter import Scatter
# from pymoo.algorithms.moo.nsga2 import NSGA2
import numpy as np
import time
import copy

import functions.syntax_eval as fun_eval
import functions.monotonicity_functions as fun_mono
import functions.create_grammar as gramm    

def fun_temp(inputs, description, temperature = None):

    '''The function is used to implement the temperature in the Simulated Annealing algorithm.
        Depending on the description, the function returns:
        - the initial temperature (description = 'init') - temperature in input is None
        - the decreased temperature starting from the temperature in input (description = 'update')
        - the number of iterations to do with the same temperature given in input (description = 'numb_it')
    '''
    
    if inputs.kind_score in ['fitness', 'stl_robustness', 'efficient_monitoring', 'efficient_monitoring_stl']:
        if description == 'init': return None # temperature = None
        elif description == 'update': return None # temperature = None
        elif description == 'numb_it': return 1 # numb_it_same_temperate_to_do = 1

    elif inputs.kind_score in ['obj_function']:
        if description == 'init': # initialize temperature
            temperature = 10000
            # print('initial temperature = ', temperature)
            return temperature

        elif description == 'update': # decrease temperature
            alpha = 0.8
            temperature = alpha * temperature
            # print('new temperature = ', temperature)
            return temperature

        elif description == 'numb_it': # number of iterations to do with the same temperature
            
            beta = 100000
            numb_it = min ( 100 , beta / temperature )
            # print('numb_it = ', numb_it)

            return numb_it

def compute_robustness_set(traces, formula_string):
            return list(map(lambda pi: fun_eval.evaluate_rob_quantitative([pi], formula_string), traces))
def compute_obj_fun(traces_pos: List[List[List[float]]], traces_neg, formula_string, verbose = -1, mode = 'cost'):
    """
    Compute the objective function value for a given set of traces and formula string.

    Parameters:
    traces (list): List of traces.
    formula_string (str): Formula string to evaluate.

    mode = 
     - cost function
     - msc
     - ratio_fp_fn

    Returns:
    float: The computed objective function value.
    """
    # print(traces_neg is not None and len(traces_neg) > 0)
    # Negative examples are available
    if traces_neg is not None and len(traces_neg) > 0:

        # all_rob_pos = []
        # # Compute the robustness of the formula on each positive trace
        # for pi in traces_pos:
        #     rho = fun_eval.evaluate_rob_quantitative( [pi], formula_string)
        #     all_rob_pos.append(rho)

        # formula_string1 = formula_string
        # formula_string2 = formula_string
        # formula_string3 = formula_string
        

        # with concurrent.futures.ProcessPoolExecutor() as executor:

        #     # Submit tasks to run in parallel
        #     future_rob_pos1 = executor.submit(compute_robustness_set, traces_pos[:len(traces_pos)//2], formula_string1)
        #     future_rob_pos2 = executor.submit(compute_robustness_set, traces_pos[len(traces_pos)//2:], formula_string2)
        #     future_rob_neg = executor.submit(compute_robustness_set, traces_neg, formula_string3)
            
        #     # Wait for both tasks to complete and get results
        #     all_rob_pos = future_rob_pos1.result()
        #     all_rob_pos += future_rob_pos2.result()
        #     all_rob_neg = future_rob_neg.result()


        # Using map to apply evaluate_rob_quantitative to each trace in traces_pos
        all_rob_pos = list(map(lambda pi: fun_eval.evaluate_rob_quantitative([pi], formula_string), traces_pos))

        # Goal: enforce SATISFACTION OF POSITIVE TRACES
        # Select only the negative robustness values
        negative_values = [abs(x) for x in all_rob_pos if x < 0]
        # print('negative_values:', negative_values)

        average_negative_values = [0 if len(negative_values) == 0 else sum(negative_values) / len(negative_values)][0]
    
        # Goal: enforce VIOLATION OF NEGATIVE TRACES
        # Compute the robustness of the formula on each negative trace
        # all_rob_neg = []
        # for ni in traces_neg:
        #     rho = fun_eval.evaluate_rob_quantitative( [ni], formula_string)
        #     all_rob_neg.append(rho)

        all_rob_neg = list(map(lambda pi: fun_eval.evaluate_rob_quantitative([pi], formula_string), traces_neg))

        # Select only the positive robustness values
        positive_values = [x for x in all_rob_neg if x >= 0]
        # print('positive_values:', positive_values)

        average_positive_values = [0 if len(positive_values) == 0 else sum(positive_values) / len(positive_values)][0]

        # Goal: Include ratio/percentage of misclassified traces
        ratio_fp = len(negative_values)/len(traces_pos)
        ratio_fn = len(positive_values)/len(traces_neg)
        
        if verbose > 0: print(f'Sat: {average_negative_values}  ,Viol: {average_positive_values},  FP: {ratio_fp}, FN: {ratio_fn}')
        
        # Double the value of msc compared to robustness terms +
        # + Penalization if ratio_fp > 0.7 
        # + Penalization if ratio_fn > 0.7

        cost = average_negative_values + average_positive_values
        cost =  (0.5 * cost) +  ratio_fp + ratio_fn 
        # print('average_negative_values:', average_negative_values, 'average_positive_values:', average_positive_values, 'ratio_fp:', ratio_fp, 'ratio_fn:', ratio_fn)
        if ratio_fn > 0.7: cost +=2 
        if ratio_fp > 0.7: cost +=2

        print('cost:', cost, formula_string)    

        if mode == 'cost':    
            return cost

        elif mode == 'msc' : 
            msc = (len(negative_values) + len(positive_values))/(len(traces_pos) + len(traces_neg))
            return cost, msc, ratio_fp, ratio_fn
        else:
            print('Mode not recognized but returning cost')
            return cost        

    # Positive examples only
    else :  

        all_rob_pos = list(map(lambda pi: fun_eval.evaluate_rob_quantitative([pi], formula_string), traces_pos))

        # Goal: enforce SATISFACTION OF POSITIVE TRACES
        # Select only the negative robustness values
        negative_values = [abs(x) for x in all_rob_pos if x < 0]

        average_negative_values = [0 if len(negative_values) == 0 else sum(negative_values) / len(negative_values)][0]

        # Goal: enforce TIGHT SATISFACTION ON POSITIVE TRACES
        # Select only the positive robustness values
        positive_values_tight = [x for x in all_rob_pos if x >= 0]
        # Compute the average of the positive robustness values
        average_positive_values_tight = [0 if len(positive_values_tight) == 0 else sum(positive_values_tight) / len(positive_values_tight)][0]
        # Cost aiming at minimizing the absolute value of the POS robustness - TIGHTNESS - 
        cost = average_negative_values + abs(average_positive_values_tight)

        # Print the values of different terms ,
        if verbose > 0: print('Sat:', average_negative_values,  ' Tight:', abs(average_positive_values_tight))
        # Just FP
        if mode == 'cost':
            return cost
        elif mode == 'msc' : 
            ratio_fp = len(negative_values)/len(traces_pos)
            print('No negative traces; cost is the ratio of false positives.')
            msc = ratio_fp

            return cost , msc, ratio_fp, 0
    

def function_to_optimize(shuffled_x, traces_pos, traces_neg, formula, count, order):
    ''' The function to optimize is the objective function. 
    This version takes as input the parameters to be embedded in the formula.'''
    
    global best_solution, best_value
    
    # Replace the parameters in the formula
    formula_string = formula.string_param

    # order tells me how componets of x have been shuffled to obtain shuffled_x. Now I want to use the original order
    x = [shuffled_x[list(order).index(j)] for j in range(len(shuffled_x))] 
    # Values of x are sorted but some parameters may be missing
    # x = (value_epsilon1, value_epsilon4, value_epsilon5)
    # # For that, use the index auxiliary to keep track of it
    index_aux = 0
    for index_epsilon in range(count.max_number_par_symbols):
        if f'epsilon{index_epsilon}-' in formula_string:
            formula_string = formula_string.replace(f'epsilon{index_epsilon}-', f'{x[index_aux]}')
            index_aux += 1
    # print('Calling function to optimize:', formula_string)
    # Compute the objective function value
    current_value = compute_obj_fun(traces_pos, traces_neg, formula_string)

    # Keep track of the best value and parameters seen so far
    if current_value < best_value:
        best_value = current_value
        best_solution = x.copy()

    return current_value



def instantiate_parameters(inputs, formula, count, traces_pos, traces_neg, tim, dict_par_inst, max_fun_dir = 8, xtol = 0.1, ftol = 0.1):
    
    '''dict_par_inst is a dictionary that contains formulas whose parameters have been instantiated'''

    np.random.seed(inputs.seed)

    time_start_refinement_parameters = time.time()
    new_formula = copy.deepcopy(formula) 

    # Instantiatewith the most likely values
    if inputs.kind_score in ['fitness', 'stl_robustness', 'efficient_monitoring', 'efficient_monitoring_stl']:
        if inputs.learning_stl: new_formula.string = fun_mono.replace_parameters_most_likely_satisfied(new_formula.string_param, inputs.par_bounds, new_formula.mono, count.max_number_par_symbols)

    # Instantiate with the values found by the optimizer
    elif inputs.kind_score in ['obj_function']:
        formula_in_dict = None
        if formula.string_param in dict_par_inst: formula_in_dict = formula.string_param
        else: 
            # print('Call enumerates equivalent formulas')
            equivalent_formulas = gramm.enumerates_equivalent_formula_strings(formula.nodes)
            # print('Equivalent formulas:', equivalent_formulas)
            for eq_formula in equivalent_formulas:
                if eq_formula in dict_par_inst:
                    formula_in_dict = eq_formula
                    break
        
        # If never seen before --> sample initial point and compute bounds
        if formula_in_dict is None:
            # print('Never seen before')
            x0 = [] #initial guess
            bounds = [] #initialize bounds
            index_aux = 0
            for index_epsilon in range(count.max_number_par_symbols):
                if f'epsilon{index_epsilon}-' in new_formula.string_param:
                    bounds.append( tuple( inputs.par_bounds[index_epsilon]))
                    x0.append(np.random.uniform(inputs.par_bounds[index_epsilon][0], inputs.par_bounds[index_epsilon][1]))
                    index_aux += 1
        # If seen before but with different iteration or tolerance --> start with the best solution seen so far and compute bounds
        elif dict_par_inst[formula_in_dict]['iteration'] != count.formula_iteration or dict_par_inst[formula_in_dict]['tol'] != ftol:
            # print('Seen before but with different iteration or tolerance')
             # Start with the best solution seen so far
            x0 = dict_par_inst[formula_in_dict]['parameters'].copy()
            # Compute bounds
            bounds = []
            index_aux = 0
            for index_epsilon in range(count.max_number_par_symbols):
                    if f'epsilon{index_epsilon}-' in new_formula.string_param:
                        bounds.append( tuple( inputs.par_bounds[index_epsilon]))
                        index_aux += 1

        if  formula_in_dict is None or dict_par_inst[formula_in_dict]['iteration'] != count.formula_iteration or dict_par_inst[formula_in_dict]['tol'] != ftol: 

            #Prepare for optimization
            global best_solution, best_value
            best_solution = None
            best_value = float('inf')
            # Shuffle the initial guess
            order = np.arange(len(x0))
            np.random.shuffle(order)
            x0 = [x0[i] for i in order]
            # Shuffle the bounds (remain a tuple)
            shuffled_bounds = tuple([bounds[i] for i in order])
            # Optimization
            max_fun = max_fun_dir * len(x0) #(it takes 8 fun evaluations to explore one direction with xtol = 0.1)
            options = {'maxfev': max_fun, 'xtol': xtol, 'ftol': ftol, 'disp': True }
            # print('About to optimize:', new_formula.string_param)
            res = scipy_minimize(function_to_optimize, x0 = x0, args = (traces_pos, traces_neg, new_formula, count, order),\
                                method='Powell', bounds = shuffled_bounds, options=options) #L-BFGS-B
            
            # Shuffle again the parameters to get the original order
            par = [res.x[list(order).index(j)] for j in range(len(res.x))] 
            # Update with the best solution seen in the process
            if res.fun > best_value: par = best_solution.copy()
            print('Ended optimization:')
            # Update the dictionary
            dict_par_inst[formula.string_param] = {'parameters': par , 'tol': ftol, 'iteration' : count.formula_iteration}

        # If seen before with the same iteration and tolerance --> use the parameters found before
        else:
            print('\n\nAlready seen before with the same iteration and tolerance')
            par = dict_par_inst[formula.string_param]['parameters'].copy()
                
        # Replace the parameters in the formula
        new_formula.string = new_formula.string_param
        index_aux = 0
        for index_epsilon in range(count.max_number_par_symbols):
            if f'epsilon{index_epsilon}-' in new_formula.string:
                new_formula.string = new_formula.string.replace(f'epsilon{index_epsilon}-', f'{par[index_aux]}')
                index_aux += 1

    tim.refinement_parameters += (time.time() - time_start_refinement_parameters )

    return new_formula, tim , dict_par_inst
                 

## For multiobjective optimization
# def instantiate_parameters_multiobje
# ctive(inputs, formula, count, traces_pos, traces_neg, tim, tol = 0.01):
    
#     time_start_refinement_parameters = time.time()
#     new_formula = copy.deepcopy(formula) 

#     # Instantiatewith the most likely values
#     if inputs.kind_score in ['fitness', 'stl_robustness', 'efficient_monitoring', 'efficient_monitoring_stl']:
#         if inputs.learning_stl: new_formula.string = fun_mono.replace_parameters_most_likely_satisfied(new_formula.string_param, inputs.par_bounds, new_formula.mono, count.max_number_par_symbols)

#     # Instantiate with the values found by the optimizer
#     elif inputs.kind_score in ['obj_function']:
        
#         n_obj = 2
#         n_ieq_constr = 0
#         xl= []
#         xu = []
#         index_aux = 0
#         for index_epsilon in range(count.max_number_par_symbols):
#             if f'epsilon{index_epsilon}-' in new_formula.string_param:
#                 xl.append(inputs.par_bounds[index_epsilon][0])
#                 xu.append(inputs.par_bounds[index_epsilon][1])
#                 index_aux += 1

#         n_var = len(xl)
#         xl = np.array(xl)
#         xu = np.array(xu)
#         print('Lower bounds in [0,1] ?:', xl)  
#         print('Upper bounds in [0,1] ?:', xu)

#         problem = MyProblem(n_var, n_obj, n_ieq_constr, xl, xu, formula = new_formula, traces_pos = traces_pos, traces_neg = traces_neg, count = count)
#         print('Formula to optimize:', new_formula.string_param)
#         # Optimizer
#         res = pymoo_minimize(problem, algorithm = NSGA2(pop_seize = 10), termination=('n_gen', 3), verbose=-1)
#         plot = Scatter()
#         plot.add(res.F, edgecolor="red", facecolor="none")
#         plot.show()
#         X = res.X
#         optimal_values = X[0,:]
#         print('\n\nOptimal values ',  optimal_values)
#         # Replace the parameters in the formula
#         new_formula.string = new_formula.string_param
#         index_aux = 0
#         for index_epsilon in range(count.max_number_par_symbols):
#             if f'epsilon{index_epsilon}-' in new_formula.string:
#                 concrete_value = optimal_values[index_aux] # optimal values
#                 new_formula.string = new_formula.string.replace(f'epsilon{index_epsilon}-', f'{concrete_value}')
#                 index_aux += 1
        
#     tim.refinement_parameters += (time.time() - time_start_refinement_parameters )
#     return new_formula, tim


# def compute_rfp(x , formula , traces_pos, count ):

#     '''Compute the ratio of false positivesby replacing concrete values of the parameters (in x)
#       in the formula parametric string (epsilon{i}-)'''

#     # Replace the parameters in the formula
#     formula_string = formula.string_param

#     # Values of x are sorted but some parameters may be missing
#     # x = (value_epsilon1, value_epsilon4, value_epsilon5)
#     # For that, use the index auxiliary to keep track of it
#     index_aux = 0
#     for index_epsilon in range(count.max_number_par_symbols):
#         if f'epsilon{index_epsilon}-' in formula_string:
#             concrete_value = x[index_aux]
#             formula_string = formula_string.replace(f'epsilon{index_epsilon}-', f'{concrete_value}')
#             index_aux += 1

#     nb_fp = 0
#     for pi in traces_pos:
#         rho = fun_eval.evaluate_rob_quantitative( [pi], formula_string)
#         if rho < 0: nb_fp += 1
    
#     return    nb_fp/len(traces_pos)


# def compute_rfn(x , formula , traces_neg, count ):
#     '''Compute the ratio of false negatives by replacing concrete values of the parameters (in x)
#       in the formula parametric string (epsilon{i}-)'''

#     # Replace the parameters in the formula
#     formula_string = formula.string_param

#     # Values of x are sorted but some parameters may be missing
#     # x = (value_epsilon1, value_epsilon4, value_epsilon5)
#     # For that, use the index auxiliary to keep track of it
#     index_aux = 0
#     for index_epsilon in range(count.max_number_par_symbols):
#         if f'epsilon{index_epsilon}-' in formula_string:
#             concrete_value = x[index_aux]
#             formula_string = formula_string.replace(f'epsilon{index_epsilon}-', f'{concrete_value}')
#             index_aux += 1

#     nb_fn = 0
#     for pi in traces_neg:
#         rho = fun_eval.evaluate_rob_quantitative( [pi], formula_string)
#         if rho >= 0: nb_fn += 1
    
#     return    nb_fn/len(traces_neg)

# class MyProblem(ElementwiseProblem):

#     def __init__(self, n_var, n_obj, n_ieq_constr, xl, xu, **kwargs):
#         super().__init__(n_var=n_var,
#                          n_obj=n_obj,
#                          n_ieq_constr=n_ieq_constr,
#                          xl=xl,
#                          xu=xu )
#         self.kwargs = kwargs
        
#     def _evaluate(self, x, out, *args, **kwargs):
#         formula = self.kwargs['formula']
#         traces_pos = self.kwargs['traces_pos']
#         traces_neg = self.kwargs['traces_neg']
#         count = self.kwargs['count']
#         f1 = compute_rfp(x , formula , traces_pos, count )
#         f2 = compute_rfn(x , formula , traces_neg, count )

#         out["F"] = [f1, f2]

    
                 