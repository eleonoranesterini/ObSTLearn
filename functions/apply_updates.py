import numpy as np
import random
import copy

import functions.create_grammar as gramm

class StudyComputationalTime:
    
    def __init__(self):
    
        self.enumerating_formulas = 0 #Tme spent enumerating the formulas
        self.monitor = 0 # time spent in the monitorin
        self.refinement_parameters = 0 # For stl formulas: refinement parameters 
        self.gen_formula = 0 # Time for generating a valid starting formula (rejecton sampling loop once length and m are fixed - candidates already enumerated
        self.apply_changes = 0 # Time dedicated to applying changes (no monitor of the new formula)
        self.trace_generation = 0 # Time to generate new traces in the active setting
        self.read_traces = 0 # Time to read the traces from the file
        
        #Previous iterations time (to study the current one alone)
        self.enumerating_formulas_prev = 0 
        self.monitor_prev = 0
        self.refinement_parameters_prev = 0 
        self.gen_formula_prev = 0 
        self.apply_changes_prev = 0 
      

class Counter:
    
    def __init__(self ):
        
        self.max_number_par_symbols = 15 #Maximum number of parameter symbols in the grammar predicate
        self.already_stored = 0 # Used to count the number of times a formula is rejected because alreday stored in the set of formulas
        self.formula_iteration = 0 # Counts the number of formulas learned/stored so far
        
        #Counter on iterations
        self.it_changes_max = 50#200 # max number of changes
        
        #self.it_mutation = 0 #counts number of iterations in which changes are applied

        #Counter on number of random mutations
        self.mutation = 0 #number of times the formula has been mutated from its starting formula before hitting a satisfied formula
        
        #!!! One value for the whole set of data (no one value for each formula)
        self.mutation_tot = 0 #total number of mutations for the whole set of learned formulas
        self.mutation_succ = 0 #total number of mutations that produces a satisfied formula in total for the whole set of learned formulas
        
        #Counter on number of rejection sampling
        self.numb_rej_max = 50 # max number of rejection sampling to form a valid starting formula
        self.num_rej = 0 #counts number of rejections.
        
        self.nb_iterations_outer = 5 # times nb_target_formula = number of Iterations (starting with different sampled formulas)
        self.iter_outer_loop = 0 #counts the nb of outer iteration
        
        #Counter on monitoring 
        self.hyper_monitor_tot = 0 #number of hyperformulas monitored before hitting a satisfied formula - taking into account also the monitors for replaced starting formulas
        self.hyper_monitor  = 0 #number of hyperformulas starting from its starting formula monitored before hitting a satisfied formula
        
        #Counter on syntactic rules
        self.strengthened_same_formula = 0 # number of times the SPECIFIC formula tried have been changed to a stricter one
        self.strengthened_formulas_succ = 0 #number of times the formulas tried to be changed to a stricter one AND IT WAS SUCCESSFULL (satisfying stopping condition)
        self.looser_same_formula = 0  # Total number of times the SPECIFIC formula have been changed to be less strict
        self.looser_formula_succ = 0 # Total number of times the formulas have been changed to be less strict AND IT WAS SUCCESSFULL (satisfying stopping condition)
 

def update_time_after_success(tim):

    # Print and update tim    
    # print(f'time_enumerating_formulas={tim.enumerating_formulas - tim.enumerating_formulas_prev}')
    # print(f'time_monitor={tim.monitor - tim.monitor_prev}')
    # print(f'time_refinement_parameters={tim.refinement_parameters - tim.refinement_parameters_prev}')
    # print(f'time_gen_formula={tim.gen_formula- tim.gen_formula_prev}')
    # print(f'time_apply_changes={tim.apply_changes - tim.apply_changes_prev}')
    
    tim.enumerating_formulas_prev = tim.enumerating_formulas
    tim.monitor_prev = tim.monitor
    tim.refinement_parameters_prev = tim.refinement_parameters
    tim.gen_formula_prev = tim.gen_formula
    tim.apply_changes_prev = tim.apply_changes

    return tim

def reset_counter_after_new_starting_formula(count):
    
    #Reset counters for the new starting formula
    #count.it_mutation = 0
    count.looser_same_formula = 0
    count.strengthened_same_formula = 0

    count.mutation = 0 #number of times the formula has been mutated
    count.hyper_monitor  = 0 #number of hyperformulas starting from the same valid formula monitored before hitting a satisfied formula
    
    return count

def update_count_after_success(count):

    #Print and update number of mutations
    
    count.hyper_monitor_tot = 0 
    count.already_stored = 0
    count.formula_iteration += 1

    return count

def write_final_metrics(filename, tim, count):

    with open(f'{filename}','a') as file:

        #Write time-related infos
        file.write(f'\ntime_enumerating_formulas={tim.enumerating_formulas}')
        file.write(f'\ntime_monitor={tim.monitor}')
        file.write(f'\ntime_refinement_parameters={tim.refinement_parameters}')
        file.write(f'time_trace_generation={tim.trace_generation}')
        file.write(f'\ntime_gen_formula={tim.gen_formula}')
        file.write(f'\ntime_apply_changes={tim.apply_changes}\n\n')
        file.write(f'time_read_traces={tim.read_traces}\n\n')
        
        #Write the total number of attempts of applying syntactic rules vs the successful ones
        file.write(f'\n#strengthened_formulas_succ = {count.strengthened_formulas_succ}')
        file.write(f'\n#strengthened_formulas = {count.strengthened_same_formula}\n')
        file.write(f'\n#looser_succ = {count.looser_formula_succ}')
        file.write(f'\n#looser_formula = {count.looser_same_formula}\n')
        
        #Write the total number of mutations
        file.write(f'\n#mutations_succ = {count.mutation_succ}')
        file.write(f'\n#mutations_tot = {count.mutation_tot}\n\n\n')

    return


def update_candidate_formula(inputs, formula, new_formula, temperature):
    """The fuction may update the current formula with the new one according to the acceptance ratio.
    For first version of the algorithm, we use Metropolis-Hastings acceptance ratio, wanting to increase the value of the cost function.
    For the second version of the algorithm, we use Simulated Annealing, wanting to minimize the value of the cost function.
    """
    
    if inputs.kind_score in ['fitness', 'stl_robustness', 'efficient_monitoring', 'efficient_monitoring_stl']:
        #Metropolis-hastings acceptance ratio --> compute the score from the cost function
        
        if new_formula.cost != 'runout':

            # Always accept new formulas that improve the current one
            if new_formula.cost > formula.cost: acceptance_ratio = 1
            # When we do not have quantitative measure, we update or reject the mutation with probability 0.5
            elif new_formula.cost == formula.cost: acceptance_ratio = 0.5

            #Here the cost is negative, so we want to maximize it -> we want to go to positive values
            else: acceptance_ratio = min(1, np.exp( (new_formula.cost - formula.cost) * 0.5))#this cost makes sense when current cost is negative
                    
            # Update the current formula with probability given by acceptance_ratio
            if random.uniform(0,1) <= acceptance_ratio:
                formula = gramm.Formula(new_formula.quantifiers, new_formula.quantifier_levels, new_formula.string_param, new_formula.nodes, new_formula.length, cost = new_formula.cost, string = new_formula.string )

    elif inputs.kind_score == 'obj_function':

        ## Apply Simulated Annealing   

        # Always accept new formulas that improves the current one ( mininimize the objective function )
        if new_formula.cost < formula.cost:  acceptance_ratio = 1
        # Accept the new formula with a certain probability
        else: 
            acceptance_ratio = np.exp( - (new_formula.cost - formula.cost)/ temperature ) 
            # Update the current formula with probability given by acceptance_ratio
            if random.uniform(0,1) < acceptance_ratio:
                formula = copy.deepcopy(new_formula)
    return formula


def update_best_formula(kind_score, formula_best, new_formula):

    """
    The fuction updates the best formula with the new one if and only if the new formula improves cost of the best found so far (if it is new).
    For first version of the algorithm, we want to increase the value of the cost function.
    For the second version of the algorithm, we want to minimize the value of the cost function.
    """
    
    if kind_score in ['fitness', 'stl_robustness', 'efficient_monitoring', 'efficient_monitoring_stl']:
        # Update to new formulas only when they improve the current one
        if new_formula.cost != 'runout' and new_formula.cost > formula_best.cost: # in this case, the cost is negative, so we want to increase it
                formula_best = copy.deepcopy(new_formula)

    elif kind_score == 'obj_function':

        if new_formula.cost < formula_best.cost:  
            formula_best = copy.deepcopy(new_formula)

    return formula_best


