import time
import os
import sys
import random

sys.path.append(sys.path[0] + '/../../')
import functions.inputs as inputs
import syn_guided_obj_fun as syntax


'''               
[0]  'time'
[1]  'ego_x'
[2]  'ego_y'
[3]  'ego_heading' 
[4]  'ego_vel_lin' (modulo)
Lead
[5]  'lead_x'
[6]  'lead_y'
[7]  'lead_heading'
[8]  'lead_vel_lin' (modulo)
Supervisor
[9]  'distance_true'
- [10] 'direction' # L, R, S, 1
[11] 'color_dist_from_black' 
[12] 'dist_obst_follower' 
[13] 'dist_obst_lead'
[14] 'ob_pos_x' 
[15] 'ob_pos_y' 

Additional variables: (all unknown are replaced by closest value in the time series) 
[16] 'yellow_line_angle' - lead
[17] 'black_car_angle'  

[18] 'distance_sensors'  
[19] 'num_pixels'
[20] 'deriv_yellow_line_angle' - lead
[21] 'deriv_black_car_angle'  
[22] number of black agglomerates
[23] size of biggest black agglomerate
[24] size of second biggest black agglomerate
[25] derivative of the number of num_pixel_black
'''              
if __name__ == "__main__":
    
    #Definition of the grammar for the quantifiers - constrained
    grammar_quantifiers = '''
    psi0 == forall psi0 | phi0
    # '''



    grammar_structure = '''
    phi0 == always[epsilon10-:end_trace] phi0 | always phi0 | eventually[epsilon11-:end_trace] phi0 | eventually phi0 | phi0 and phi0 | phi0 or phi0 | not phi0 | predicate0
    '''
    
    
    # # Definition of the grammar for predicates
    grammar_predicate = [[  " pi[0][3] < epsilon0-" ,
                            " pi[0][4] < epsilon1-" ,
        
                            " pi[0][17] < epsilon2-" ,
                            " pi[0][18] < epsilon3-" ,
                            " pi[0][19] < epsilon4-" ,
                            
                            " pi[0][21] < epsilon5-" ,
                            " pi[0][22] < epsilon6-" ,
                            " pi[0][23] < epsilon7-" ,
                            " pi[0][24] < epsilon8-" ,
                            " pi[0][25] < epsilon9-"
                            ]]
    par_bounds = [[0,1]] * len(grammar_predicate[0]) + [[50, 600]] + [[50, 600]] + [[50, 600]]
    
    name_folder_data = 'data/lead_follower/final_datasets/training_main'
    case_study = 'lead_follower'
    name_text_file_formulas = 'results.text'
    horizon = 100
    numb_simulation = 200
    numb_formulas = 10 # number of formulas to be learned
    seeds = [112488, 985088, 546601,  896940, 636390, 849677, 457172, 704427, 465296, 541041]# for the 10 ensembles learned for the experiments

    for iteration in range(1, 11):# 10 ensembles of {numb_formulas} monitors

        seed_learning = seeds[iteration-1]#random.randint(0, 1000000) 
        
        start_trace_index = 0
        
        name_formulas_folder = f'res_LF_ensemble_{iteration}'    
        
        # # Output files
        if not(os.path.exists(name_formulas_folder)): os.mkdir(name_formulas_folder)
        output_file_name = f'{name_formulas_folder}/{name_text_file_formulas}'
        if os.path.exists(output_file_name): os.remove(output_file_name)

        with open(f'{output_file_name}','a') as file:
            file.write('List of seeds used:\n')
            file.write(f'seed_learning = {seed_learning}\n')


        input_data = inputs.InputData(
            seed = seed_learning,
            numb_quantifiers=[1,1],  # Exactly 1 quantifier
            grammar_quantifiers=grammar_quantifiers,
            grammar_structure=grammar_structure,
            grammar_predicate=grammar_predicate,
            max_length= 7)
        
        # Additional options:
        input_data.learning_stl = True
        input_data.par_bounds = par_bounds  
        input_data.kind_score = 'obj_function'
        input_data.to_be_present = None #[ grammar_predicate[0], 1 ]

        input_data.bool_mutation_quantifiers = False  # Quantifiers not modified during the learning process (since one forall is the only option admissible from grammar_predicate)
        input_data.store_enumeration_formulas = False
        input_data.store_formulas = True
        input_data.n_target = numb_formulas
        input_data.name_output_folder = name_formulas_folder
        input_data.horizon = horizon


        start_time_learning = time.time()    
        output_data, input_data = syntax.main(name_folder_data , input_data,  output_file_name = name_text_file_formulas,  numb_simulation = numb_simulation )

        # Compute time 
        end_time_learning = time.time() 
        time_learning = end_time_learning - start_time_learning 
        # Store learning time 
        with open(f'{output_file_name}','a') as file:
            file.write('\n\nTotal learning time (seconds):\n')
            file.write(f'\ntime_learning = {time_learning}')

