import numpy as np
import time
import os
import sys
import random


sys.path.append(sys.path[0] + '/../../')
import functions.inputs as inputs
import syn_guided_obj_fun as syntax



'''               

(supervisor)
[0]  'time'
[1]  'ego_x'
[2]  'ego_y' 
[3]  'yaw' 
[4]  'cone0_x'   [5] 'cone0_y'   [6] 'cone0_min_dist'
[7]  'cone1_x'   [8] 'cone1_y'   [9]'cone1_min_dist' 
[10] 'cone2_x',   [11]'cone2_y'   [12]'cone2_min_dist' 
[13] 'speed_car'
[14] min distance from cones

(sensors on the ego car)
[15] steering_angle
[16] target_speed
[17] current_speed                  
[18] estimated_dist_cones

 
[19] number bounding boxes
[20] size of biggest bounding box
[21] size of second biggest bounding box
[22] confidence of biggest bounding box

[23] number of pixels of the color of the traffic cones
[24] number of orange agglomerates  
[25] size of biggest orange agglomerate
[26] size of second biggest orange agglomerate

[27] derivative of the size of the biggest bounding box in time 10
[28] derivative of the size of biggest orange agglomerate in time 10
     
(features - supervisor)
[29] distance of the color of the broken car from the color orange in norm 1
[30] hue of the color of the broken car
[31] saturation of the color of the broken car
[32] value of the color of the broken car
'''              



#Definition of the grammar for the quantifiers
grammar_quantifiers = '''
psi0 == forall psi0 | phi0
'''


grammar_structure = '''
phi0 ==  always[epsilon10-:end_trace] phi0 | always phi0 | eventually phi0 | eventually[epsilon11-:end_trace] phi0 | phi0 and phi0 | phi0 or phi0 | not phi0 | predicate0 
'''

grammar_predicate = [[   "pi[0][15] < epsilon0-" ,
                         "pi[0][19] < epsilon1-" ,
                        " pi[0][20] < epsilon2-" ,
                        " pi[0][21] < epsilon3-" ,
                        " pi[0][23] < epsilon4-" ,
                        " pi[0][24] < epsilon5-" ,
                        " pi[0][25] < epsilon6-", 
                        " pi[0][27] < epsilon7-" ,
                        " pi[0][28] < epsilon8-" , 
                        " pi[0][17] < epsilon9-" 
                      ]
                      ]

case_study = 'traffic_cones'
par_bounds = [[0,1]] *  len(grammar_predicate[0]) + [[50,200]] + [[50,200]]
name_folder_data = 'data/traffic_cones/final_datasets/training_main'
horizon = 100
name_text_file_formulas = 'results.text'
numb_simulation = 200
numb_formulas = 10  # number of formulas to be learned

seeds = [731454, 3599, 240612, 953367, 223171, 497659, 50403, 534437, 120668, 894631]


for iteration in range(1, 11):
    seed_learning = seeds[iteration - 1] #random.randint(0, 1000000)


    
    name_formulas_folder = f'res_TC_ensemble_{iteration}'
    # Output files
    if not(os.path.exists(name_formulas_folder)): os.mkdir(name_formulas_folder)
    output_file_name = f'{name_formulas_folder}/{name_text_file_formulas}'
    if os.path.exists(output_file_name): os.remove(output_file_name)

    
    with open(f'{output_file_name}','a') as file:
        file.write('List of seeds used:\n')
        file.write(f'\nseed_learning = {seed_learning}\n')


    start_time_learning = time.time()


    
    input_data = inputs.InputData(
        seed = seed_learning,
        numb_quantifiers=[1,1],  # Exactly 1 quantifier
        grammar_quantifiers=grammar_quantifiers,
        grammar_structure=grammar_structure,
        grammar_predicate=grammar_predicate,
        max_length= 7)
    
    # Additional options:
    input_data.learning_stl = True
    input_data.par_bounds = par_bounds  #[[0,1]] * len(grammar_predicate[0]) 
    # input_data.fraction = fraction
    input_data.kind_score = 'obj_function'
    input_data.to_be_present = None #[ grammar_predicate[0], 1 ]

    input_data.bool_mutation_quantifiers = False  # Quantifiers not modified during the learning process (since one forall is the only option admissible from grammar_predicate)
    input_data.store_enumeration_formulas = False
    input_data.store_formulas = True
    input_data.n_target = numb_formulas
    input_data.name_output_folder = name_formulas_folder
    input_data.horizon = horizon
        
    output_data, input_data = syntax.main(name_folder_data, input_data,  output_file_name = name_text_file_formulas,  numb_simulation=numb_simulation)

    # Compute time 
    end_time_learning = time.time() 
    time_learning = end_time_learning - start_time_learning 
    # Store learning time 
    with open(f'{output_file_name}','a') as file:
        file.write('\n\n\nLearning time (seconds):\n')
        file.write(f'\ntime_learning = {time_learning}')



