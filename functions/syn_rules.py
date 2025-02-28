import random
import copy

import functions.syntax_eval as fun_eval
import functions.create_grammar as gramm
    
def looser(formula, grammar, inputs ):
    
    '''Render the current formula looser; it outputs a new formula only if it is consistent with the grammar. 
    No checks on the satisfaction are carried out here.'''
    
    
    ##Indices in the tabu list and can_be_changed list 
    # 0 represents (forall to exists)
    # 1 represents (always to eventually)
    # 2 represents (and to or)
    # 3 represents (<-> to implies)
    # 4 represents (and to implies)
    
    if True: #inputs.length is not None or formula.length >= inputs.max_length:
        ## rules not changing the lenght of the formula
        rules = [ ['A','E'], ['always',['eventually']] , ['and',['or']] , ['<->',['implies']], ['and', ['implies']]]
    else:
        # rules also changing the lenght of the formula
        rules = [ ['A','E'], ['always',['eventually']] , ['and',['or']] , ['<->',['implies']], ['and', ['implies']]\
                 , ['always' , ['always', 'eventually']], ['always', ['eventually', 'always']]]
    number_of_rules = len(rules)
    
    
    # Tabu_moves is a list of lists. 
    # Each list is related to one specific change (see list above).
    # The list contain the indices of the position of the operators (e.g. always for rule 0) for which the transformation (to eventually) is not allowed anymore.
    tabu_moves = [[]] * number_of_rules # Empty tabu list
    
    
    while True: 
        
        ## Check which rule is applicable and assign equal probability to the corresponding weigths
        can_be_changed = [False] * number_of_rules
        weights = [ 0 ] * number_of_rules
        
        ## !!!
        ## If rule 0 is applicable: 
        if  inputs.bool_mutation_quantifiers and sum(formula.quantifiers) > len(tabu_moves[0]): #  #'1' in quantifiers > # tabu forall
            # can_be_changed[0] = True
            #Fixed probability for quantifiers
            # weights[0] = 10 # 1/10 of probability for the rule on the quantifiers
            
            #For the first experiments we do not allow for rules to the quantifiers
            can_be_changed[0] = False
            
        for i in range(1, number_of_rules):
            # If rule i is applicable: 
            if formula.string.count(rules[i][0]) > len(tabu_moves[i]): can_be_changed[i] = True
        
        #If none of the syntactic rules can be applied
        if sum(can_be_changed) == 0 : return False, formula
        
        # Equal probability to the allowed formulas
        value_eq_prob = (100 - weights[0] )/ sum(can_be_changed)
        
        for i in range(number_of_rules):
            if can_be_changed[i]: weights[i] = value_eq_prob
        
        #Sample which rule to apply
        case = random.choices(list(range(number_of_rules)), weights = weights, k = 1)[0]
        aux = 0
        
        if case == 0:  ##  Forall to Exists
            #Update quantifiers
            new_quantifiers = copy.deepcopy(formula.quantifiers)
            #Randomly select which forall to replace
            indx = random.sample(range(sum(formula.quantifiers)) ,1)[0] # sample which forall (1st, 2nd,..)
            while True:
                if indx in tabu_moves[case]: indx = (indx +1) % (sum(formula.quantifiers) ) #change forall if the selected one is tabu
                else: break # stop if the sampled quantifier is not tabu
            for el_quant in range(len(new_quantifiers)):
                if new_quantifiers[el_quant] == 1:
                    if aux == indx: 
                        new_quantifiers[el_quant] = 0 #update to exists
                        break
                    else: aux += 1
                    
            # Update quantifiers levels       
            new_quantifier_levels = copy.deepcopy(formula.quantifier_levels)
            numb_fa = 0 #number of forall encountered so far (while looking for the indx_th one)
            
            for ind_ql in range(1, len(new_quantifier_levels)):
                # If number of forall is greater than the position of the selected forall -> apply changes and exit
                if numb_fa + new_quantifier_levels[ind_ql][0].count('A') >= el_quant:
                    #Replace the A in new_quantifier_levels[ind_ql][0] appearing in position (indx - numb_fa)
                    for el in range(1,len(new_quantifier_levels[ind_ql][0])):
                        if new_quantifier_levels[ind_ql][0][:el].count('A') + numb_fa == el_quant:  
                            new_quantifier_levels[ind_ql][0] = new_quantifier_levels[ind_ql][0][:el-1] + new_quantifier_levels[ind_ql][0][el-1].replace('A', 'E') +new_quantifier_levels[ind_ql][0][el:]
                            break
                    break
                # Else , update the number of forall and consider the following item
                numb_fa += new_quantifier_levels[ind_ql][0].count('A')        
            
            new_formula = gramm.Formula(new_quantifiers, new_quantifier_levels, formula.string_param, formula.nodes, formula.length ,string= formula.string)
        else: # rule 1, 2,3,4
        
            new_nodes = copy.deepcopy(formula.nodes)
            
            #Randomly select which 'starting' operator to replace
            indx = random.sample(range(formula.string.count(rules[case][0])) ,1)[0] # sample which always (1st, 2nd,..)
            while True:
                if indx in tabu_moves[case]: indx = (indx +1) % (formula.string.count(rules[case][0]) ) #change always if the selected one is tabu
                else: break # stop if the sampled always is not tabu
            for index_node, node in enumerate(new_nodes):
                if rules[case][0] in node.data:
                    if aux == indx: 
                        node.data = rules[case][1][0] #update
                        if case in [5,6]:
                            # Add new node in the list
                            new_nodes = gramm.insert_leftchild_unary_node_in_list_nodes(new_nodes, rules[case][1][1], index_node + 1)
                        break
                    else: aux += 1
            
            new_formula = gramm.Formula(formula.quantifiers, formula.quantifier_levels, gramm.tree_to_formula(new_nodes), new_nodes, len(new_nodes), string = gramm.tree_to_formula(new_nodes))
        
        # The new formula obeys to the rules of the grammar
        if gramm.check_if_in_gramm(new_formula.nodes, grammar,inputs): 
            # print('ok looser', new_formula.string_param)
            return True, new_formula # type: ignore
        # The new formula DOES NOT obey to the rules of the grammar
        else: 
            #Update the tabu list
            # print('not ok looser', new_formula.string_param)
            aux_append = tabu_moves[case].copy()
            aux_append.append(indx)
            tabu_moves[case] =  aux_append.copy()
    

def strengthen(formula, grammar , inputs, bool_pos_ex_only = False , monitor = None, count = None, traces_pos = None, set_database= None ):
    
    '''Render the current formula stricter; it outputs a new formula only if it is consistent with the grammar and if the new formula is new and also satisfied.
    
    bool_pos_ex_only    indicates whether we are learning from pos examples only. 
                        If True, the function, before returning a modified formula, checks if the new formula is the grammar, if its new cost is positive 
                                               and that the new formula is not already in the set of learned formulas.
                                               It uses nodes_param because nodes would not be parametric (the function is designed to be used 
                                               when a valid satisfied formula is areaady available and we want to see if it is possible to improve
                                               it by making it more tigth - basically, is the last time before returning the formula).

                        If False, jut check whether the new formula is in the grammar.

    ALl the facoltative variables are used only if bool_pos_ex_only is True.

    '''
    
    
    ##Indices in the tabu list and can_be_changed list 
    # 0 represents (exists to forall)
    
    # 1 represents (eventually to always)
    # 2 represents (or to and)
    # 3 represents (implies to <->)
    # 4 represents (implies to and)
    
    # 5 represents (eventually to eventually always)
    # 6 represents (eventually to always eventually)
    
    if True: #inputs.length is not None or formula.length >= inputs.max_length:
        rules = [ ['E','A'], \
             # rules not changing the lenght of the formula
             ['eventually',['always']] , ['or',['and']] , ['implies', ['<->']], [ 'implies', ['and']] ]
    
    else:
        rules = [ ['E','A'], \
             # rules not changing the lenght of the formula
             ['eventually',['always']] , ['or',['and']] , ['implies', ['<->']], [ 'implies', ['and']] ,
             # rules changing the lenght of the formula
             ['eventually' , ['eventually', 'always']], ['eventually', ['always', 'eventually']]]
   
    number_of_rules = len(rules)
    
    
    # Tabu_moves is a list of lists. 
    # Each list is related to one specific change (see list above).
    # The list contain the indices of the position of the operators (e.g. always for rule 0) for which the transformation (to eventually) is not allowed anymore.
    tabu_moves = [[]] * number_of_rules # Empty tabu list
    
    
    while True: 
        
        can_be_changed = [False] * number_of_rules
        weights = [ 0 ] * number_of_rules
        
        numb_exists = len(formula.quantifiers) - sum(formula.quantifiers) # Number of exists
        
        ## If quantifier exists can be changed
        if  inputs.bool_mutation_quantifiers and  numb_exists  > len(tabu_moves[0]): #  #'0' in quantifiers > # tabu exists
            # can_be_changed[0] = True
            # Fixed probability for quantifiers
            # weights[0] = 10 # 1/10 of probability for the rule on the quantifiers
            
            #For the first experiments we do not allow for rules to the quantifiers
            can_be_changed[0] = False
            
        for i in range(1, number_of_rules):
            # If rule i is applicable: 
            #If the rule changes the length of the formula, check whether the length of the formula was fixed 
            if formula.string.count(rules[i][0]) > len(tabu_moves[i]): can_be_changed[i] = True
        
        #If none of the syntactic rules can be applied
        if sum(can_be_changed) == 0 : return False, formula
        
        # Equal probability to the allowed formulas
        value_eq_prob = (100 - weights[0] )/ sum(can_be_changed)
        
        for i in range(number_of_rules):
            if can_be_changed[i]: weights[i] = value_eq_prob
        
        #Sample which rule to apply
        case = random.choices(list(range(number_of_rules)), weights = weights, k = 1)[0]
        aux = 0
        
        if case == 0:  ##  Exists to Forall
            #Update quantifiers
            new_quantifiers = copy.deepcopy(formula.quantifiers)
            #Randomly select which forall to replace
            indx = random.sample(range(numb_exists) ,1)[0] # sample which exists (1st, 2nd,..)
            while True:
                if indx in tabu_moves[2]: indx = (indx +1) % (numb_exists) #change exists if the selected one is tabu
                else: break # stop if the sampled quantifier is not tabu
            for el_quant in range(len(new_quantifiers)):
                if new_quantifiers[el_quant] == 0:
                    if aux == indx: 
                        new_quantifiers[el_quant] = 1 #update to exists
                        break
                    else: aux += 1
                    
            # Update quantifiers levels       
            new_quantifier_levels = copy.deepcopy(formula.quantifier_levels)
            numb_ex = 0 #number of exists encountered so far (while looking for the indx_th one)
            
            for ind_ql in range(1, len(new_quantifier_levels)):
                # If number of exists is greater than the position of the selected exists -> apply changes and exit
                if numb_ex + new_quantifier_levels[ind_ql][0].count('E') >= el_quant:
                    #Replace the E in new_quantifier_levels[ind_ql][0] appearing in position (indx - numb_ex)
                    for el in range(1,len(new_quantifier_levels[ind_ql][0])):
                        if new_quantifier_levels[ind_ql][0][:el].count('E') + numb_ex == el_quant:  
                            new_quantifier_levels[ind_ql][0] = new_quantifier_levels[ind_ql][0][:el-1] + new_quantifier_levels[ind_ql][0][el-1].replace('E', 'A') +new_quantifier_levels[ind_ql][0][el:]
                            break
                    break
                # Else , update the number of exists and consider the following item
                numb_ex += new_quantifier_levels[ind_ql][0].count('E')        
            
            new_formula = gramm.Formula(new_quantifiers, new_quantifier_levels, formula.string_param, formula.nodes, formula.length, string = formula.string)
    
        else:
            new_nodes = copy.deepcopy(formula.nodes)
            if bool_pos_ex_only: new_nodes_param = copy.deepcopy(formula.nodes_param)
        
            #Randomly select which 'starting' operator to replace
            indx = random.sample(range(formula.string.count(rules[case][0])) ,1)[0] # sample which always (1st, 2nd,..)
            while True:
                if indx in tabu_moves[case]: indx = (indx +1) % (formula.string.count(rules[case][0]) ) #change always if the selected one is tabu
                else: break # stop if the sampled always is not tabu
                
            for index_node, node in enumerate(new_nodes):
                if rules[case][0] in node.data:
                    if aux == indx: 
                        node.data = rules[case][1][0]
                        if bool_pos_ex_only: new_nodes_param[index_node].data = rules[case][1][0]
                        if case in [5,6]: 
                            # Add new node in the list
                            new_nodes = gramm.insert_leftchild_unary_node_in_list_nodes(new_nodes, rules[case][1][1], index_node + 1)
                            if bool_pos_ex_only: new_nodes_param = gramm.insert_leftchild_unary_node_in_list_nodes(new_nodes_param, rules[case][1][1], index_node + 1)
                        break
                    else: aux += 1
                       
            if bool_pos_ex_only: 
                new_formula = gramm.Formula(formula.quantifiers, formula.quantifier_levels, gramm.tree_to_formula(new_nodes_param), new_nodes, len(new_nodes_param), string = gramm.tree_to_formula(new_nodes) )
            else:
                new_formula = gramm.Formula(formula.quantifiers, formula.quantifier_levels, gramm.tree_to_formula(new_nodes), new_nodes, len(new_nodes), string = gramm.tree_to_formula(new_nodes))
        
        # Check whether the new formula is satisfied and whether it is new
        if bool_pos_ex_only:  
            # The new formula obeys to the rules of the grammar and is not already stored as a satisfied formula
            if not set([fun_eval.print_formula(new_formula.quantifiers, new_formula.string_param)]).issubset(set_database):
                bool_inclusion = gramm.check_if_in_gramm(new_formula.nodes, grammar, inputs)
                if bool_inclusion: 
                    new_formula.cost = fun_eval.cost(traces_pos, None, new_formula, inputs, monitor, count)
                    ## !!! Update the number of attempted strengthening formulas
                    count.strengthened_same_formula += 1
                    if new_formula.cost != 'runout' and new_formula.cost > 0 : return True, new_formula
            # The new formula DOES NOT obey the rules of the grammar or it is violated
            #Update the tabu list
            aux_append = tabu_moves[case].copy()
            aux_append.append(indx)
            tabu_moves[case] =  aux_append.copy()
        
        # Just check whether the new formula is in the grammar
        elif not bool_pos_ex_only:   
            # The new formula obeys to the rules of the grammar
            if gramm.check_if_in_gramm(new_formula.nodes, grammar,inputs): 
                return True, new_formula # type: ignore
            # The new formula DOES NOT obey to the rules of the grammar
            else: 
                #Update the tabu list
                aux_append = tabu_moves[case].copy()
                aux_append.append(indx)
                tabu_moves[case] =  aux_append.copy()
