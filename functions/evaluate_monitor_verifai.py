import sys

from verifai.monitor import specification_monitor
from functions.syntax_eval import evaluate_rob_boolean

class monitor_spec_falsifier(specification_monitor):
    def __init__(self, final_formula):
        self.final_formula = final_formula

        def specification(traj):
            string_eval, _ = evaluate_rob_boolean( [ traj ] , final_formula)
            if string_eval == 'sat' : 
                print(f'\n {final_formula},  Sat\n')
                return True
            elif string_eval == 'unsat' : 
                print(f'\n {final_formula},  Unsat\n')
                return False
            else: 
                print('Error in evaluating the formula: the monitor evaluate_rob_boolean returned an unexpected value')
                sys.exit()
        
        super().__init__(specification)
