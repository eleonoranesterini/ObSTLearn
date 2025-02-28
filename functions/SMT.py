"""
SMT solver to check trivial formulas: contradictions or tautologies

Recall that Z3 can solve nonlinear polynomial constraints, but 2**x is not a polynomial.
"""

import z3


            
def tree_to_Z3(nodes):
    
    ## HYPOTHESES:
    ## nodes[0] = root
    ##nodes[-1] ='hole'
    
    start = '' 
    formula = f'{translate_node_to_Z3(start, nodes,  0 )}'
    
    return formula

def translate_node_to_Z3(formula, nodes, i):
    
    if  'or' in nodes[i].data:
        return f'{formula} z3.Or ( {translate_node_to_Z3(formula,nodes, nodes.index(nodes[i].leftchild))} , {translate_node_to_Z3(formula,nodes, nodes.index(nodes[i].rightchild))} )'
    
    elif 'and' in nodes[i].data:
        return f'{formula} z3.And ( {translate_node_to_Z3(formula,nodes, nodes.index(nodes[i].leftchild))} , {translate_node_to_Z3(formula,nodes, nodes.index(nodes[i].rightchild))} )'
       
    elif  'not' in nodes[i].data:
        return  f'{formula}z3.Not( {translate_node_to_Z3(formula, nodes, nodes.index(nodes[i].leftchild))} )'
    
    elif '<->' in nodes[i].data:
        return f'{formula}( ({translate_node_to_Z3(formula,nodes, nodes.index(nodes[i].leftchild))}) == ({translate_node_to_Z3(formula,nodes, nodes.index(nodes[i].rightchild))}) )'
    
    elif  'implies' in nodes[i].data:
        return f'{formula}z3.Implies( {translate_node_to_Z3(formula,nodes, nodes.index(nodes[i].leftchild))} , {translate_node_to_Z3(formula,nodes, nodes.index(nodes[i].rightchild))} )'
         
    
    else: #PREDICATE
        return nodes[i].data


# def evaluate_satisfaction(formula, pi):
    
#     '''The function evaluates the satisfaction of a formula using SMT solver'''
    
#     s = z3.Solver()
#     s.add(eval(formula))
    
#     return s.check()
    

def evaluate_triviality(nodes, m ,n):
    
    '''Returns TRUE if the formula expressed by nodes is trivially satisfied.
       Returns FALSE otherwise. '''
    
    # m = number of quantifiers
    # n = number of variables
    # --> m * n is the number of variables that need to be inizialized 
        
    #X = [ Int('x%s' % i) for i in range(5) ]
    
    pi = []
    
    for i in range(m):
        
        pi.append([z3.Real(f'pi[{i}][{j}]' ) for j in range(n) ])
    
    
    formula = tree_to_Z3(nodes)
    # print('Z3 format:', formula)
    s = z3.Solver()
    
    s.add( z3.Not(eval(formula)))
    
    if s.check() == z3.unsat: return True #the formula is always True (trivial)
    else: return False #the formula is not always true, so the formula is not trivial
  
def simplify_formula(nodes, m , n):
    
    '''The function simplifies the formula.
    HOWEVER, 
    1) the output needs to be translated into English from the Poolish notation
    2) the z3 tool makes the formula in the form of only having == and <= as predicates symbols 
    Hence, the 'simplied' formula could be even longer than the original one
    3) EX 3 in images_saved/22_09_06 is not solved '''
    
    pi = []
    for i in range(m):
        pi.append([z3.Real(f'pi[{i}][{j}]' ) for j in range(n) ])
    
    formula = tree_to_Z3(nodes)
    
    return z3.simplify( eval(formula))
    

