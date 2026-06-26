import numpy as np
import pandas as pd
from scipy.optimize import fsolve

#equation 2
def equation_kkt(epsilon_1, epsilon_global, moyenne_ref, m, L=1.0):

    epsilon_2 = epsilon_global - epsilon_1 
    
    #Pour éviter les divisions par zéro ou les logs négatifs
    if epsilon_1 <= 0 or epsilon_1 >= epsilon_global or epsilon_2 <= 0:
        return 1e6 
    
    return (L / np.sqrt(m)) * (1 / (np.sqrt(np.log(1 / epsilon_2)) * epsilon_2)) - (moyenne_ref / (epsilon_1 ** 2))


def optimiser_epsilons(epsilon_global, moyenne_ref, m, L=1.0, approximation=False):
    #page 2 approximation avec 1/2
    if approximation:
        #l'approximation 1/2 n'est valide que si l'epsilon global choisi le permet
        eps_1 = 0.5 if epsilon_global > 0.5 else epsilon_global * 0.5
        eps_2 = epsilon_global - eps_1
        return eps_1, eps_2
    
    # Résolution numérique de l'équation KKT (équation 2)
    estimation_initiale = epsilon_global / 2
    try:
        eps_1_opt = fsolve(equation_kkt, x0=estimation_initiale, args=(epsilon_global, moyenne_ref, m, L))[0]
        eps_2_opt = epsilon_global - eps_1_opt
        
        #contraintes
        if 0 < eps_1_opt < epsilon_global:
            return eps_1_opt, eps_2_opt
    except:
        pass
    
    #si echec solution de repli 
    return estimation_initiale, epsilon_global - estimation_initiale


