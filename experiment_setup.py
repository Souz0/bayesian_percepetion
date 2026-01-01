# -*- coding: utf-8 -*-
"""
Simulates experimental trials by generating visual and auditory stimulus
samples using a probabilistic generative model. 

Based on the generative model described in Kording et al. (2007),
section "Material and Methods", subsection "Generative model".

Kording, K.P., Beierholm, U., Ma, W.J., Quartz, S., Tenenbaum, J.B., Shams, 
L.: Causal Inference in Multisensory Perception. PLoS ONE 2(9), e943 (2007),
doi:10.1371/journal.pone.0000943

First released on Thu Nov 22 2023  

@author: João Filipe Ferreira
"""

# Global imports
import numpy as np              # Maths
from scipy.stats import binom   # Binomial distribution generator (from statistics toolbox)

def experiment_simulation(p_common, mu_p, sigma_p, sigma_v, sigma_a, trials):
    """
    Generates simulated experimental trial data for visual and auditory stimuli.
    
    Uses a generative model to sample stimulus parameters and sensor readings.
    
    Args:
        p_common: Probability of stimuli sharing a common cause 
        mu_p: Mean of prior distribution over stimulus source locations
        sigma_p: Std dev of prior over source locations 
        sigma_v: Std dev of visual sensory noise distribution 
        sigma_a: Std dev of auditory sensory noise distribution
        trials: Number of trials to simulate
        
    Returns: 
        real_C_exp: Generated values for number of sources 
        real_s_v_exp: Generated actual visual source locations
        real_s_a_exp: Generated actual auditory source locations  
        x_v_exp: Generated sensor readings on visual stimuli
        x_a_exp: Generated sensor readings on auditory stimuli
    """

    # ** STUDENT: MODIFY THE FOLLOWING CODE **

    rand_num = np.random.default_rng()
    real_C_exp = rand_num.binomial(n=1,p=p_common, size=trials) + 1

    real_s_v_exp = rand_num.normal(loc=mu_p, scale=sigma_p, size=trials)
    real_s_a_exp = np.zeros(trials)

    for i in real_C_exp:
        if real_C_exp[i] == 1:
            real_s_a_exp[i] = real_s_v_exp[i]
        else:
            real_s_a_exp[i] = rand_num.normal(loc=mu_p, scale=sigma_p)


    # Generate values for sensor readings x_v and x_a
    x_v_exp = rand_num.normal(loc = real_s_v_exp, scale = sigma_v, size = trials)
    x_a_exp = rand_num.normal(loc = real_s_a_exp, scale = sigma_a, size = trials)
    
    ## ** STUDENT: END OF MODIFIED CODE **

    return real_C_exp, real_s_v_exp, real_s_a_exp, x_v_exp, x_a_exp