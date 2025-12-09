import numpy as np
import control_ctr 


class ref_trj:
    def __init__(self):
        pass
    
    def generate_ref_trj(self):
        self.trj_x = np.linspace(0.1, 20, 200)
        self.trj_y = np.full(200, 0.25)
        
        return self.trj_x, self.trj_y
    
    
    
    