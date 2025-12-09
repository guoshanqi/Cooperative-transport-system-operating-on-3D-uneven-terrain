import numpy as np
import math
import ref_trj
from scipy.linalg import solve_discrete_are
import cvxpy as cp

class control:
    def __init__(self, spd = 0.165):
        
        self.v_ref = 0.2
        self.theta_ref = 0.0
        self.control_period = 0.001  # 控制周期
        
        self.integral_x1, self.integral_y1 = 0.0, 0.0
        self.integral_x2, self.integral_y2 = 0.0, 0.0
        self.integral_x3, self.integral_y3 = 0.0, 0.0
        self.integral_x4, self.integral_y4 = 0.0, 0.0
        self.integral_h = 0.0
        self.pre_error = 0.0
        self.T_s = 0.05
        
        self.zr = 0.1
        
        self.ref_trj = ref_trj.ref_trj()
        self.trj_x, self.trj_y = self.ref_trj.generate_ref_trj()
        self.get_robots_ref_trj(self.trj_x, self.trj_y)
        
        self.u_ref = [spd, 0.0, 0.0]
        
        self.u_min = [0.1, -0.3, -0.1]
        self.u_max = [0.3, 0.3, 0.1]   

    # def generate_ref_trj_circle(self):
        
    #     x_last, y_last, z_last, theta_last = 0.25, 0.25, 0.0, 0.0  # Initialize last position of the virtual leader
    #     time = 0.0
    #     T_s = 1
    #     while time <= 120:

    #         if 0 <= time and time <= 10:
    #             w_leader = 0.0
    #         elif 10 < time and time <= 40:
    #             w_leader = 2.8 * 0.001 * (time - 5)
    #         elif 40 < time and time <= 80:
    #             w_leader = -0.05
    #         x_leader_theta = w_leader * T_s + theta_last
    #         v_leader = 0.2
    #         x_leader = v_leader * np.cos(x_leader_theta) * T_s + x_last
    #         y_leader = v_leader * np.sin(x_leader_theta) * T_s + y_last
    #         z_leader = 0.0

            

    #         x_last, y_last, z_last, theta_last = x_leader, y_leader, z_leader, x_leader_theta 

    #         if time == 0.0:
    #             trj = [x_leader + 0.15 * np.cos(theta_last), y_leader + 0.15 * np.sin(theta_last), 0.0, 0.0, 0.0, 0.0, 0.0]
    #         else:
    #             trj.extend([x_leader + 0.15 * np.cos(theta_last), y_leader + 0.15 * np.sin(theta_last), 0.0, 0.0, 0.0, 0.0, 0.0])

    #         time += T_s
        
        
    #     return trj

    # if 0 <= time and time <= 10:
    #             w_leader = 0.0
    #         elif 10 < time and time <= 20:
    #             w_leader = 0.2  * np.pi/8
    #         elif 20 < time and time <= 40:
    #             w_leader = -0.2  * np.pi/8
    #         elif 40 < time and time <= 60:
    #             w_leader = 0.2  * np.pi/8
    #         elif 60 < time and time <= 80:
    #             w_leader = -0.2  * np.pi/8
    #         elif 80 < time and time <= 90:
    #             w_leader = 0.2 * np.pi/8


    def generate_ref_trj_circle(self):
        
        x_last, y_last, z_last, theta_last = 0.25, 0.25, 0.0, 0.0  # Initialize last position of the virtual leader
        time = 0.0
        T_s = 1
        while time <= 90:

            if 0 <= time and time <= 10:
                w_leader = 0.0
            elif 10 < time and time <= 20:
                w_leader = 0.5  * np.pi/8
            elif 20 < time and time <= 40:
                w_leader = -0.5  * np.pi/8
            elif 40 < time and time <= 60:
                w_leader = 0.5  * np.pi/8
            elif 60 < time and time <= 80:
                w_leader = -0.5  * np.pi/8
            elif 80 < time and time <= 90:
                w_leader = 0.5 * np.pi/8

            x_leader_theta = w_leader * T_s + theta_last
            v_leader = 0.2
            x_leader = v_leader * np.cos(x_leader_theta) * T_s + x_last
            y_leader = v_leader * np.sin(x_leader_theta) * T_s + y_last
            z_leader = 0.0

            

            x_last, y_last, z_last, theta_last = x_leader, y_leader, z_leader, x_leader_theta 

            if time == 0.0:
                trj = [x_leader, y_leader, 0.0, 0.0, 0.0, 0.0, 0.0]
            else:
                trj.extend([x_leader, y_leader, 0.0, 0.0, 0.0, 0.0, 0.0])

            time += T_s
        
        
        return trj
    
    def generate_ref_trj_skid(self):
        
        x_last, y_last, z_last = 0.0, 0.0, 0.0  # Initialize last position of the virtual leader
        time = 0.0
        T_s = 1
        while time <= 40:

            if 0 <= time and time <= 10:
                theta = 0.0
            elif 10 < time and time <= 20:
                theta = 45/180*np.pi
            elif 20 < time and time <= 30:
                theta = -45/180*np.pi
            elif 30 < time and time <= 40:
                theta = 0
            

            x_leader_theta = theta
            v_leader = 0.2
            x_leader = v_leader * np.cos(x_leader_theta) * T_s + x_last
            y_leader = v_leader * np.sin(x_leader_theta) * T_s + y_last
            z_leader = 0.0

            

            x_last, y_last, z_last, theta_last = x_leader, y_leader, z_leader, x_leader_theta 

            if time == 0.0:
                trj = [x_leader, y_leader, 0.0, 0.0, 0.0, 0.0, 0.0]
            else:
                trj.extend([x_leader, y_leader, 0.0, 0.0, 0.0, 0.0, 0.0])

            time += T_s
        
        
        return trj

    
    def get_robots_ref_trj(self, trj_x, trj_y):
        
        self.trj1_x = trj_x
        self.trj1_y = trj_y - 0.25
        
        self.trj2_x = trj_x + 0.5
        self.trj2_y = trj_y - 0.25
        
        self.trj3_x = trj_x
        self.trj3_y = trj_y + 0.25
        
        self.trj4_x = trj_x + 0.5
        self.trj4_y = trj_y + 0.25
        
        self.h = 0.32
        
    def find_target_point(self, position, trj_x, trj_y):
        distances = []
        for i in range(len(trj_x)):
            distance = abs(math.sqrt((position[0] - trj_x[i])**2 + (position[1] - trj_y[i])**2))
            distances.append(distance)
        index = np.argmin(distances)       
        
        return index + 5
    
    def calculate_velocity(self, targetx, targety, pos_x, pos_y, yaw):
        xe1 = targetx - pos_x 
        ye1 = targety - pos_y 
        
        self.integral_x1 += xe1 * 0.01
        self.integral_y1 += ye1 * 0.01
        
        vc = 0.1 * xe1 +  self.v_ref * np.cos(self.theta_ref - yaw) + 0.01 * self.integral_x1
        wc = 6.1 * (self.theta_ref - yaw) + 1 * ye1 * self.v_ref + 0.01 * self.integral_y1
                
        return vc, wc
    
    def calculate_lift_velocity(self, targeth, pos_h):
        eh = targeth - (pos_h + 0.241)
        self.integral_h += eh * self.control_period
        derivative = (eh - self.pre_error) / self.control_period
        vh = 30 * eh + 5 * self.integral_h + 0.001 * derivative
        self.pre_error = eh
        return vh
 
class H_infinity:
    def __init__(self):
        self.T_s = 0.05
        self.n = 3
        self.m = 3
        self.p = 2
        self.A_d = np.eye(self.n)
        self.L = 0.1

        self.C_d = np.eye(self.n)
        self.D_d = np.zeros((self.n, self.m))

        self.P = cp.Variable((self.n, self.n), symmetric=True)
        self.Y = cp.Variable((self.m, self.n))

    def caluculate_M(self, theta, phi, psi, h):
        M_current = np.array([
            [np.cos(theta)*np.cos(phi), (h+0.141)*(np.cos(theta)*np.sin(psi)-np.sin(theta)*np.sin(phi)*np.cos(psi)), np.sin(theta)*np.sin(psi)+np.cos(theta)*np.sin(phi)*np.cos(psi)],
            [np.sin(theta)*np.cos(phi), (h+0.141)*(np.sin(theta)*np.sin(psi)+np.cos(theta)*np.sin(phi)*np.cos(psi)), -np.cos(theta)*np.sin(psi)+np.sin(theta)*np.sin(phi)*np.cos(psi)],
            [-np.sin(phi), 0.0, np.cos(phi)*np.cos(psi)],
            [np.cos(theta)*np.cos(phi), -self.L*np.sin(theta)*np.cos(phi), 0],
            [np.sin(theta)*np.cos(phi), self.L*np.cos(theta)*np.cos(phi), 0],
            ])
        return M_current

    def calculate_D(self, theta, phi, psi, h):
        D_current = (h + 0.141) * np.array([
            [np.cos(theta)*np.cos(phi)*np.cos(psi), -np.cos(theta)*np.sin(phi)*np.sin(psi)+np.sin(theta)*np.cos(psi)],
            [np.sin(theta)*np.cos(phi)*np.cos(psi), -np.sin(theta)*np.sin(phi)*np.sin(psi)-np.cos(theta)*np.cos(psi)],
            [-np.sin(phi)*np.cos(psi), -np.cos(phi)*np.sin(psi)],
            [-self.L*np.cos(theta)*np.sin(phi), 0],
            [-self.L*np.sin(theta)*np.cos(phi), 0],
        ])
        return D_current
    
    def compute_SF_gain(self, M_current, D_current, T_s=0.05, gamma_val=0.0, solver=cp.SCS):

        n = 5   
        m = 3   
        p = 2   

        
        # x[k+1] = x[k] + T_s*(M0*u[k] + D0*w[k])
        A_d = np.eye(n)        # 3x3
        B_d = T_s * M_current         # 3x3
        E_d = T_s * D_current         # 3x2

        #  z[k] = x[k] 
        C_d = np.array([[0.2, 0, 0, 0, 0], 
                        [0, 0.2, 0, 0, 0], 
                        [0, 0, 1, 0, 0],
                        [0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 1],
                        ]) # Define P matrix
        # D_d = np.zeros((n, m))

        P = cp.Variable((n, n), symmetric=True)
        Y = cp.Variable((m, n))

        cf = 0
        if cf == 1:
            LMI = cp.bmat([
                [P,                   (A_d @ P + B_d @ Y).T,    P @ E_d],
                [A_d @ P + B_d @ Y,   P,                        np.zeros((n, p))],
                [E_d.T @ P,           np.zeros((p, n)),         gamma_val**2 * np.eye(p)]
            ])
            constraints = [P >> 1e-6*np.eye(n), LMI >> 1e-6*np.eye(n + n + p)]
            prob = cp.Problem(cp.Minimize(gamma_val), constraints) # type: ignore
            prob.solve(solver=cp.SCS, verbose=False)
        else:
            LMI = cp.bmat([
                [-P,                   (A_d @ P + B_d @ Y).T],
                [A_d @ P + B_d @ Y,   -P,                   ],
            ])
            constraints = [P >> 0, LMI << 0]
            prob = cp.Problem(cp.Minimize(gamma_val), constraints) # type: ignore
            prob.solve(solver=cp.MOSEK, verbose=False)

        if prob.status in ['optimal', 'optimal_inaccurate']:
            P_opt = P.value
            Y_opt = Y.value
            K = Y_opt @ np.linalg.inv(P_opt) # type: ignore
       
        
        return K 
    
    def calculate_PL(self, x, y, z, theta, phi, psi, h): 
        
        xl = x + (h+0.141)*(np.cos(theta)*np.sin(psi)-np.sin(theta)*np.sin(phi)*np.cos(psi))
        yl = y + (h+0.141)*(-np.cos(theta)*np.sin(psi) + np.sin(theta)*np.sin(phi)*np.cos(psi))
        zl = z + (h+0.141)*(np.cos(phi)*np.cos(psi))
        
        return xl, yl, zl
    
    def calculate_PR(self, x, y, z, theta, phi, psi):
        xr = x + self.L*np.cos(theta)*np.cos(phi)
        yr = y + self.L*np.sin(theta)*np.cos(phi)
        zr = z - self.L*np.sin(phi)
        
        return xr, yr, zr
    
    def compute_Hinf_gain(self, M_current, D_current, T_s=0.05):
        # 
        
        n = 5    # 
        m = 3    # 
        p = 2    # 
        
        A = np.eye(n)        # 3x3
        B = T_s * M_current         # 3x3
        Bd = T_s * D_current         # 3x2
        
        
        C = np.array([
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
            ])  
        D2 = np.array([[0, 0]])  
        

        X = cp.Variable((n, n), symmetric=True)  
        W = cp.Variable((m, n))                  
        gamma = cp.Variable(nonneg=True)         

        M11 = -X
        M12 = A @ X + B @ W 
        M13 = np.zeros((n, n))
        M14 = Bd
        
        M21 = M12.T
        M22 = -X
        M23 = X.T @ C.T
        M24 = np.zeros((n, p))
        
        M31 = M13.T
        M32 = M23.T
        M33 = -gamma * np.eye(n)
        M34 = np.zeros((n, p))
        
        M41 = M14.T
        M42 = M24.T
        M43 = M34.T
        M44 = -gamma * np.eye(p)
        
        LMI = cp.bmat([
            [M11,  M12,  M13, M14],
            [M21,  M22,  M23, M24],
            [M31,  M32,  M33, M34],
            [M41,  M42,  M43, M44]
        ])
        

        
        constraints = [
            X >> 0,              
            LMI << 0              
        ]

        
        problem = cp.Problem(cp.Minimize(gamma), constraints) # type: ignore
        mosek_params = {
        "MSK_DPAR_INTPNT_CO_TOL_REL_GAP": 1e-4,    # 放宽对偶间隙
        "MSK_DPAR_INTPNT_CO_TOL_PFEAS": 1e-5,      # 放宽原始可行性
        "MSK_DPAR_INTPNT_CO_TOL_DFEAS": 1e-5,      # 放宽对偶可行性
        "MSK_IPAR_INTPNT_MAX_ITERATIONS": 1000,     # 限制迭代次数
        }

        problem.solve(solver=cp.MOSEK, verbose=False, mosek_params=mosek_params)  


        if problem.status in ['optimal', 'optimal_inaccurate']:            
            #  K = Y * X^{-1}
            X_opt = np.linalg.inv(X.value) # type: ignore
            K = W.value @ X_opt
            return K
        else:
            print(f"fail! state: {problem.status}")


class multi_agent:
    def __init__(self):
        self.T_s = 0.05
        self.n = 3
        self.m = 3
        self.p = 2
        self.A_d = np.eye(self.n)
        
        self.L = 0.15

        self.C_d = np.eye(self.n)
        self.D_d = np.zeros((self.n, self.m))

        self.P = cp.Variable((self.n, self.n), symmetric=True)
        self.Y = cp.Variable((self.m, self.n))
        
        self.L_m = np.array([[4, -1, -1, -1],
                            [-1, 4, -1, -1],
                            [-1, -1, 4, -1],
                            [-1, -1, -1, 4]])  
        # self.L_c = np.load("./train_NN/K.npy") 
        

    def caluculate_M(self, theta, phi, psi, h):
        M_current = np.array([
            [np.cos(theta)*np.cos(phi), (h+0.141)*(np.cos(theta)*np.sin(psi)-np.sin(theta)*np.sin(phi)*np.cos(psi)), np.sin(theta)*np.sin(psi)+np.cos(theta)*np.sin(phi)*np.cos(psi)],
            [np.sin(theta)*np.cos(phi), (h+0.141)*(np.sin(theta)*np.sin(psi)+np.cos(theta)*np.sin(phi)*np.cos(psi)), -np.cos(theta)*np.sin(psi)+np.sin(theta)*np.sin(phi)*np.cos(psi)],
            [-np.sin(phi), 0.0, np.cos(phi)*np.cos(psi)],
            [np.cos(theta)*np.cos(phi), -self.L*np.sin(theta)*np.cos(phi), 0],
            [np.sin(theta)*np.cos(phi), self.L*np.cos(theta)*np.cos(phi), 0],
            ])
        return M_current

    def calculate_D(self, theta, phi, psi, h):
        D_current = (h + 0.141) * np.array([
            [np.cos(theta)*np.cos(phi)*np.cos(psi), -np.cos(theta)*np.sin(phi)*np.sin(psi)+np.sin(theta)*np.cos(psi)],
            [np.sin(theta)*np.cos(phi)*np.cos(psi), -np.sin(theta)*np.sin(phi)*np.sin(psi)-np.cos(theta)*np.cos(psi)],
            [-np.sin(phi)*np.cos(psi), -np.cos(phi)*np.sin(psi)],
            [-self.L*np.cos(theta)*np.sin(phi)/(h + 0.141), 0],
            [-self.L*np.sin(theta)*np.cos(phi)/(h + 0.141), 0],
        ])
        return D_current
    
    def calculate_PL(self, x, y, z, theta, phi, psi, h): 
        
        xl = x + (h+0.141)*(np.sin(theta)*np.sin(psi) + np.cos(theta)*np.sin(phi)*np.cos(psi))
        yl = y + (h+0.141)*(-np.cos(theta)*np.sin(psi) + np.sin(theta)*np.sin(phi)*np.cos(psi))
        zl = z + (h+0.141)*(np.cos(phi)*np.cos(psi))
        
        return xl, yl, zl
    
    def calculate_PR(self, x, y, z, theta, phi, psi):
        xr = x + self.L*np.cos(theta)*np.cos(phi)
        yr = y + self.L*np.sin(theta)*np.cos(phi)
        zr = z - self.L*np.sin(phi)
        
        return xr, yr, zr
    
    def compute_multi_gain(self, diag_bd, diag_dd, T_s=0.05):
        # define system matrices
        
        n = 5    # states dimension
        m = 3    # input dimension
        p = 2    # disturbance dimension
        
        L_m = np.array([[3, -1, -1, 0],
                       [-1, 3, -1, 0],
                       [-1, -1, 4, -1],
                       [0, 0, -1, 2]]) 
        C_d = 0.001 * np.array([
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
            ])  # output matrix
        
        A_d = np.eye(n)        # 3x3
        diag_bd = T_s * diag_bd         # 3x3
        diag_dd = T_s * diag_dd         # 3x2
                        
        I_N = np.eye(4)
        A_1 = np.kron(I_N, A_d)
        B_1 = diag_bd 
        D_1 = diag_dd
        C_1 = np.kron(I_N, C_d)
        
        W1 = cp.Variable((5, 5), symmetric=True)
        W2 = cp.Variable((5, 5), symmetric=True)  
        W3 = cp.Variable((5, 5), symmetric=True)  
        W4 = cp.Variable((5, 5), symmetric=True)  
        
        W = cp.bmat([[W1, np.zeros((5, 5)), np.zeros((5, 5)), np.zeros((5, 5))],
                    [np.zeros((5, 5)), W2, np.zeros((5, 5)), np.zeros((5, 5))],
                    [np.zeros((5, 5)), np.zeros((5, 5)), W3, np.zeros((5, 5))],
                    [np.zeros((5, 5)), np.zeros((5, 5)), np.zeros((5, 5)), W4]])

        Z = cp.Variable((12, 20))
        
        gamma = cp.Variable(nonneg=True)         # H∞ performance index

        M11 = -W
        M12 = A_1 @ W + B_1 @ Z
        M13 = np.zeros((20, 20))
        M14 = D_1
        M21 = M12.T
        M22 = -W
        M23 = W.T @ C_1.T
        M24 = np.zeros((20, 8))
        M31 = M13.T
        M32 = M23.T
        M33 = -gamma * np.eye(20)
        M34 = np.zeros((20, 8))
        M41 = M14.T
        M42 = M24.T
        M43 = M34.T
        M44 = -gamma * np.eye(8)
        
        LMI = cp.bmat([
            [M11,  M12,  M13, M14],
            [M21,  M22,  M23, M24],
            [M31,  M32,  M33, M34],
            [M41,  M42,  M43, M44]
        ])
        
        # constraints
        constraints = [W1 >> 0,                
                       W2 >> 0,
                       W3 >> 0,
                        W4 >> 0,
            LMI << 0]

    
        mosek_params = {
        "MSK_DPAR_INTPNT_CO_TOL_REL_GAP": 1e-3,    
        "MSK_DPAR_INTPNT_CO_TOL_PFEAS": 1e-5,      
        "MSK_DPAR_INTPNT_CO_TOL_DFEAS": 1e-5,      
        "MSK_IPAR_INTPNT_MAX_ITERATIONS": 10000,     
        }
    
        # build optimization problem
        prob = cp.Problem(cp.Minimize(gamma), constraints) # type: ignore
        # use mosek
        prob.solve(solver=cp.MOSEK, verbose=False, mosek_params=mosek_params)  # use mosek solver
        # result processing
        if prob.status in ['optimal', 'optimal_inaccurate']:            
            #  K = Z * W^{-1}
            W_opt = np.linalg.inv(W.value) # type: ignore
            Z_va = Z.value
            Lm_inv = np.linalg.inv(np.kron(L_m, np.eye(n)))
            K = Z_va @ W_opt @ Lm_inv
            return K
        else:
            print(f"optimize fail! state: {prob.status}")
         


            
    

        
    
        
        