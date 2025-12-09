
import numpy as np
import init_zmq
import control_ctr
import plot
import pandas as pd
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import coppeliaSimType
import scipy.linalg as la
import torch
import torch.nn as nn
from datetime import datetime

def use_refcontrol():
    vc1, wc1 = control_ref.calculate_velocity(control_ref.trj1_x[index], control_ref.trj1_y[index], robot1_pos[0], robot1_pos[1], robots1_orintation[2])
    vc2, wc2 = control_ref.calculate_velocity(control_ref.trj2_x[index], control_ref.trj2_y[index], robot2_pos[0], robot2_pos[1], robots2_orintation[2])
    vc3, wc3 = control_ref.calculate_velocity(control_ref.trj3_x[index], control_ref.trj3_y[index], robot3_pos[0], robot3_pos[1], robots3_orintation[2])
    vc4, wc4 = control_ref.calculate_velocity(control_ref.trj4_x[index], control_ref.trj4_y[index], robot4_pos[0], robot4_pos[1], robots4_orintation[2])
    
    vc1 = np.clip(vc1, control_ref.u_min[0], control_ref.u_max[0])
    wc1 = np.clip(wc1, control_ref.u_min[1], control_ref.u_max[1])
    vc2 = np.clip(vc2, control_ref.u_min[0], control_ref.u_max[0])
    wc2 = np.clip(wc2, control_ref.u_min[1], control_ref.u_max[1])
    vc3 = np.clip(vc3, control_ref.u_min[0], control_ref.u_max[0])
    wc3 = np.clip(wc3, control_ref.u_min[1], control_ref.u_max[1])
    vc4 = np.clip(vc4, control_ref.u_min[0], control_ref.u_max[0])
    wc4 = np.clip(wc4, control_ref.u_min[1], control_ref.u_max[1])
    
    
    v_l1, v_r1 = this_sim.motor_driver(vc1, wc1)
    v_l2, v_r2 = this_sim.motor_driver(vc2, wc2)
    v_l3, v_r3 = this_sim.motor_driver(vc3, wc3)
    v_l4, v_r4 = this_sim.motor_driver(vc4, wc4)
    
    this_sim.set_velocity_robots1(v_l1, v_l1, v_r1, v_r1)
    this_sim.set_velocity_robots2(v_l2, v_l2, v_r2, v_r2)
    this_sim.set_velocity_robots3(v_l3, v_l3, v_r3, v_r3)
    this_sim.set_velocity_robots4(v_l4, v_l4, v_r4, v_r4)
    
    pass

def send_control():
    v_l1, v_r1 = this_sim.motor_driver(u1[0], u1[1])
    v_l2, v_r2 = this_sim.motor_driver(u2[0], u2[1])
    v_l3, v_r3 = this_sim.motor_driver(u3[0], u3[1])
    v_l4, v_r4 = this_sim.motor_driver(u4[0], u4[1])
    
    this_sim.set_velocity_robots1(v_l1, v_l1, v_r1, v_r1)
    this_sim.set_velocity_robots2(v_l2, v_l2, v_r2, v_r2)
    this_sim.set_velocity_robots3(v_l3, v_l3, v_r3, v_r3)
    this_sim.set_velocity_robots4(v_l4, v_l4, v_r4, v_r4)
    
    this_sim.set_shaft_velocity(u1[2], u2[2], u3[2], u4[2])
    
    pass

class MyNet(nn.Module):
    def __init__(self, input_dim=12, output_dim=5):
        super(MyNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.model(x)   


if __name__ == "__main__":

    #-------------- initialize simulation----------------#
    client = RemoteAPIClient()
    sim: coppeliaSimType.SimType = client.require('sim')
    sim.setStepping(True)  # Enable simulation stepping
    this_sim = init_zmq.simulation_init(sim)  # Initialize simulation

    control_ma = control_ctr.multi_agent()  # Initialize controller
    control_ref = control_ctr.control() # Initialize reference trajectory
    sub_controller = control_ctr.control() # Initialize sub controller
    
    
    num_robots = 4
    plot_v = plot.Realtimeplot(num_robots, 'velocity', this_sim.max_simtime, 0, 0.5) # Initialize linear velocity plot
    plot_w = plot.Realtimeplot(num_robots, 'angular velocity', this_sim.max_simtime, -1, 1) # Initialize angular velocity plot
    plot_h = plot.Realtimeplot(num_robots, 'shaft', this_sim.max_simtime, sub_controller.u_min[2], sub_controller.u_max[2]) # Initialize shaft velocity plot
    plot_payload = plot.Realtimeplot(4, 'payload', this_sim.max_simtime, -0.15, 0.15) # Initialize payload plot

    
    # load neural network model
    model1 = MyNet()
    model1.load_state_dict(torch.load('./train_NN/model.pth'))
    model1.eval()
    
    model2 = MyNet()
    model2.load_state_dict(torch.load('./train_NN/model2.pth'))
    model2.eval()
    
    model3 = MyNet()
    model3.load_state_dict(torch.load('./train_NN/model3.pth'))
    model3.eval()
    
    model4 = MyNet()
    model4.load_state_dict(torch.load('./train_NN/model4.pth'))
    model4.eval()
    # store data
    data = []
    
    sim.startSimulation()  # Start the simulation

    


    while sim.getSimulationTime() < this_sim.max_simtime:
        
        # Get the current time
        time_sim = sim.getSimulationTime()
        
        ## read robots states
        robot1_pos, robots1_orintation = this_sim.find_position_robot(this_sim.body1_handle)
        robot1_v_linear, robot1_v_angular = this_sim.find_velocity_robot(this_sim.body1_handle)
        h1 = this_sim.find_shaft_position(this_sim.shaft1_handle1)
        xl1, yl1, zl1 = control_ma.calculate_PL(robot1_pos[0], robot1_pos[1], robot1_pos[2], robots1_orintation[2], robots1_orintation[1], robots1_orintation[0], h1)
        xr1, yr1, zr1 = control_ma.calculate_PR(robot1_pos[0], robot1_pos[1], robot1_pos[2], robots1_orintation[2], robots1_orintation[1], robots1_orintation[0])
        
        robot2_pos, robots2_orintation = this_sim.find_position_robot(this_sim.body2_handle)
        robot2_v_linear, robot2_v_angular = this_sim.find_velocity_robot(this_sim.body2_handle)
        h2 = this_sim.find_shaft_position(this_sim.shaft2_handle2)
        xl2, yl2, zl2 = control_ma.calculate_PL(robot2_pos[0], robot2_pos[1], robot2_pos[2], robots2_orintation[2], robots2_orintation[1], robots2_orintation[0], h2)
        xr2, yr2, zr2 = control_ma.calculate_PR(robot2_pos[0], robot2_pos[1], robot2_pos[2], robots2_orintation[2], robots2_orintation[1], robots2_orintation[0])
            
        robot3_pos, robots3_orintation = this_sim.find_position_robot(this_sim.body3_handle)
        robot3_v_linear, robot3_v_angular = this_sim.find_velocity_robot(this_sim.body3_handle)
        h3 = this_sim.find_shaft_position(this_sim.shaft3_handle3)
        xl3, yl3, zl3 = control_ma.calculate_PL(robot3_pos[0], robot3_pos[1], robot3_pos[2], robots3_orintation[2], robots3_orintation[1], robots3_orintation[0], h3)
        xr3, yr3, zr3 = control_ma.calculate_PR(robot3_pos[0], robot3_pos[1], robot3_pos[2], robots3_orintation[2], robots3_orintation[1], robots3_orintation[0])
        
        robot4_pos, robots4_orintation = this_sim.find_position_robot(this_sim.body4_handle)
        robot4_v_linear, robot4_v_angular = this_sim.find_velocity_robot(this_sim.body4_handle)
        h4 = this_sim.find_shaft_position(this_sim.shaft4_handle4)
        xl4, yl4, zl4 = control_ma.calculate_PL(robot4_pos[0], robot4_pos[1], robot4_pos[2], robots4_orintation[2], robots4_orintation[1], robots4_orintation[0], h4)
        xr4, yr4, zr4 = control_ma.calculate_PR(robot4_pos[0], robot4_pos[1], robot4_pos[2], robots4_orintation[2], robots4_orintation[1], robots4_orintation[0])
        
        payload_pos, payload_orintation = this_sim.find_position_robot(this_sim.payload_handle)
        
        rod = 2
        if sub_controller.trj_x.all() and sub_controller.trj_y.all():
            index = sub_controller.find_target_point(robot1_pos, sub_controller.trj_x, sub_controller.trj_y)
        
            ## control the robot 1
            M_c1 = control_ma.caluculate_M(robots1_orintation[2], robots1_orintation[1], robots1_orintation[0], h1)
            D_c1 = control_ma.calculate_D(robots1_orintation[2], robots1_orintation[1], robots1_orintation[0], h1)
            # M_c1 = np.round(M_c1, rod)
            # D_c1 = np.round(D_c1, rod)
            x1 = np.array([xl1, yl1, zl1, xr1, yr1])
            x_ref1 = np.array([sub_controller.trj1_x[index], sub_controller.trj1_y[index], sub_controller.h, sub_controller.trj1_x[index], sub_controller.trj1_y[index]])
            x_error1 = x1 - x_ref1
            
            ## control the robot 2
            M_c2 = control_ma.caluculate_M(robots2_orintation[2], robots2_orintation[1], robots2_orintation[0], h2)
            D_c2 = control_ma.calculate_D(robots2_orintation[2], robots2_orintation[1], robots2_orintation[0], h2)
            # M_c2 = np.round(M_c2, rod)
            # D_c2 = np.round(D_c2, rod)
            x2 = np.array([xl2, yl2, zl2, xr2, yr2])
            x_ref2 = np.array([sub_controller.trj2_x[index], sub_controller.trj2_y[index], sub_controller.h, sub_controller.trj2_x[index], sub_controller.trj2_y[index]])
            x_error2 = x2 - x_ref2
            
            ## control the robot 3
            M_c3 = control_ma.caluculate_M(robots3_orintation[2], robots3_orintation[1], robots3_orintation[0], h3)
            D_c3 = control_ma.calculate_D(robots3_orintation[2], robots3_orintation[1], robots3_orintation[0], h3)
            # M_c3 = np.round(M_c3, rod)
            # D_c3 = np.round(D_c3, rod)
            x3 = np.array([xl3, yl3, zl3, xr3, yr3])
            x_ref3 = np.array([sub_controller.trj3_x[index], sub_controller.trj3_y[index], sub_controller.h, sub_controller.trj3_x[index], sub_controller.trj3_y[index]])
            x_error3 = x3 - x_ref3
            
            ## control the robot 4
            M_c4 = control_ma.caluculate_M(robots4_orintation[2], robots4_orintation[1], robots4_orintation[0], h4)
            D_c4 = control_ma.calculate_D(robots4_orintation[2], robots4_orintation[1], robots4_orintation[0], h4)
            # M_c4 = np.round(M_c4, rod)
            # D_c4 = np.round(D_c4, rod)
            x4 = np.array([xl4, yl4, zl4, xr4, yr4])
            x_ref4 = np.array([sub_controller.trj4_x[index], sub_controller.trj4_y[index], sub_controller.h, sub_controller.trj4_x[index], sub_controller.trj4_y[index]])
            x_error4 = x4 - x_ref4  

            M_d = la.block_diag(M_c1,M_c2,M_c3,M_c4)
            D_d = la.block_diag(D_c1,D_c2,D_c3,D_c4)
            
            combined_input1 = np.array([robots2_orintation[2], robots2_orintation[1], robots2_orintation[0], h2,
                      robots3_orintation[2], robots3_orintation[1], robots3_orintation[0], h3,
                        robots4_orintation[2], robots4_orintation[1], robots4_orintation[0], h4]) * 1e4 * 0.05
            combined_tensor1 = torch.tensor(combined_input1, dtype=torch.float32)
            
            combined_input2 = np.array([robots1_orintation[2], robots1_orintation[1], robots1_orintation[0], h1,
                      robots3_orintation[2], robots3_orintation[1], robots3_orintation[0], h3,
                        robots4_orintation[2], robots4_orintation[1], robots4_orintation[0], h4]) * 1e4 * 0.05
            combined_tensor2 = torch.tensor(combined_input2, dtype=torch.float32)
            
            combined_input3 = np.array([robots1_orintation[2], robots1_orintation[1], robots1_orintation[0], h1,
                      robots2_orintation[2], robots2_orintation[1], robots2_orintation[0], h2,
                        robots4_orintation[2], robots4_orintation[1], robots4_orintation[0], h4]) * 1e4 * 0.05
            combined_tensor3 = torch.tensor(combined_input3, dtype=torch.float32)
            
            combined_input4 = np.array([robots1_orintation[2], robots1_orintation[1], robots1_orintation[0], h1,    
                      robots2_orintation[2], robots2_orintation[1], robots2_orintation[0], h2,
                        robots3_orintation[2], robots3_orintation[1], robots3_orintation[0], h3]) * 1e4 * 0.05
            combined_tensor4 = torch.tensor(combined_input4, dtype=torch.float32)
            
            with torch.no_grad():
                output1 = model1(combined_tensor1)
                output2 = model2(combined_tensor2)
                output3 = model3(combined_tensor3)
                output4 = model4(combined_tensor4)
            
            kk = 0.01
            
            K = control_ma.compute_multi_gain(M_d,D_d)
            K = K  # Scale the gain matrix
            K11 = K[0:3, 0:5]
            K12 = K[0:3, 5:10]
            K13 = K[0:3, 10:15]
            K14 = K[0:3, 15:20]
            K21 = K[3:6, 0:5]
            K22 = K[3:6, 5:10]
            K23 = K[3:6, 10:15]
            K24 = K[3:6, 15:20]
            K31 = K[6:9, 0:5]
            K32 = K[6:9, 5:10]
            K33 = K[6:9, 10:15]
            K34 = K[6:9, 15:20]
            K41 = K[9:12, 0:5]
            K42 = K[9:12, 5:10]
            K43 = K[9:12, 10:15]
            K44 = K[9:12, 15:20]
            
            
            err1 = [x_error1[0]*kk, x_error1[1], x_error1[2], x_error1[3]*kk, x_error1[4]] 
            err_12 = [xl1 + 0.5 - xl2, yl1 - yl2, zl1 - zl2, xr1 + 0.5 - xr2, yr1 - yr2]
            err_13 = [xl1 - xl3, yl1 + 0.5 - yl3, zl1 - zl3, xr1 - xr3, yr1 + 0.5 - yr3]
            err_14 = [xl1 +0.5 - xl4, yl1 + 0.5 - yl4, zl1 - zl4, xr1 + 0.5 - xr4, yr1 + 0.5 - yr4]  
            a11, a12, a13, a14= K11 @ err1, K12 @ err_12, K13 @ err_13, K14 @ err_14
            net_fd1 = np.linalg.pinv(M_c1) @ output1.numpy() * 1e-4
            u1 = sub_controller.u_ref + a11 + a12 + a13 + a14 + net_fd1
            u1 = np.clip(u1, sub_controller.u_min, sub_controller.u_max)

            
            err2 = [x_error2[0]*kk, x_error2[1], x_error2[2], x_error2[3]*kk, x_error2[4]]
            err21 = [xl2 - 0.5 - xl1, yl2 - yl1, zl2 - zl1, xr2 - 0.5 - xr1, yr2 - yr1]
            err23 = [xl2 - 0.5 - xl3, yl2 + 0.5 - yl3, zl2 - zl3, xr2 - 0.5 - xr3, yr2 + 0.5 - yr3]
            err24 = [xl2 - xl4, yl2 + 0.5 - yl4, zl2 - zl4, xr2 - xr4, yr2 + 0.5 - yr4]
            a21, a22, a23, a24 = K22 @ err2, K21 @ err21, K23 @ err23, K24 @ err24
            net_fd2 = np.linalg.pinv(M_c2) @ output2.numpy() * 1e-4 
            u2 = sub_controller.u_ref + a21 + a22 + a23 + a24 + net_fd2
            u2 = np.clip(u2, sub_controller.u_min, sub_controller.u_max)

            
            err3 = [x_error3[0]*kk, x_error3[1], x_error3[2], x_error3[3]*kk, x_error3[4]]
            err31 = [xl3 - xl1, yl3 - 0.5 - yl1, zl3 - zl1, xr3 - xr1, yl3 - 0.5 - yr1]
            err32 = [xl3 + 0.5 - xl2, yl3 - 0.5 - yl2, zl3 - zl2, xr3 + 0.5 - xr2, yr3 - 0.5 - yr2]
            err34 = [xl3 +0.5 - xl4, yl3 - yl4, zl3 - zl4, xr3 + 0.5 - xr4, yr3 - yr4]
            a31, a32, a33, a34 = K33 @ err3, K31 @ err31, K32 @ err32, K34 @ err34
            net_fd3 = np.linalg.pinv(M_c3) @ output3.numpy() * 1e-4
            u3 = sub_controller.u_ref + a31 + a32 + a33 + a34 + net_fd3
            u3 = np.clip(u3, sub_controller.u_min, sub_controller.u_max)
            
            err4 = [x_error4[0]*kk, x_error4[1], x_error4[2], x_error4[3]*kk, x_error4[4]]
            err_14 = [xl4 - 0.5 - xl1, yl4 - 0.5 - yl1, zl4 - zl1, xr4 - 0.5 - xr1, yr4 - 0.5 - yr1]
            err_24 = [xl4 - xl2, yl4 - 0.5 - yl2, zl4 - zl2, xr4 - xr2, yr4 - 0.5 - yr2]
            err_34 = [xl4 - 0.5 - xl3, yl4 - yl3, zl4 - zl3, xr4 - 0.5 - xr3, yr4 - yr3]
            a41, a42, a43, a44 = K44 @ err4, K41 @ err_14, K42 @ err_24, K43 @ err_34
            net_fd4 = np.linalg.pinv(M_c4) @ output4.numpy() * 1e-4
            u4 = sub_controller.u_ref + a41 + a42 + a43 + a44 + net_fd4
            u4 = np.clip(u4, sub_controller.u_min, sub_controller.u_max)
            
            
            send_control()  # Send control commands to the robots

            # 将数据添加到列表中
            data.append([time_sim, payload_pos[0], payload_pos[1], payload_pos[2], payload_orintation[0], payload_orintation[1], payload_orintation[2], u1[0], u1[1], u1[2], u2[0], u2[1], u2[2], u3[0], u3[1], u3[2], u4[0], u4[1], u4[2]])
            

        try:
            plot_v.update(time_sim, [u1[0], u2[0], u3[0], u4[0]])
            plot_w.update(time_sim, [u1[1], u2[1], u3[1], u4[1]])
            plot_h.update(time_sim, [u1[2], u2[2], u3[2], u4[2]])
            plot_payload.update(time_sim, [0, payload_orintation[0], payload_orintation[1], payload_orintation[2]])
        except:
            print('Plotting error')
            plot_v.show()
            plot_w.show()
            plot_h.show()
            plot_payload.show()
            break   
        if time_sim > 70:
            break
        
        sim.step()  # Step the simulation
    sim.stopSimulation()  # Stop the simulation

    # 将数据转换为 DataFrame
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f'./new_data/H_data_multi_{current_time}.csv'
    df = pd.DataFrame(data, columns=['time', 'x', 'y', 'z', 'alpha', 'beta', 'gamma', 'u1', 'w1', 'h1', 'u2', 'w2', 'h2', 'u3', 'w3', 'h3', 'u4', 'w4', 'h4'])
    df.to_csv(filename, index=False)







