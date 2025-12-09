
import numpy as np
import init_zmq
import control_ctr
import plot
import pandas as pd
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import coppeliaSimType
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

    
if __name__ == "__main__":
    
    
    
    #-------------- initialize simulation----------------#
    client = RemoteAPIClient()
    sim: coppeliaSimType.SimType = client.require('sim')
    sim.setStepping(True)  # Enable stepping mode
    this_sim = init_zmq.simulation_init(sim)  # Initialize simulation

    control_h = control_ctr.H_infinity()  # Initialize controller
    control_ref = control_ctr.control() # Initialize reference trajectory
    sub_controller = control_ctr.control(0.19) # Initialize sub controller
    
    
    num_robots = 4
    plot_v = plot.Realtimeplot(num_robots, 'velocity', this_sim.max_simtime, 0, 0.5) # Initialize linear velocity plot
    plot_w = plot.Realtimeplot(num_robots, 'angular velocity', this_sim.max_simtime, -0.3, 0.3) # Initialize angular velocity plot
    plot_h = plot.Realtimeplot(num_robots, 'shaft', this_sim.max_simtime, sub_controller.u_min[2], sub_controller.u_max[2]) # Initialize shaft velocity plot
    plot_payload = plot.Realtimeplot(4, 'payload', this_sim.max_simtime, -0.15, 0.15) # Initialize payload plot

    # store data
    data = []
    
    
    sim.startSimulation()  # Start the simulation
    cf = 0 # 1: state feedback, 0: Hinf
    if cf == 1:
        R = np.array([[0.01, 0, 0],
                  [0, 1, 0],
                  [0, 0, 1],
                  ])
        P = np.array([[0.01, 0, 0, 0, 0], 
                    [0, 1, 0, 0, 0], 
                    [0, 0, 1, 0, 0],
                    [0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 1],
                    ]) # Define P matrix
    else:
        R = np.array([[0.1, 0, 0],
                      [0, 0.1, 0],
                      [0, 0, 0.3],
                        ])
        P = np.array([[1, 0, 0, 0, 0], 
                      [0, 1, 0, 0, 0], 
                      [0, 0, 1, 0, 0],
                      [0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 1],
                    ]) # Define P matrix
    while sim.getSimulationTime() < this_sim.max_simtime:
        
        # Get the current time
        time_sim = sim.getSimulationTime()
        
        ## read robots states
        robot1_pos, robots1_orintation = this_sim.find_position_robot(this_sim.body1_handle)
        robot1_v_linear, robot1_v_angular = this_sim.find_velocity_robot(this_sim.body1_handle)
        h1 = this_sim.find_shaft_position(this_sim.shaft1_handle1)
        xl1, yl1, zl1 = control_h.calculate_PL(robot1_pos[0], robot1_pos[1], robot1_pos[2], robots1_orintation[2], robots1_orintation[1], robots1_orintation[0], h1)
        xr1, yr1, zr1 = control_h.calculate_PR(robot1_pos[0], robot1_pos[1], robot1_pos[2], robots1_orintation[2], robots1_orintation[1], robots1_orintation[0])
        
        robot2_pos, robots2_orintation = this_sim.find_position_robot(this_sim.body2_handle)
        robot2_v_linear, robot2_v_angular = this_sim.find_velocity_robot(this_sim.body2_handle)
        h2 = this_sim.find_shaft_position(this_sim.shaft2_handle2)
        xl2, yl2, zl2 = control_h.calculate_PL(robot2_pos[0], robot2_pos[1], robot2_pos[2], robots2_orintation[2], robots2_orintation[1], robots2_orintation[0], h2)
        xr2, yr2, zr2 = control_h.calculate_PR(robot2_pos[0], robot2_pos[1], robot2_pos[2], robots2_orintation[2], robots2_orintation[1], robots2_orintation[0])
            
        robot3_pos, robots3_orintation = this_sim.find_position_robot(this_sim.body3_handle)
        robot3_v_linear, robot3_v_angular = this_sim.find_velocity_robot(this_sim.body3_handle)
        h3 = this_sim.find_shaft_position(this_sim.shaft3_handle3)
        xl3, yl3, zl3 = control_h.calculate_PL(robot3_pos[0], robot3_pos[1], robot3_pos[2], robots3_orintation[2], robots3_orintation[1], robots3_orintation[0], h3)
        xr3, yr3, zr3 = control_h.calculate_PR(robot3_pos[0], robot3_pos[1], robot3_pos[2], robots3_orintation[2], robots3_orintation[1], robots3_orintation[0])
        
        robot4_pos, robots4_orintation = this_sim.find_position_robot(this_sim.body4_handle)
        robot4_v_linear, robot4_v_angular = this_sim.find_velocity_robot(this_sim.body4_handle)
        h4 = this_sim.find_shaft_position(this_sim.shaft4_handle4)
        xl4, yl4, zl4 = control_h.calculate_PL(robot4_pos[0], robot4_pos[1], robot4_pos[2], robots4_orintation[2], robots4_orintation[1], robots4_orintation[0], h4)
        xr4, yr4, zr4 = control_h.calculate_PR(robot4_pos[0], robot4_pos[1], robot4_pos[2], robots4_orintation[2], robots4_orintation[1], robots4_orintation[0])
        
        payload_pos, payload_orintation = this_sim.find_position_robot(this_sim.payload_handle)
        
        round = 5
        
        if sub_controller.trj_x.all() and sub_controller.trj_y.all():
            index = sub_controller.find_target_point(robot1_pos, sub_controller.trj_x, sub_controller.trj_y)
        
            ## control the robot 1
            M_c1 = control_h.caluculate_M(robots1_orintation[2], robots1_orintation[1], robots1_orintation[0], h1)
            M_c1 = np.round(M_c1, round)
            D_c1 = control_h.calculate_D(robots1_orintation[2], robots1_orintation[1], robots1_orintation[0], h1)
            D_c1 = np.round(D_c1, round)
            if cf == 1:
                K1   = control_h.compute_SF_gain(M_c1, D_c1) # 计算状态反馈增益
            else:
                K1   = control_h.compute_Hinf_gain(M_c1, D_c1) # 计算Hinf增益
            # K1 = np.round(K1, round)
            x1 = np.array([xl1, yl1, zl1, xr1, yr1, zr1])
            x_ref1 = np.array([sub_controller.trj1_x[index], sub_controller.trj1_y[index], sub_controller.h, sub_controller.trj1_x[index], sub_controller.trj1_y[index], sub_controller.zr])
            x_error1 = x1 - x_ref1
            err1 = [x_error1[0]*0.01, x_error1[1], x_error1[2], x_error1[3]*0.01, x_error1[4]] 
            u1 = sub_controller.u_ref + R @ K1 @ P @ err1
            u1 = np.clip(u1, sub_controller.u_min, sub_controller.u_max)

            ## control the robot 2
            M_c2 = control_h.caluculate_M(robots2_orintation[2], robots2_orintation[1], robots2_orintation[0], h2)
            M_c2 = np.round(M_c2, round)
            D_c2 = control_h.calculate_D(robots2_orintation[2], robots2_orintation[1], robots2_orintation[0], h2)
            D_c2 = np.round(D_c2, round)
            if cf == 1:
                K2   = control_h.compute_SF_gain(M_c2, D_c2) # 计算状态反馈增益
            else:
                K2   = control_h.compute_Hinf_gain(M_c2, D_c2) # 计算Hinf增益
            # K2 = np.round(K2, round)
            x2 = np.array([xl2, yl2, zl2, xr2, yr2, zr2])
            x_ref2 = np.array([sub_controller.trj2_x[index], sub_controller.trj2_y[index], sub_controller.h, sub_controller.trj2_x[index], sub_controller.trj2_y[index], sub_controller.zr])
            x_error2 = x2 - x_ref2
            err2 = [x_error2[0]*0.01, x_error2[1], x_error2[2], x_error2[3]*0.01, x_error2[4]]
            u2 = sub_controller.u_ref + R @ K2 @ P @ err2
            u2 = np.clip(u2, sub_controller.u_min, sub_controller.u_max)

            ## control the robot 3
            M_c3 = control_h.caluculate_M(robots3_orintation[2], robots3_orintation[1], robots3_orintation[0], h3)
            M_c3 = np.round(M_c3, round)
            D_c3 = control_h.calculate_D(robots3_orintation[2], robots3_orintation[1], robots3_orintation[0], h3)
            D_c3 = np.round(D_c3, round)
            if cf == 1:
                K3   = control_h.compute_SF_gain(M_c3, D_c3) # 计算状态反馈增益
            else:
                K3   = control_h.compute_Hinf_gain(M_c3, D_c3)
            # K3 = np.round(K3, round)
            x3 = np.array([xl3, yl3, zl3, xr3, yr3, zr3])
            x_ref3 = np.array([sub_controller.trj3_x[index], sub_controller.trj3_y[index], sub_controller.h, sub_controller.trj3_x[index], sub_controller.trj3_y[index], sub_controller.zr])
            x_error3 = x3 - x_ref3
            err3 = [x_error3[0]*0.01, x_error3[1], x_error3[2], x_error3[3]*0.01, x_error3[4]]
            u3 = sub_controller.u_ref + R @ K3 @ P @ err3
            u3 = np.clip(u3, sub_controller.u_min, sub_controller.u_max)

            ## control the robot 4
            M_c4 = control_h.caluculate_M(robots4_orintation[2], robots4_orintation[1], robots4_orintation[0], h4)
            M_c4 = np.round(M_c4, round)
            D_c4 = control_h.calculate_D(robots4_orintation[2], robots4_orintation[1], robots4_orintation[0], h4)
            D_c4 = np.round(D_c4, round)
            if cf == 1:
                K4   = control_h.compute_SF_gain(M_c4, D_c4)
            else:
                K4   = control_h.compute_Hinf_gain(M_c4, D_c4)
            # K4 = np.round(K4, round)
            x4 = np.array([xl4, yl4, zl4, xr4, yr4, zr4])
            x_ref4 = np.array([sub_controller.trj4_x[index], sub_controller.trj4_y[index], sub_controller.h, sub_controller.trj4_x[index], sub_controller.trj4_y[index], sub_controller.zr])
            x_error4 = x4 - x_ref4
            err4 = [x_error4[0]*0.01, x_error4[1], x_error4[2], x_error4[3]*0.01, x_error4[4]]
            u4 = sub_controller.u_ref + R @ K4 @ P @ err4
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
    sim.stopSimulation(True)  # Stop the simulation

    # 将数据转换为 DataFrame
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f'./new_data/H_data_sce1_{current_time}.csv'
    df = pd.DataFrame(data, columns=['time', 'x', 'y', 'z', 'alpha', 'beta', 'gamma', 'u1', 'w1', 'h1', 'u2', 'w2', 'h2', 'u3', 'w3', 'h3', 'u4', 'w4', 'h4'])
    if cf == 1:
        df.to_csv('./new_data/H_data_gs_40.csv', index=False)
    else:
        df.to_csv(filename, index=False)







