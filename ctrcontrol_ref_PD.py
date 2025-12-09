
import numpy as np
import time
import init_zmq
import control_ctr
import plot
import pandas as pd
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import coppeliaSimType
#-------------- initialize simulation----------------#
client = RemoteAPIClient()
sim: coppeliaSimType.SimType = client.require('sim')
sim.setStepping(True)  # Enable simulation stepping
this_sim = init_zmq.simulation_init(sim)  # Initialize simulation

num_robots = 4
plot_v = plot.Realtimeplot(num_robots, 'velocity', this_sim.max_simtime, 0, 0.5) # Initialize linear velocity plot
plot_w = plot.Realtimeplot(num_robots, 'angular velocity', this_sim.max_simtime, -0.3, 0.3) # Initialize angular velocity plot
plot_payload = plot.Realtimeplot(4, 'payload', this_sim.max_simtime, -0.2, 0.2) # Initialize payload plot

controller = control_ctr.control() # Initialize controller

# 用于存储数据的列表
data = []

#-------------- start simulation----------------#
sim.startSimulation()  # Start the simulation

while sim.getSimulationTime() < this_sim.max_simtime:
    sim_time = sim.getSimulationTime()
    ## read robots states
    robot1_pos, robots1_orintation = this_sim.find_position_robot(this_sim.body1_handle)
    robot1_v_linear, robot1_v_angular = this_sim.find_velocity_robot(this_sim.body1_handle)
    h1 = this_sim.find_shaft_position(this_sim.shaft1_handle1)
    
    robot2_pos, robots2_orintation = this_sim.find_position_robot(this_sim.body2_handle)
    robot2_v_linear, robot2_v_angular = this_sim.find_velocity_robot(this_sim.body2_handle)
    h2 = this_sim.find_shaft_position(this_sim.shaft2_handle2)
    
    robot3_pos, robots3_orintation = this_sim.find_position_robot(this_sim.body3_handle)
    robot3_v_linear, robot3_v_angular = this_sim.find_velocity_robot(this_sim.body3_handle)
    h3 = this_sim.find_shaft_position(this_sim.shaft3_handle3)
    
    robot4_pos, robots4_orintation = this_sim.find_position_robot(this_sim.body4_handle)
    robot4_v_linear, robot4_v_angular = this_sim.find_velocity_robot(this_sim.body4_handle)
    h4 = this_sim.find_shaft_position(this_sim.shaft4_handle4)
    
    payload_pos, payload_orintation = this_sim.find_position_robot(this_sim.payload_handle)
    
    if controller.trj_x.all() and controller.trj_y.all():
        index = controller.find_target_point(robot1_pos, controller.trj_x, controller.trj_y)
        
        if index:
            vc1, wc1 = controller.calculate_velocity(controller.trj1_x[index], controller.trj1_y[index], robot1_pos[0], robot1_pos[1], robots1_orintation[2])
            vh1 = controller.calculate_lift_velocity(controller.h, h1)
            vh1 = np.clip(vh1, -0.1, 0.1)
            
            vc2, wc2 = controller.calculate_velocity(controller.trj2_x[index], controller.trj2_y[index], robot2_pos[0], robot2_pos[1], robots2_orintation[2])
            vh2 = controller.calculate_lift_velocity(controller.h, h2)
            vh2 = np.clip(vh2, -0.1, 0.1)
            
            vc3, wc3 = controller.calculate_velocity(controller.trj3_x[index], controller.trj3_y[index], robot3_pos[0], robot3_pos[1], robots3_orintation[2])
            vh3 = controller.calculate_lift_velocity(controller.h, h3)
            vh3 = np.clip(vh3, -0.1, 0.1)
            
            vc4, wc4 = controller.calculate_velocity(controller.trj4_x[index], controller.trj4_y[index], robot4_pos[0], robot4_pos[1], robots4_orintation[2])
            vh4 = controller.calculate_lift_velocity(controller.h, h4)
            vh4 = np.clip(vh4, -0.1, 0.1)
            
            vc1 = np.clip(vc1, controller.u_min[0], 0.2)
            wc1 = np.clip(wc1, controller.u_min[1], 0.2)
            vc2 = np.clip(vc2, controller.u_min[0], 0.2)
            wc2 = np.clip(wc2, controller.u_min[1], 0.2)
            vc3 = np.clip(vc3, controller.u_min[0], 0.2)
            wc3 = np.clip(wc3, controller.u_min[1], 0.2)
            vc4 = np.clip(vc4, controller.u_min[0], 0.2)
            wc4 = np.clip(wc4, controller.u_min[1], 0.2)
            
            v_l1, v_r1 = this_sim.motor_driver(vc1, wc1)
            v_l2, v_r2 = this_sim.motor_driver(vc2, wc2)
            v_l3, v_r3 = this_sim.motor_driver(vc3, wc3)
            v_l4, v_r4 = this_sim.motor_driver(vc4, wc4)
            
            this_sim.set_velocity_robots1(v_l1, v_l1, v_r1, v_r1)
            this_sim.set_velocity_robots2(v_l2, v_l2, v_r2, v_r2)
            this_sim.set_velocity_robots3(v_l3, v_l3, v_r3, v_r3)
            this_sim.set_velocity_robots4(v_l4, v_l4, v_r4, v_r4)
            
            
            this_sim.set_shaft_velocity(vh1, vh2, vh3, vh4)
            
            # 将数据添加到列表中
            data.append([sim_time, payload_pos[0], payload_pos[1], payload_pos[2], payload_orintation[0], payload_orintation[1], payload_orintation[2]])
            
        
        else:
            print('End of trajectory')
            break
    else:
        print('No trajectory')
        break 
      
    try:
        plot_v.update(sim_time, [vc1, vc2, vc3, vc4])
        plot_w.update(sim_time, [wc1, wc2, wc3, wc4])
        plot_payload.update(sim_time, [0, payload_orintation[0], payload_orintation[1], payload_orintation[2]])
    except:
        print('Plotting error')
        plot_v.show()
        plot_w.show()
        break      
    
    sim_time += controller.control_period
    
    if sim_time > 70:
        break
    
    sim.step()  # Step the simulation


sim.stopSimulation()  # Stop the simulation
        
# 将数据转换为 DataFrame
df = pd.DataFrame(data, columns=['time', 'x', 'y', 'z', 'alpha', 'beta', 'gamma'])
df.to_csv('./data/PD_data_random.csv', index=False)



