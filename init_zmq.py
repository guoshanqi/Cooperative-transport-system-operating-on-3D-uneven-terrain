import coppeliaSimType

class simulation_init:
    def __init__(self, sim: coppeliaSimType.SimType):
        
        self.sim = sim
        self.init_first_robots()
        self.init_second_robots()
        self.init_third_robots()
        self.init_forth_robots()
        self.init_robots_body()
        
        self.wheel_radius = 0.06
        self.axis_distance = 0.12
        self.wheel_distance = 0.35/0.66
        
        self.max_simtime = 70 

     
    # init the robots body    
    def init_robots_body(self):
        self.body1_handle = self.sim.getObject('/robot1')
        self.body2_handle = self.sim.getObject('/robot2')
        self.body3_handle = self.sim.getObject('/robot3')
        self.body4_handle = self.sim.getObject('/robot4')
        self.payload_handle = self.sim.getObject('/payload')
 
    # init the first robots        
    def init_first_robots(self):
        self.lf_motor_handle1 = self.sim.getObject('/robot1/left_front_motor')
        self.lr_motor_handle1 = self.sim.getObject('/robot1/left_rear_motor')
        self.rf_motor_handle1 = self.sim.getObject('/robot1/right_front_motor')
        self.rr_motor_handle1 = self.sim.getObject('/robot1/right_rear_motor')
        self.shaft1_handle1 = self.sim.getObject( '/cy2shaft1')
    
    def set_velocity_robots1(self, v_lf, v_lr, v_rf, v_rr):
        self.sim.setJointTargetVelocity( self.lf_motor_handle1, v_lf)
        self.sim.setJointTargetVelocity( self.lr_motor_handle1, v_lr)
        self.sim.setJointTargetVelocity( self.rf_motor_handle1, v_rf)
        self.sim.setJointTargetVelocity( self.rr_motor_handle1, v_rr)
    
    # init the second robots
    def init_second_robots(self):
        self.lf_motor_handle2 = self.sim.getObject('/robot2/left_front_motor')
        self.lr_motor_handle2 = self.sim.getObject('/robot2/left_rear_motor')
        self.rf_motor_handle2 = self.sim.getObject('/robot2/right_front_motor')
        self.rr_motor_handle2 = self.sim.getObject('/robot2/right_rear_motor')
        self.shaft2_handle2 = self.sim.getObject('/cy2shaft2')
    
    def set_velocity_robots2(self, v_lf, v_lr, v_rf, v_rr):
        self.sim.setJointTargetVelocity(self.lf_motor_handle2, v_lf)
        self.sim.setJointTargetVelocity(self.lr_motor_handle2, v_lr)
        self.sim.setJointTargetVelocity(self.rf_motor_handle2, v_rf)
        self.sim.setJointTargetVelocity(self.rr_motor_handle2, v_rr)
        
    # init the third robots
    def init_third_robots(self):
        self.lf_motor_handle3 = self.sim.getObject('/robot3/left_front_motor')
        self.lr_motor_handle3 = self.sim.getObject('/robot3/left_rear_motor')
        self.rf_motor_handle3 = self.sim.getObject('/robot3/right_front_motor')
        self.rr_motor_handle3 = self.sim.getObject('/robot3/right_rear_motor')
        self.shaft3_handle3 = self.sim.getObject('/cy2shaft3')
        
    def set_velocity_robots3(self, v_lf, v_lr, v_rf, v_rr):
        self.sim.setJointTargetVelocity(self.lf_motor_handle3, v_lf)
        self.sim.setJointTargetVelocity(self.lr_motor_handle3, v_lr)
        self.sim.setJointTargetVelocity(self.rf_motor_handle3, v_rf)
        self.sim.setJointTargetVelocity(self.rr_motor_handle3, v_rr)
        
    # init the forth robots
    def init_forth_robots(self):
        self.lf_motor_handle4 = self.sim.getObject('/robot4/left_front_motor')
        self.lr_motor_handle4 = self.sim.getObject('/robot4/left_rear_motor')
        self.rf_motor_handle4 = self.sim.getObject('/robot4/right_front_motor')
        self.rr_motor_handle4 = self.sim.getObject('/robot4/right_rear_motor')
        self.shaft4_handle4 = self.sim.getObject('/cy2shaft4')

    def set_velocity_robots4(self, v_lf, v_lr, v_rf, v_rr):
        self.sim.setJointTargetVelocity(self.lf_motor_handle4, v_lf)
        self.sim.setJointTargetVelocity(self.lr_motor_handle4, v_lr)
        self.sim.setJointTargetVelocity(self.rf_motor_handle4, v_rf)
        self.sim.setJointTargetVelocity(self.rr_motor_handle4, v_rr)
        
    # find the position of the robot 
    def find_position_robot(self, handler):
        position = self.sim.getObjectPosition(handler)# type: ignore
        orientation = self.sim.getObjectOrientation(handler)# type: ignore
        return position, orientation
    
    def find_shaft_position(self, handler):
        position = self.sim.getJointPosition(handler)
        return position 
    
    # find the velocity of the robot
    def find_velocity_robot(self, handler):
        linear_velocity, angular_velocity = self.sim.getObjectVelocity(handler) # type: ignore
        return linear_velocity, angular_velocity
    
    # set the force  of the shaft
    def set_shaft_force(self, force1, force2, force3, force4):
        self.sim.setJointTargetForce(self.shaft1_handle1, force1)# type: ignore
        self.sim.setJointTargetForce(self.shaft2_handle2, force2)# type: ignore
        self.sim.setJointTargetForce(self.shaft3_handle3, force3)# type: ignore
        self.sim.setJointTargetForce(self.shaft4_handle4, force4)# type: ignore
    
    # set the velocity of the shaft
    def set_shaft_velocity(self, velocity1, velocity2, velocity3, velocity4):
        self.sim.setJointTargetVelocity(self.shaft1_handle1, velocity1)
        self.sim.setJointTargetVelocity(self.shaft2_handle2, velocity2)  
        self.sim.setJointTargetVelocity(self.shaft3_handle3, velocity3)
        self.sim.setJointTargetVelocity(self.shaft4_handle4, velocity4)

    # motor driver
    def motor_driver(self, v, w):
        w_l = v/0.06 - w*self.wheel_distance/0.06/2
        w_r = v/0.06 + w*self.wheel_distance/0.06/2
        return w_l, w_r 