1.Introduction
=
This work is about cooperative transportation systems (CTSs) operating on 3D uneven terrain. The automated ground vehicle (AGV) in this work consists of a four wheel-differential drive (FWD) chassis and an active end-effector (a lift mechanism). Then,  we establish a kinematic model of the AGV in a form of linear parameters varying system. Moreover, we utilize a neural network to obtain nonlinear errors between kinematic model and real physical world when CTS moving cross bumps. The error is named as coupling effect which means AGVs in the CTS could affect each other due to hard constraints. Furthermore, we design a $H\infty$ consensus control law to improve payload stability and tracking accuracy. It is common to see that other papers use consensus control law to solely improve tracking accuracy of the payload, but this work is trying to investigate how to introduce the consensus control into active end-effectors and how much the consensus control improves payload stability. Finally, we verify the methods in Coppeliasim. The simulation video can bee seen here:
<video src="https://private-user-images.githubusercontent.com/90321869/524529128-5fe45383-2773-4557-a2c2-46ed0a26fac1.mp4" controls="controls" width="500" height="300"></video>

2.Files
=
## 📁 Project Structure
```
├── scenarios/                         # scenarios in simulation
│   ├── ctr_multi_1.ttt/               # scenario 1 
│   ├── ctr_multi_2.ttt/               # scenario 2  
│   ├── ctr_multi_3.ttt/               # scenario 3  
│   └── ...                    
├── init_zmq.py/                       # initialze the zmq functions 
├── coppeliaSimType.py                 # define zmq functions to help autofill 
├── ctrcontrol_multiagent.py           # main file for simulation 
├── ctrcontrol_multiagent2.py          # mail file for simulation 
├── ctrcontrol_multiagent_Lc1.py       # mail file for simulation 
├── model.pth/                    # neural network model 
├── model2.pth/                   # neural network model2
├── model3.pth/                   # neural network model3 
├── model4.pth/                   # neural network model4 
├── plot.py/                      # plot the data in real time
├── ref_trj.py/                   # generate reference trajectory 
├── README.md                     # read me  
└── control_ctr.py                # py file to store necessary control related functions including solver
```
3.How to use
=
<img width="128" height="128" alt="image" src="https://github.com/user-attachments/assets/522e527d-a4f4-44cc-b458-31fb9f57f924" /> 
<img width="50" height="50" alt="image" src="https://github.com/user-attachments/assets/24e99941-524d-443c-a8a4-7dd5030900d2" />
 <img width="128" height="128" alt="image" src="https://github.com/user-attachments/assets/95ed117d-dde2-4fe2-971a-3ce92f3cb2b6" />


This work is based on Coppeliasim + Python. 
1. Download Coppeliasim (EDU version is free) and python 3.12.9
2. clone the project into your workspace 
3. ZeroMQ (ZMQ) remote API is used to exchange info between python and coppeliasim. Follow the instrument provided by Coppeliasim and install ZMQ in python
4. Install necessazry packs used in python codes.
5. open Coppeliasim and open scene "ctr_multi_7.ttt"
6. open python and go to the workfolder and open "ctrcontrol_multiagent.py"
7. run python file

4.Notes
=
1.There are some warnings about types. Please ignore the warnings because the warnings are from "coppeliaSimType.py" that I redefined ZMQ functions so that I can use autofill function to access zmq functions faster.
2.Due to some works being studying for future paper, some files could not be uploaded. But here is the basic project helping people to understand this work. 
   
