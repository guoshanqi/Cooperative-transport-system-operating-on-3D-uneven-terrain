1.Introduction
=
This work is about cooperative transportation systems (CTSs) operating on 3D uneven terrain. The automated ground vehicle (AGV) in this work consists of a four wheel-differential drive (FWD) chassis and an active end-effector (a lift mechanism). Then,  we establish a kinematic model of the AGV in a form of linear parameters varying system. Moreover, we utilize a neural network to obtain nonlinear errors between kinematic model and real physical world when CTS moving cross bumps. The error is named as coupling effect which means AGVs in the CTS could affect each other due to hard constraints. Furthermore, we design a $H\infty$ consensus control law to improve payload stability and tracking accuracy. It is common to see that other papers use consensus control law to solely improve tracking accuracy of the payload, but this work is trying to investigate how to introduce the consensus control into active end-effectors and how much the consensus control improves payload stability. Finally, we verify the methods in Coppeliasim. The simulation video can bee seen here:

2.How to use
=



This work is based on Coppeliasim + Python. 
