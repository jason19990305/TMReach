import numpy as np
import mujoco
import mujoco.viewer
import matplotlib.pyplot as plt
import cv2
import time

import os

class RobotReach():
    def __init__(self , render):
        self.num_actions = 6
        self.num_states = 19
        self.render = render
        print(os.getcwd())

        self.model = mujoco.MjModel.from_xml_path('TM12s/scene_reach.xml') # load xml model
        self.data = mujoco.MjData(self.model)
        
        if self.render :
            self.viewer = mujoco.viewer.launch_passive(self.model , self.data) # Get viewer
            self.viewer.cam.distance = self.model.stat.extent * 2 # set camera distance
            self.viewer.user_scn.ngeom = 1

        mujoco.mj_resetData(self.model, self.data)  # Reset state and time.
        self.count = 0
        self.center = np.array([0.4 , 0.4 , 0.7 ])

    def reset(self):
        #print("Reset")
        mujoco.mj_resetData(self.model, self.data)  # Reset state and time.
        mujoco.mj_step(self.model, self.data)
        
        #self.desired_goal = (np.random.rand(1,3)[0] - 0.5 ) * 2  + 1#np.array([0.5 , 0.5 , 0.5])
        #self.desired_goal[2] = 2.5
        self.desired_goal = self.center + (np.random.rand(1,3)[0]) * 0.1 # np.array([0.377 ,0.324 , 0.418])
        state = []
        self.count = 0
        #self.desired_goal = np.random.rand(1,3)[0]
        self.achieved_goal = self.data.geom_xpos[self.data.geom("end").id].copy()
        #print(self.achieved_goal)
        state.append(self.desired_goal[0].copy())
        state.append(self.desired_goal[1].copy())
        state.append(self.desired_goal[2].copy())
        state.append(self.achieved_goal[0].copy())
        state.append(self.achieved_goal[1].copy())
        state.append(self.achieved_goal[2].copy())
        state.append(self.data.joint('joint_1').qpos.copy()[0])
        state.append(self.data.joint('joint_2').qpos.copy()[0])
        state.append(self.data.joint('joint_3').qpos.copy()[0])
        state.append(self.data.joint('joint_4').qpos.copy()[0])
        state.append(self.data.joint('joint_5').qpos.copy()[0])
        state.append(self.data.joint('joint_6').qpos.copy()[0])
        state.append(self.data.qvel.copy()[0])
        state.append(self.data.qvel.copy()[1])
        state.append(self.data.qvel.copy()[2])
        state.append(self.data.qvel.copy()[3])
        state.append(self.data.qvel.copy()[4])
        state.append(self.data.qvel.copy()[5])
        state.append(self.data.time)
        
        if self.render:
            mujoco.mjv_initGeom(
                    self.viewer.user_scn.geoms[0] ,
                    type = mujoco.mjtGeom.mjGEOM_SPHERE ,
                    size = [0.02, 0, 0] ,
                    pos = self.desired_goal ,
                    mat = np.eye(3).flatten() ,
                    rgba = np.array([1.0, 0 , 0 , 1.0])
                )

        return np.array(state)
    def distance(self,ach,des):
        
        return np.linalg.norm(ach - des)
    
    def step(self , action):

        #print(self.desired_goal)

        self.data.ctrl[0] = action[0] * 6.28
        self.data.ctrl[1] = action[1] * 6.28
        self.data.ctrl[2] = action[2] * 3.14
        self.data.ctrl[3] = action[3] * 6.28
        self.data.ctrl[4] = action[4] * 6.28
        self.data.ctrl[5] = action[5] * 6.28
        
        d = 0
        for i in range(5) :
            
            mujoco.mj_step(self.model, self.data)
            self.achieved_goal = self.data.geom_xpos[self.data.geom("end").id].copy()
            d =  self.distance(self.achieved_goal,self.desired_goal).copy()
            if d < 0.005:
                break

            if self.render :
                #print(self.data.sensor('accelerometer').data)
                #print(self.data.qvel.copy())
                time.sleep(0.003)
                self.viewer.sync()
        
        state = []
        #print("-------")
        #print(self.desired_goal)
        #print(self.achieved_goal)
        
        
        
        state.append(self.desired_goal[0].copy())
        state.append(self.desired_goal[1].copy())
        state.append(self.desired_goal[2].copy())
        state.append(self.achieved_goal[0].copy())
        state.append(self.achieved_goal[1].copy())
        state.append(self.achieved_goal[2].copy())
        state.append(self.data.joint('joint_1').qpos.copy()[0])
        state.append(self.data.joint('joint_2').qpos.copy()[0])
        state.append(self.data.joint('joint_3').qpos.copy()[0])
        state.append(self.data.joint('joint_4').qpos.copy()[0])
        state.append(self.data.joint('joint_5').qpos.copy()[0])
        state.append(self.data.joint('joint_6').qpos.copy()[0])
        state.append(self.data.qvel.copy()[0])
        state.append(self.data.qvel.copy()[1])
        state.append(self.data.qvel.copy()[2])
        state.append(self.data.qvel.copy()[3])
        state.append(self.data.qvel.copy()[4])
        state.append(self.data.qvel.copy()[5])
        state.append(self.data.time)



        r = -1 * d
        done = False
        truncated = False

        if self.render : 
            print(r)
        
        if d < 0.005:
            r = 10.0
            done = True
            
        if done and self.render: 
            time.sleep(1)
        

        self.count += 1
        if self.count >= 100 or done:
            truncated = True 
        info = 0
        return np.array(state) , r , done ,truncated , info


