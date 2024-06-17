import time
import mujoco
import mujoco.viewer
import matplotlib.pyplot as plt

import numpy as np


duration = 20  # (seconds)

model = mujoco.MjModel.from_xml_path('scene_reach.xml') # load xml model



data = mujoco.MjData(model)

viewer = mujoco.viewer.launch_passive(model,data) # Get viewer

viewer.cam.distance = model.stat.extent * 2 # set camera distance

mujoco.mj_resetData(model, data)  # Reset state and time.

renderer = mujoco.Renderer(model)

scene = renderer.scene



sensor_data = []
angle1 = []
angle2 = []
angle3 = []
angle4 = []
angle5 = []
angle6 = []


count = 0
while data.time < duration:
    mujoco.mj_step(model, data)

    sensor_data.append(data.sensor('accelerometer').data.copy())
    angle1.append(data.joint('joint_1').qpos.copy())
    angle2.append(data.joint('joint_2').qpos.copy())
    angle3.append(data.joint('joint_3').qpos.copy())
    angle4.append(data.joint('joint_4').qpos.copy())
    angle5.append(data.joint('joint_5').qpos.copy())
    angle6.append(data.joint('joint_6').qpos.copy())
    data.ctrl[0] += 0.001

    print(data.joint('joint_1').qpos.copy()[0])

    #viewer.sync()

sensor_data = np.array(sensor_data)
angle1 = np.array(angle1)
angle2 = np.array(angle2)
angle3 = np.array(angle3)
angle4 = np.array(angle4)
angle5 = np.array(angle5)
angle6 = np.array(angle6)
print(sensor_data.shape)
plt.figure(1)
plt.plot(sensor_data[:,0] , label = 'x')
plt.plot(sensor_data[:,1] , label = 'y')
plt.plot(sensor_data[:,2] ,label = 'z')
plt.legend()
plt.title('Accelerometer values')


print(angle1.shape)
plt.figure(2)
plt.plot(angle1 , label = 'joint_1')
plt.plot(angle2 , label = 'joint_2')
plt.plot(angle3 , label = 'joint_3')
plt.plot(angle4 , label = 'joint_4')
plt.plot(angle5 , label = 'joint_5')
plt.plot(angle6 , label = 'joint_6')
plt.legend()
plt.title('Joint Angle')



plt.show()