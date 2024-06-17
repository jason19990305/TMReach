Certainly! Here's the translated version in English for your GitHub README.md:

# TMReach DDPG

## Overview

TMReach DDPG is an implementation of the Deep Deterministic Policy Gradient (DDPG) algorithm for controlling the TM Robot tm12s robotic arm. Below are the key components and files associated with this project:

1. **DDPG (Deep Deterministic Policy Gradient):**
   - DDPG is an off-policy reinforcement learning algorithm that combines elements of both actor-critic methods.
   - It is particularly well-suited for continuous action spaces.
   - The main architecture consists of an actor network (policy) and a critic network (value function).

2. **Model:**
   - The `model` file contains the neural network parameters (usually saved as a `.pt` file).
   - These parameters define the policy and value functions used by the DDPG algorithm.

3. **TM12s Mujoco Model:**
   - The `TM12s` directory contains the Mujoco XML file (`tm12s.xml`) that describes the TM Robot tm12s arm.
   - Additionally, it includes `.obj` and `.mtl` files representing the visual components of the arm.
   - This Mujoco model allows simulation of the robot arm's dynamics.

4. **TMReach.py:**
   - The `TMReach.py` script is the main program for setting training parameters and running the DDPG training process.
   - It configures hyperparameters, initializes the actor and critic networks, and manages training episodes.

5. **TMReach.yaml:**
   - The `TMReach.yaml` file stores parameters related to state normalization during training.
   - State normalization helps stabilize training by scaling state features appropriately.

6. **TMReach_Run.py:**
   - The `TMReach_Run.py` script allows you to load a trained model and perform simulations.
   - You can use this script to evaluate the trained policy on the TM Robot tm12s arm.

## Mujoco Model Details

- The Mujoco model for the TM Robot tm12s arm is constructed based on resources from the TM Robot ROS GitHub repository.
- ROS tools and the `obj2mjcf` utility are used to process URDF-format components and convert obj models into Mujoco simulations.
- Actuators are added to the model to allow programmatic control of the arm's joints.
- The reward function is computed based on the distance between the desired goal and the achieved goal during training.

## DDPG Algorithm

- DDPG uses an actor-critic architecture:
  - The actor (policy) network learns the optimal policy for selecting actions.
  - The critic (value function) network estimates the value of a given state-action pair.
- Soft updates are used to update the target networks gradually.
- Additional parameters include `use_state_norm` (for state normalization) and `tau` (used in soft updates).

## Usage

- To train the DDPG model, modify the parameters in `TMReach.py` and run the script.
- After training, you can use `TMReach_Run.py` to load the trained model and perform simulations.

![](https://github.com/jason19990305/TMReach/blob/main/TMreach.gif)
![](https://github.com/jason19990305/TMReach/blob/main/Reward.png)