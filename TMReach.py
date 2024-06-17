
import argparse
import numpy as np 
from datetime import datetime


from torch.distributions import Normal
import torch.nn.functional as F # for mean square error
import torch.nn as nn # for neural network

from DDPG.Agent import Agent
from TM12s.Robot import RobotReach



class main():
    def __init__(self,args):
        env = RobotReach(render = False)    # The wrapper encapsulates the gym env
        num_actions = env.num_actions
        num_states = env.num_states
        print(num_actions)
        print(num_states)
        # args
        args.num_actions = num_actions
        args.num_states = num_states

        # print args 
        print("---------------")
        for arg in vars(args):
            print(arg,"=",getattr(args, arg))
        print("---------------")

        # create agent
        hidden_layer_num_list = [128,128,128,128]
        agent = Agent(args,env,hidden_layer_num_list)

        # trainning
        agent.train() 
        
        # saving parameter
        agent.state_norm.save_yaml(args.env_name+".yaml")
        now = datetime.now()
        path_actor = "model/DDPG_Actor_"+args.env_name + now.strftime("%Y%m%d_%H%M") + ".pt"
        path_critic = "model/DDPG_Critic_"+args.env_name + now.strftime("%Y%m%d_%H%M") +".pt"
        agent.save_actor_model(path_actor)
        agent.save_critic_model(path_critic)

        # evaluate 
        env_evaluate = RobotReach(render = True)
        
        for i in range(10000):
            evaluate_reward = agent.evaluate_policy(env_evaluate)
            print("Evaluate reward:",evaluate_reward)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameters Setting for DDPG")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate of actor")
    parser.add_argument("--var", type=float, default=3, help="Normal noise var")
    parser.add_argument("--tau", type=float, default=0.001, help="Parameter for soft update")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--mem_min", type=float, default=64, help="minimum size of replay memory before updating actor-critic.")
    parser.add_argument("--env_name", type=str, default='TMReach', help="Enviroment name")

    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--buffer_size", type=int, default=int(10000), help="Learning rate of actor")
    parser.add_argument("--use_state_norm", type=bool, default=True, help="Using state normalization.")
    parser.add_argument("--max_train_steps", type=int, default=int(3e5), help=" Maximum number of training steps")
    parser.add_argument("--evaluate_freq_steps", type=float, default=2e3, help="Evaluate the policy every 'evaluate_freq' steps")
    args = parser.parse_args()

    
    main(args)