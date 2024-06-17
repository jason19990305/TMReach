
from torch.utils.tensorboard import SummaryWriter 
from torch.distributions import Normal
import torch.nn.functional as F
import numpy as np 

import datetime
import random
import torch
import time
import copy
import os




# Custom class
from DDPG.ReplayBuffer import ReplayBuffer
from DDPG.Normalization import Normalization
from DDPG.ActorCritic import Actor , Critic



class Agent():
    def __init__(self,args,env,hidden_layer_num_list=[64,64]):

        # Hyperparameter
        self.evaluate_freq_steps = args.evaluate_freq_steps
        self.max_train_steps = args.max_train_steps
        self.use_state_norm = args.use_state_norm
        self.num_actions = args.num_actions
        self.batch_size = args.batch_size
        self.num_states = args.num_states
        self.mem_min = args.mem_min
        self.gamma = args.gamma
        self.set_var = args.var
        self.var = self.set_var
        self.tau = args.tau
        self.lr = args.lr

        # variable
        self.total_steps = 0
        self.training_count = 0
        self.episode_steps = 0
        self.evaluate_count = 0
        self.replace_counter = 0 
        self.evaluate_rewards = []

        # other
        self.env = env
        self.env_name = args.env_name
        self.replay_buffer = ReplayBuffer(args)
        self.state_norm = Normalization(shape=self.num_states)  # Trick :state normalization
        home_directory = os.path.expanduser( '~' )
        log_dir=home_directory+'/Log/DDPG_'+args.env_name+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.writer = SummaryWriter(log_dir=log_dir)
        self.random_seed = random.random()



        # Actor-Critic
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.actor = Actor(args,hidden_layer_num_list.copy()).to(self.device)
        self.critic = Critic(args,hidden_layer_num_list.copy()).to(self.device)
        self.actor_target =  Actor(args,hidden_layer_num_list.copy()).to(self.device)
        self.critic_target =  Critic(args,hidden_layer_num_list.copy()).to(self.device)
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr, eps=1e-5)
        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr, eps=1e-5)

        print(self.actor)
        print(self.critic)
        print("Log path : ",log_dir)
        print("-----------")

    def choose_action(self,state):

        state = torch.tensor(state, dtype=torch.float)

        s = torch.unsqueeze(state,0).to(self.device)
        with torch.no_grad():
            a = self.actor(s)
            a = Normal(a , abs(self.var) + 1e-4).sample()
            
            a = torch.clamp(a,-1,1)
        return a.cpu().numpy().flatten()

    def evaluate_action(self,state):

        state = torch.tensor(state, dtype=torch.float)
        s = torch.unsqueeze(state,0).to(self.device)

        with torch.no_grad():
            a = self.actor(s)     
        return a.cpu().numpy().flatten()

    def evaluate_policy(self, env , render = False):
        times = 10
        evaluate_reward = 0
        for i in range(times):
            s = env.reset()
            if self.use_state_norm:
                s = self.state_norm(s, update=False)  # During the evaluating,update=False
            done = False
            episode_reward = 0
            episode_steps = 0
            while True:
                a = self.evaluate_action(s)  # We use the deterministic policy during the evaluating
                
                s_, r, done, truncted, _ = env.step(a)

                if self.use_state_norm:                
                    s_ = self.state_norm(s_, update=False)
                episode_reward += r
                s = s_
                #print(episode_reward)
                if truncted or done:
                    break
                episode_steps += 1
            evaluate_reward += episode_reward

        return evaluate_reward / times


    def train(self):
        time_start = time.time()
        episode_count = 0
        while self.total_steps < self.max_train_steps:
            s = self.env.reset()

            if self.use_state_norm:
                s = self.state_norm(s) # Trick : state Normalization

            while True:
                a = self.choose_action(s)
                s_, r, done , truncated , _ = self.env.step(a)
                if self.use_state_norm:
                    s_ = self.state_norm(s_) # Trick : state Normalization

                # storage data
                self.replay_buffer.store(s, a, [r], s_, done)
                
                # update state
                s = s_

                if self.replay_buffer.count >= self.mem_min:
                    self.training_count += 1
                    self.update()

                if self.total_steps % self.evaluate_freq_steps == 0:
                    self.evaluate_count += 1
                    evaluate_reward = self.evaluate_policy(self.env)
                    self.evaluate_rewards.append(evaluate_reward)
                    evaluate_average_reward = np.mean(self.evaluate_rewards[-50:])
                    time_end = time.time()
                    h = (time_end - time_start) / (60 * 60)
                    m = (time_end - time_start) / (60) - h * 60
                    second = (time_end - time_start) - (h * 60 + m) * 60 
                    print("---------")
                    print("Time : %02d:%02d:%02d"%(h,m,second))
                    print("Training episode : %d\tStep : %d / %d"%(episode_count,self.total_steps,self.max_train_steps))
                    print("Evaluate count : %d\tEvaluate reward : %0.2f\tAverage reward : %0.2f"%(self.evaluate_count,evaluate_reward,evaluate_average_reward))
                    self.writer.add_scalar('step_success_rate_{}'.format(self.env_name), self.evaluate_rewards[-1], global_step=self.total_steps)

                self.total_steps += 1
                if done or truncated:
                    break
            episode_count += 1

    def update(self):
        s, a, r, s_, done = self.replay_buffer.numpy_to_tensor()  # Get training data .type is tensor    

        index = np.random.choice(len(r),self.batch_size,replace=False)

        minibatch_s = s[index].to(self.device)
        minibatch_a = a[index].to(self.device)
        minibatch_r = r[index].to(self.device)
        minibatch_s_ = s_[index].to(self.device)
        minibatch_done = done[index].to(self.device)

        # update Actor
        action = self.actor(minibatch_s)
        value = self.critic(minibatch_s,action)
        actor_loss = -torch.mean(value)
        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5) # Trick : Clip grad
        self.optimizer_actor.step()


        # Update Critic
        next_action = self.actor_target(minibatch_s_)
        next_value = self.critic_target(minibatch_s_,next_action)
        v_target = minibatch_r + self.gamma * next_value * (1 - minibatch_done)

        value = self.critic(minibatch_s,minibatch_a)
        critic_loss = F.mse_loss(value,v_target)
        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5) # Trick : Clip grad
        self.optimizer_critic.step()


        self.lr_decay(total_steps=self.total_steps)
        self.var_decay(total_steps=self.total_steps)

        self.soft_update(self.critic_target,self.critic, self.tau)
        self.soft_update(self.actor_target, self.actor, self.tau)    

    def soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
            
    def var_decay(self, total_steps):
        new_var = self.set_var * (1 - total_steps / self.max_train_steps)
        self.var = new_var + 10e-10
        
    def lr_decay(self, total_steps):
        lr_a_now = self.lr * (1 - total_steps / self.max_train_steps)
        lr_c_now = self.lr * (1 - total_steps / self.max_train_steps)
        for opt in self.optimizer_actor.param_groups:
            opt['lr'] = lr_a_now
        for p in self.optimizer_critic.param_groups:
            opt['lr'] = lr_c_now
            
    def save_actor_model(self,path):
        print("Save actor model:",path)
        torch.save(self.actor, path)
        
    def save_critic_model(self,path):
        print("Save critic model:",path)
        torch.save(self.critic, path)
        
    def load_actor_model(self,path):
        print("Load actor model:",path)
        self.actor = torch.load(path).train()
        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr, eps=1e-5)
        
    def load_critic_model(self,path):
        print("Load critic model:",path)
        self.critic = torch.load(path).train()
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr, eps=1e-5)