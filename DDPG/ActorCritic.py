import torch.nn.functional as F 
import torch.nn as nn 
import torch

def orthogonal_init(layer, gain=1.0): # Trick : Orthogonal Initialization
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)

class Actor(nn.Module):
    def __init__(self,args,hidden_layers=[64,64]):
        super(Actor, self).__init__()
        self.num_states = args.num_states
        self.num_actions = args.num_actions
        # add in list
        hidden_layers.insert(0,self.num_states) # first layer
        hidden_layers.append(self.num_actions) # last layer
        print(hidden_layers)

        # create layers
        layer_list = []
        for i in range(len(hidden_layers)-1):
            input_num = hidden_layers[i]
            output_num = hidden_layers[i+1]
            layer = nn.Linear(input_num,output_num)
            
            layer_list.append(layer)
            orthogonal_init(layer_list[-1]) 
        orthogonal_init(layer_list[-1], gain=0.01)

        # put in ModuleList
        self.layers = nn.ModuleList(layer_list)
        self.tanh = nn.Tanh()

    # when actor(s) will activate the function 
    def forward(self,s):

        for layer in self.layers:
            s = self.tanh(layer(s))
        return s
    


class Critic(nn.Module):
    def __init__(self, args,hidden_layers=[64,64]):
        super(Critic, self).__init__()
        self.num_states = args.num_states
        self.num_actions = args.num_actions
        # add in list
        hidden_layers.insert(0,self.num_states+self.num_actions)
        hidden_layers.append(1)
        print(hidden_layers)

        # create layers
        layer_list = []

        for i in range(len(hidden_layers)-1):
            input_num = hidden_layers[i]
            output_num = hidden_layers[i+1]
            layer = nn.Linear(input_num,output_num)
            
            layer_list.append(layer)
            orthogonal_init(layer_list[-1])
        # put in ModuleList
        self.layers = nn.ModuleList(layer_list)
        self.tanh = nn.Tanh()

    def forward(self,s,a):
        input_data = torch.cat((s,a),dim=1)
        for i in range(len(self.layers)-1):
            input_data = self.tanh(self.layers[i](input_data))

        # predicet value
        v_s = self.layers[-1](input_data)
        return v_s
    