# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 17:02:20 2022

@author: scottjhy
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Distribution, Normal
from torch.distributions.categorical import Categorical
from torch.distributions.one_hot_categorical import OneHotCategorical
from torch.nn import init
import numpy as np
from torch.autograd import Variable

def masked_softmax(vec, mask,dim=-1):
    masked_vec = vec * mask.float()
    ## choose the max in masked_vec exculding those indices not masked
    with torch.no_grad():
        tmp_mask=1.0/mask.float()
        # print(tmp_mask)
        masked_max_vec=vec+(mask.float()-tmp_mask*torch.sign(tmp_mask))
        # print(masked_max_vec)
        max_vec = torch.max(masked_max_vec, dim=dim, keepdim=True)[0]
        if max_vec.cpu()[0].numpy()==torch.tensor([float("inf")]):
            print("wrong!")
            print("wrong!")
            print("wrong!")
            print("vec is",vec)
            print("mask is",mask)
        # print(max_vec)
    # print(max_vec.size())
    
    exps = torch.exp((masked_vec-max_vec)*mask.float())
    masked_exps = (exps) * mask.float()
    masked_sums = masked_exps.sum(dim, keepdim=True)
    zeros=(masked_sums == 0)
    masked_sums += zeros.float()
    return masked_exps/(masked_sums)

class policy_network(nn.Module):
    def __init__(self,input_num,output_num,n_hid,n_layers,rest=12,num_freq=4,dropout=0.0):
        ## input_num = state_dim + action_dim (800 + 360)
        ## output_num = action_dim (360)
        ## n_hid = 256
        ## n_layers = 3
        
        super(policy_network, self).__init__()
        self.input_num=input_num
        self.output_num=output_num
        self.num_freq=num_freq
        self.n_hid=n_hid
        self.rest=rest
        self.n_layers=n_layers
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gru=nn.GRU(n_hid, n_hid, n_layers, dropout=dropout)
        self.eps=0.005
        ## MLP for s_t to x_t(GRU current information)
        self.linear1=torch.nn.Linear(input_num+self.rest,n_hid)
        self.linear1_1=torch.nn.Linear(n_hid,n_hid)
        
        ## MLP for h_t to a_t(angle)
        self.linear2=torch.nn.Linear(n_hid,n_hid)        
        self.linear_out=torch.nn.Linear(n_hid,output_num)
        
        ## from angle and hidden state to freq
        self.linear1_freq=torch.nn.Linear(n_hid+self.output_num,n_hid)
        self.linear1_1_freq=torch.nn.Linear(n_hid,n_hid)
        self.linear_out_freq=torch.nn.Linear(n_hid,num_freq)
        
        
        ## what's this part?
        self.in1=torch.nn.InstanceNorm1d(n_hid)
        self.in1_1=torch.nn.InstanceNorm1d(n_hid)
        self.in2=torch.nn.InstanceNorm1d(n_hid)
        self.in1_freq=torch.nn.InstanceNorm1d(n_hid)
        self.in1_1_freq=torch.nn.InstanceNorm1d(n_hid)
        self.init_weights()
        
        ## note the starting position of used angle distribution
        self.start_act=self.input_num-self.output_num
        self.noise=1e-1

    def init_weights(self):
        torch.nn.init.orthogonal_(self.linear1.weight)
        torch.nn.init.constant_(self.linear1.bias, 0.1)
        torch.nn.init.orthogonal_(self.linear1_1.weight)
        torch.nn.init.constant_(self.linear1_1.bias, 0.1)
        torch.nn.init.orthogonal_(self.linear2.weight)
        torch.nn.init.constant_(self.linear2.bias, 0.1)
        torch.nn.init.orthogonal_(self.linear_out.weight)
        torch.nn.init.constant_(self.linear_out.bias, 0.1)
        torch.nn.init.orthogonal_(self.linear1_freq.weight)
        torch.nn.init.constant_(self.linear1_freq.bias,0.1)
        torch.nn.init.orthogonal_(self.linear_out_freq.weight)
        torch.nn.init.constant_(self.linear_out_freq.bias,0.1)
        torch.nn.init.orthogonal_(self.linear1_1_freq.weight)
        torch.nn.init.constant_(self.linear1_1_freq.bias,0.1)
        for name, param in self.gru.named_parameters():
            if 'bias' in name:
                torch.nn.init.constant_(param, 0.1)
            elif 'weight' in name:
                torch.nn.init.orthogonal_(param)
    
    def init_hidden(self, batch_size):
        ## initial size is (3 * batch_size(1) * 256)
        return torch.zeros(self.n_layers, batch_size, self.n_hid).to(self.device)

    
    ## from h_{t-1} and s_t to h_t(hidden2) and a_t(angle)(out)
    def forward(self,state,hidden):
        out=torch.relu(self.in1(self.linear1(state)))
        out=torch.relu(self.in1_1(self.linear1_1(out)))
        rnn_out,hidden2=self.gru(out,hidden)
        out=torch.relu(self.in2(self.linear2(rnn_out)))
        out=self.linear_out(out)
        # out1=torch.softmax(out,dim=2)
        return out,rnn_out,hidden2
    
    ## get freq distribution
    def forward2(self,out,one_hot):
        input=torch.cat([out,one_hot],dim=2)
        out1=torch.relu(self.in1_freq(self.linear1_freq(input)))
        out1=torch.relu(self.in1_1_freq(self.linear1_1_freq(out1)))
        out1=self.linear_out_freq(out1)
        '''batch_size=out1.size()[1]
        mean=out1[:,:,0].view(-1,batch_size)
        std=torch.exp(out1[:,:,1])#+self.noise
        std=std.view(-1,batch_size)'''
        return out1
    
    
    ## compute action
    def act(self,state,hidden,choose_act,deterministic=False):
        ## compute angle distribution and sample
        out1,rnn_out,_=self.forward(state,hidden)
        mask=torch.zeros(len(choose_act)).to(self.device)
        mask[abs(choose_act)<1e-8]=1.0
        out1=masked_softmax(out1,mask)
        out1+=0.1*mask*self.eps
        out1=out1/torch.sum(out1,dim=2,keepdim=True)
        dist1=Categorical(out1)
        angle=dist1.sample()
        #print(angle)
        
        ## compute freq distribution
        angle_hot=F.one_hot(angle,self.output_num).type_as(rnn_out).detach()
        out2=self.forward2(rnn_out,angle_hot)
        mask2=torch.ones(self.num_freq).to(self.device)
        out2=masked_softmax(out2,mask2)
        out2+=1/5*self.eps
        dist2=Categorical(out2)
        freq=dist2.sample()
        if deterministic:
            angle=out1[0,0,:].sort(descending=True)[1][0]
            '''if torch.cuda.is_available():
                angle=angle.cpu()
                mean=mean.cpu()'''
            freq=out2[0,0,:].sort(descending=True)[1][0]
            if torch.cuda.is_available():
                angle=angle.cpu()
                freq=freq.cpu()
            return angle.numpy().reshape([1,]),freq.numpy().reshape([1,])[0],out2

        if torch.cuda.is_available():
            angle=angle.cpu()
            freq=freq.cpu()
        return angle.numpy()[0],freq.numpy()[0][0],out2


    def prob(self,state,hidden,act,freq):
        ## state = (collected new data(800), used angle distibution(360), used doze(1))
        
        ## from h_{t-1} and s_t to h_t(rnn_out) and a_t(angle)(out)
        out1,rnn_out,hidden2=self.forward(state,hidden)
        batch_size=out1.size()[1]
        
        ## get used angles and mask the used position as 0
        one_hot=state[:,:,self.start_act:-self.rest]
        mask=torch.zeros(one_hot.size()).to(self.device)
        mask[abs(one_hot)<1e-8]=1
        # out1=mask*out1+1e-10
        # p=torch.sum(out1,dim=2).unsqueeze(2)
        # out1=out1/p
        # out1=torch.sum(act*out1,dim=2)
        
        
        out1=masked_softmax(out1,mask)
        out1+=0.1*mask*self.eps
        out1=out1/torch.sum(out1, dim=2,keepdim=True)
        outp1=out1
        out1=torch.sum(act*out1,dim=2)
        out2=self.forward2(rnn_out,act)
        mask2=torch.ones(out2.size()).to(self.device)
        out2=masked_softmax(out2,mask2)
        out2+=1/5*self.eps
        out2 = out2/torch.sum(out2, dim=2,keepdim=True)
        outp2 = out2
        ss = freq.size()
        f=torch.zeros([ss[0],batch_size,self.num_freq]).to(self.device)
        for i in range(batch_size):
            for j in range(ss[0]):
                f[j,i,freq[j,i,0].cpu().numpy()]=1
        out2=torch.sum(f*out2,dim=2)
        log_p1=(out1+1e-10).log()
        log_p2=(out2+1e-10).log()
        p1=out1
        p2=out2
        
        ## p1 = p_{\pi_{\theta}}(angle | state), p2 is p_{\pi_{\theta}}(freq | state)
        return p1,p2,log_p1,log_p2,hidden2,outp1,outp2



class critic_network(nn.Module):
    def __init__(self,input_num,n_hid,n_layers,rest=12,dropout=0.0):
        super(critic_network, self).__init__()
        self.input_num=input_num
        self.n_hid=n_hid
        self.rest=rest
        self.n_layers=n_layers
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gru=nn.GRU(n_hid, n_hid, n_layers, dropout=dropout)
        self.linear1=nn.Linear(input_num+self.rest,n_hid)
        self.linear1_1=nn.Linear(n_hid,n_hid)
        self.in1=torch.nn.InstanceNorm1d(n_hid)
        self.in1_1=torch.nn.InstanceNorm1d(n_hid)
        self.in2=torch.nn.InstanceNorm1d(n_hid)
        self.linear2=nn.Linear(n_hid,n_hid)
        self.linear_out=nn.Linear(n_hid,1)
        self.init_weights()

    def init_weights(self):
        torch.nn.init.orthogonal_(self.linear1.weight)
        torch.nn.init.constant_(self.linear1.bias, 0.0)
        torch.nn.init.orthogonal_(self.linear1_1.weight)
        torch.nn.init.constant_(self.linear1_1.bias, 0.0)
        torch.nn.init.orthogonal_(self.linear2.weight)
        torch.nn.init.constant_(self.linear2.bias, 0.0)
        torch.nn.init.orthogonal_(self.linear_out.weight)
        torch.nn.init.constant_(self.linear_out.bias, 0.0)
        for name, param in self.gru.named_parameters():
            if 'bias' in name:
                torch.nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                torch.nn.init.orthogonal_(param)
        
        

    def init_hidden(self, bsz):
        #weight = next(self.parameters())
        return torch.zeros(self.n_layers, bsz, self.n_hid).to(self.device)

    def forward(self,state,hidden):
        #print("state size{}".format(state.size()))
        batch_size=state.size()[1]
        out=torch.relu(self.in1(self.linear1(state)))
        out=torch.relu(self.in1_1(self.linear1_1(out)))
        out,hidden2=self.gru(out,hidden)
        out=torch.relu(self.in2(self.linear2(out)))
        out=self.linear_out(out).view(-1,batch_size)
        return out,hidden2