# -*- coding: utf-8 -*-
"""
Created on Sat Aug 12 19:38:47 2023

@author: scott
"""

import torch
import torch.nn as nn
import torch.optim as optim
import math
import numpy as np
import random
from torch_network_1freq import policy_network as policy_network
from torch_network_1freq import critic_network as critic_network
import torch.nn.functional as F
import copy
import torch.multiprocessing as mp
from pde_model_gen import *
import random
from utils import MemorySlide
import matplotlib.pyplot as plt
#import ct_animation2
from utils import *
import pickle
import time

'''run an episode and record all the state, action and rewards'''

def set_seed(seed_value=42):
    """Set seed for reproducibility.
    """
    # Python
    random.seed(seed_value)
    # NumPy
    np.random.seed(seed_value)
    # PyTorch
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

    # Ensure that randomness is reasonably consistent
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def run_env(ppo,rest,ep):
    ppo.start_epoch(1)
    obs,_=ppo.env.reset(rest=rest)
    total_reward=0
    total_reward2=0
    total_reward3=0
    I=1
    time_long=0
    act=np.zeros((ppo.action_dim,))
    old_act=np.zeros((ppo.action_dim,))
    history=[]
    r = rest
    for i in range(ppo.long_time):
        with torch.no_grad():
            if i!=0:
                obs_norm=np.clip((obs-ppo.statestat.mean)/(ppo.statestat.std+1e-8),-40,40)
            else:
                obs_norm=obs
                
            obs1=np.concatenate([obs_norm,np.zeros([ppo.state_dim-len(obs_norm),]),old_act],axis=0)
            ## compute the new action
            angle,freq,out2=ppo.action(obs1,r,old_act)
            ppo.freqprob.append(out2)
            r -= 1
            next_obs,reward,done,_=ppo.env.step(angle,freq)
            if i==ppo.long_time-1:
                #reward=ppo.env.show_psnr()
                done=True
                
            time_long+=1
            if ppo.reward_scale:
                reward1=np.clip(reward/(ppo.runningstat.std+1e-8),-40,40)
            else:
                reward1=reward
            ## store next observation and action
            act[angle]+=freq+1
            rest0=np.zeros([ppo.rest])
            rest0[r]=1
            rest1=np.zeros([ppo.rest])
            rest1[r-1]=1
            next_obs_norm=np.clip((next_obs-ppo.statestat.mean)/(ppo.statestat.std+1e-8),-40,40)
            next_obs1=np.concatenate([next_obs_norm,np.zeros([ppo.state_dim-len(next_obs_norm),]),act],axis=0)
            p1,p2,advantage,value=ppo.process(obs1,r+1,angle,freq,next_obs1,r,reward1,done)
            ppo.next_hidden(obs1,r+1)
            history.append(((obs1.copy(),rest0),(angle,freq,p1,p2,advantage,value,reward,obs.copy()),reward1,done,(next_obs1.copy(),rest1)))
            old_act[angle]+=freq+1
            total_reward+=(I*reward)
            total_reward2+=reward
            total_reward3+=(I*reward1)
            I*=ppo.gamma
            obs=next_obs
            if done:
                break
        torch.cuda.empty_cache()
        
    ## compute the time of repeat action
    repeat=old_act-1
    repeat=repeat.clip(0,200)
    repeat=repeat.sum()
    return history,(total_reward,total_reward2,total_reward3,time_long,ppo.policy_net.noise,repeat,psnr2(ppo.env.true_img,0)+total_reward)


class PPO_RB():
    def __init__(self,size,rest,freq_bound,sample_size,state_dim,action_dim,hidden_cell,n_layers,num_freq=4,method="ridge",actor_lr=1e-4,
                 critic_lr=1e-4,gamma=1,use_linear_lr_decay=True,shape="oval",save_path='../ctmodel/',
                 load_path='../ctmodel/',max_process=2,max_length=100,sensor_pos="circle"):
        self.reward_scale=None
        self.state_dim=state_dim
        self.action_dim=action_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net=policy_network(self.state_dim+self.action_dim, self.action_dim,hidden_cell[0],n_layers[0],num_freq=num_freq,rest=rest).to(self.device)
        self.critic_net=critic_network(self.state_dim+self.action_dim,hidden_cell[1],n_layers[1],rest=rest).to(self.device)
        self.target_policy_net=policy_network(self.state_dim+self.action_dim, self.action_dim,hidden_cell[0],n_layers[0],num_freq=num_freq,rest=rest).to(self.device)
        self.target_critic_net=critic_network(self.state_dim+self.action_dim,hidden_cell[1],n_layers[1],rest=rest).to(self.device)
        self.gamma=1
        self.shape =shape
        self.freq_bound=freq_bound
        self.rest=rest
        self.freqprob=[]
        self.actor_lr=actor_lr
        self.critic_lr=critic_lr
        self.batch=10
        self.train_ep=5
        self.num_process=8
        self.epsilon=2
        self.buffer_size=1000
        self.replay_buffer=MemorySlide(self.buffer_size)
        self.long_time=max_length
        self.max_length=max_length
        self.save_path=save_path
        self.freq=[]
        self.sensor_pos=sensor_pos
        img_path='../ct_data'
        self.env=IS(size,rest,num_freq,sample_size,shape=shape,sensor_pos=self.sensor_pos)
        self.runningstat=RunningStat(())
        '''stat_fp=open('./obs_stat.txt','rb')
        self.statestat=pickle.load(stat_fp)'''
        self.statestat=RunningStat(())
        print(f"gamma:{self.gamma}actor lr:{self.actor_lr},critic lr:{self.critic_lr}")
        self.value_clip = True
        self.psnr=[]
        self.adv_scale = True
        self.use_grad_clip = True
        self.use_linear_lr_decay=use_linear_lr_decay
        self.max_grad_norm = 1.0
        self.policy_net.share_memory()
        self.target_policy_net.share_memory()
        self.critic_net.share_memory()
        self.target_critic_net.share_memory()
        self.target_policy_net.load_state_dict(self.policy_net.state_dict())
        self.target_critic_net.load_state_dict(self.critic_net.state_dict())
        self.policy_net.noise=float(0)
        self.target_policy_net.noise=float(0)
        self.actor_optimizer = torch.optim.Adam(self.policy_net.parameters(), self.actor_lr,betas=(0.9,0.999),amsgrad=True)
        self.critic_optimizer=torch.optim.Adam(self.critic_net.parameters(),self.critic_lr,betas=(0.9,0.999),amsgrad=True)
        if use_linear_lr_decay:
            self.actor_scheduler=optim.lr_scheduler.StepLR(self.actor_optimizer,80,0.99)
            self.critic_scheduler=optim.lr_scheduler.StepLR(self.critic_optimizer,80,0.99)
            #self.actor_scheduler=optim.lr_scheduler.CosineAnnealingLR(self.actor_optimizer,200,1e-5)
            #self.critic_scheduler=optim.lr_scheduler.CosineAnnealingLR(self.critic_optimizer,200,1e-4)    
    
    '''run multiple episodes at the same time and train networks'''
    def multi_train(self,all_ep):
        ## save rewards, repeat times, time length for multi episodes
        avg_rew=[]
        avg_repeat=[]
        avg_timelen=[]
        for ep in range(all_ep):
            ## run 12 epsodes at the same time and save results
            psnr_sum = 0
            for i in range(6):
                history,info=run_env(self,self.rest,ep)
                history=self.gae(history,0.995)
                avg_rew.append(info[0])
                #print(f"epoch {ep}:reward {info[0]},noise level:{info[4]},time length {info[3]},non-discount:{info[1]},scale reward:{info[2]},repeat:{info[5]},psnr:{info[6]}")
                print(f"epoch {ep}:psnr:{info[6]}")
                avg_timelen.append(info[3])
                self.replay_buffer.insert(history)
                psnr_sum += info[6]/8
            self.psnr.append(psnr_sum)
            self.policy_net.eps = self.policy_net.eps*0.995
            torch.cuda.empty_cache()
            
            for i in range(self.train_ep):
                self.train(min(len(self.replay_buffer.buffer),self.batch))
            self.replay_buffer.clear()
            if ep%10==0:
                avg_len=int(np.mean(np.array(avg_timelen)))
                #self.test_train(avg_len)
                #print(self.avg_H)
                #print(f"avg entropy:{np.mean(np.array(self.avg_H))}")
                #print(f"avg time-len:{avg_len}")
                avg_timelen=[]
                self.avg_H=[]
                #print(f"avg reward:{np.mean(np.array(avg_rew))}")
                #print(f"learning rate:actor:{self.actor_optimizer.param_groups[0]['lr']},critic:{self.critic_optimizer.param_groups[0]['lr']}")
                self.save(self.save_path,ep)
                avg_rew=[]
                avg_repeat=[]

    '''set up the innitial hidden state '''
    def start_epoch(self,batch_size):
        start_a_hidden=self.policy_net.init_hidden(batch_size)
        start_v_hidden=self.critic_net.init_hidden(batch_size)
        start_a_hidden_t=self.target_policy_net.init_hidden(batch_size)
        start_v_hidden_t=self.target_critic_net.init_hidden(batch_size)
        self.hidden_v_pre=start_v_hidden
        self.hidden_a_pre=start_a_hidden
        self.hidden_a_pre_t=start_a_hidden_t
        self.hidden_v_pre_t=start_v_hidden_t

    def gae(self,history,lam):
        ## history = ((state),(angle,freq,p1,p2,advantage,value,reward,obs),reward1,done,(next_state))
        lens=len(history)
        result_history=[]
        gae=0
        rs=0
        j=0
        for i in reversed(range(lens)):
            ## rs = sum \gamma^{k}Value(s_k)
            rs=self.gamma*rs+history[i][1][5]
            j+=1
            self.runningstat.push(rs)
            observe=history[i][1][7].tolist()
            for data in observe:
                self.statestat.push(data)
            
            ## delta = advantage = Q(s,a) - V(s)
            delta=history[i][1][4]
            
            ## gae = advantage + \gamma * 0.96 * gae - c * log(p(a|s))      (here a = angle + doze)
            gae=delta+self.gamma*lam*gae
            result_history.append((history[i][0],(history[i][1][0],history[i][1][1],history[i][1][2],history[i][1][3],
                                   gae,history[i][1][5],history[i][1][6]),history[i][2],history[i][3],history[i][4]))
        ## result_history = (state,(angle,freq,p1,p2,gae,value,reward),reward1,done,next_state)
        #print(result_history)
        result_history.reverse()
        #print(result_history)
        return result_history

    '''use policy network and target critic network to compute estimated advantage and value '''
    def process(self,obs,r1,angle,freq,next_obs,r2,reward,if_end):
        #self.policy_net.resample()
        self.policy_net.train()
        self.target_policy_net.eval()
        self.target_critic_net.eval()
        
        ## resize the state and action information
        s = torch.tensor(obs, dtype=torch.float,device=self.device).view(-1,1,self.state_dim+self.action_dim)
        rest1=torch.zeros([1,1,self.rest],dtype=torch.float,device=self.device)
        rest1[:,:,r1-1]=1
        #rest1=torch.tensor([r1],dtype=torch.float,device=self.device).view(-1,1,1)
        s2 = torch.tensor(next_obs, dtype=torch.float,device=self.device).view(-1,1,self.state_dim+self.action_dim)
        rest2=torch.zeros([1,1,self.rest],dtype=torch.float,device=self.device)
        rest2[:,:,r2-1]=1
        #rest2=torch.tensor([r2],dtype=torch.float,device=self.device).view(-1,1,1)
        state=torch.cat([s,rest1],dim=2)
        next_state=torch.cat([s2,rest2],dim=2)
        angle1=torch.zeros(1,1,self.action_dim)
        angle1[0,0,angle]=1.0
        angle1=angle1.to(self.device)
        freq=torch.tensor([freq],dtype=torch.float,device=self.device).view(-1,1,1)
        with torch.no_grad():
            ## p1, p2 is probability of angle and doze
            p1,p2,_,_,_,_,_=self.policy_net.prob(state,self.hidden_a_pre,angle1,freq)
            value,next_hidden=self.target_critic_net(state,self.hidden_v_pre)
            next_value,_=self.target_critic_net(next_state,next_hidden)
            if if_end:
                advantage=reward-value
            else:
                advantage=next_value*self.gamma+reward-value
            advantage.detach()
            p1=p1.detach()
            p2=p2.detach()
        if torch.cuda.is_available():
            advantage=advantage.cpu()
            p1=p1.cpu()
            p2=p2.cpu()
            value=value.cpu()
        p1=p1.numpy()[0,0]
        p2=p2.numpy()[0,0]
        advantage=advantage.numpy()[0,0]
        value=value.numpy()[0,0]
        return p1,p2,advantage,value
    
    '''compute the new hidden h_t and update it to the hidden_pre '''
    def next_hidden(self,s,r):
        s = torch.tensor(s, dtype=torch.float,device=self.device).view(-1,1,self.state_dim+self.action_dim)
        rest=torch.zeros([1,1,self.rest],dtype=torch.float,device=self.device)
        rest[:,:,r-1]=1
        state=torch.cat([s,rest],dim=2)
        with torch.no_grad():
            _,_,self.hidden_a_pre=self.policy_net(state,self.hidden_a_pre)
            _,self.hidden_v_pre = self.critic_net(state,self.hidden_v_pre)
        return self.hidden_a_pre.clone(),self.hidden_v_pre.clone()

    '''use all information from episodes from batch generated from the current network to optimize them'''
    def train(self,batch_size):
        self.policy_net.train()
        self.critic_net.train()
        self.target_critic_net.eval()
        #self.policy_net.resample()
        #self.actor_policy_net.resample()

        batch_train=self.replay_buffer.sample(batch_size)
        self.start_epoch(batch_size)
        pre_obs_batch=[]
        pre_r_batch=[]
        next_obs_batch=[]
        next_r_batch=[]
        action_batch=[]
        freq_batch=[]
        if_end_batch=[]
        reward_batch=[]
        mask_batch=[]
        count_eff=0
        p1_old_batch=[]
        p2_old_batch=[]
        advantage_batch=[]
        value_batch=[]
        
        ## max_length = 179
        for i in range(self.rest):
            pre_obs=[]
            next_obs=[]
            pre_r=[]
            next_r=[]
            reward=[]
            action=torch.zeros(batch_size,self.action_dim)
            freq=[]
            if_end=[]
            p1_old=[]
            p2_old=[]
            advantage=[]
            value=[]
            ## batch_size = 12
            ## batch_train[j] = history of episode j
            ## history[i] = (state,(angle,freq,p1,p2,gae,value,reward),reward1,done,next_state)
            '''save data of the same step t from different episodes in the batch'''
            for j in range(batch_size):
                pre_obs.append(batch_train[j][i][0][0])
                next_obs.append(batch_train[j][i][4][0])
                pre_r.append(batch_train[j][i][0][1])
                next_r.append(batch_train[j][i][4][1])
                for t in batch_train[j][i][1][0]:
                    action[j,t]=1
                reward.append(batch_train[j][i][2])
                if np.isnan(batch_train[j][i][2]):
                    return 0
                freq.append(batch_train[j][i][1][1])
                p1_old.append(batch_train[j][i][1][2])
                p2_old.append(batch_train[j][i][1][3])
                    
                ## advantage = gae 
                advantage.append(batch_train[j][i][1][4])
                value.append(batch_train[j][i][1][5])
                if batch_train[j][i][3]:
                    if_end.append(0.0)
                else:
                    if_end.append(1.0)
                count_eff+=1
            pre_obs_batch.append(pre_obs)
            pre_r_batch.append(pre_r)
            p1_old_batch.append(p1_old)
            p2_old_batch.append(p2_old)
            advantage_batch.append(advantage)
            value_batch.append(value)
            if i==0:
                next_obs_batch.append(pre_obs)
                next_obs_batch.append(next_obs)
                next_r_batch.append(pre_r)
                next_r_batch.append(next_r)
            else:
                next_obs_batch.append(next_obs)
                next_r_batch.append(next_r)
            action_batch.append(action.tolist())
            if_end_batch.append(if_end)
            reward_batch.append(reward)
            freq_batch.append(freq)
        pre_obs=torch.tensor(pre_obs_batch,dtype=torch.float,device=self.device)
        pre_r=torch.tensor(pre_r_batch,dtype=torch.float,device=self.device).view(-1,batch_size,self.rest)
        pre_state=torch.cat([pre_obs,pre_r],dim=2)
        next_obs=torch.tensor(next_obs_batch,dtype=torch.float,device=self.device)
        next_r=torch.tensor(next_r_batch,dtype=torch.float,device=self.device).view(-1,batch_size,self.rest)
        next_state=torch.cat([next_obs,next_r],dim=2)
        action=torch.tensor(action_batch,dtype=torch.float,device=self.device).view(-1,batch_size,self.action_dim)
        if_end=torch.tensor(if_end_batch,dtype=torch.float,device=self.device)
        reward=torch.tensor(reward_batch,dtype=torch.float,device=self.device)
        #print(log_pi_old)
        freq=torch.tensor(freq_batch,dtype=torch.float,device=self.device).view(-1,batch_size,1)
        p1_old=torch.tensor(p1_old_batch,dtype=torch.float,device=self.device).view(-1,batch_size)
        p2_old=torch.tensor(p2_old_batch,dtype=torch.float,device=self.device).view(-1,batch_size)
        advantage=torch.tensor(advantage_batch,dtype=torch.float,device=self.device).view(-1,batch_size)
        value_pre=torch.tensor(value_batch,dtype=torch.float,device=self.device).view(-1,batch_size)

        self.critic_optimizer.zero_grad()
        with torch.no_grad():
            #v_target,_=self.target_critic_net(pre_state,self.hidden_v_pre_t)
            v_target=value_pre+advantage
            v_target=v_target.detach()
        v_pred,_ = self.critic_net(pre_state,self.hidden_v_pre)
        closs = (v_pred - v_target) ** 2 
        if self.value_clip:
            closs1=(value_pre+(v_pred-value_pre).clamp(-self.epsilon,self.epsilon)-v_target)**2
            closs=torch.max(closs,closs1)
        #print(closs)
        closs=closs.sum()/count_eff
        #print("closs:",closs)
        closs.backward()
        self.critic_optimizer.step()

        self.actor_optimizer.zero_grad()
        if self.adv_scale:
            advantage_mean=(advantage).sum()/count_eff
            advantage_std=torch.sqrt((advantage-advantage_mean).pow(2).sum()/count_eff)
            advantage=(advantage-advantage_mean)/(advantage_std+1e-8)
        self.a=0.3
        p1,p2,_,_,_,outp1,outp2= self.policy_net.prob(pre_state,self.hidden_a_pre,action,freq)
        action_prior = 1/(self.rest+1)*torch.ones(outp1.shape)/(360-self.rest)
        for aa in [0,60,80,100,120,180,240,260,280,300]:
            action_prior[:,:,aa] = 1/(self.rest+1)-(aa-4.5)/(self.rest+1)/10
        #print(outp2[[1,2],0,:])
        #print(outp2[[1,2],1,:])
        action_prior = action_prior.to(self.device)
        freq_prior = 0.6*torch.ones(outp2.shape, device=self.device)/(4-1)
        freq_prior[:,:,3] = 0.4
        freq_prior[[0,self.rest//2],:,0] = 0.4
        freq_prior[[0,self.rest//2],:,3] = 0.6/(4-1)
        r=(p1*p2)/(p1_old*p2_old)
        r2=(-self.a*r+(1+self.a)*(1-self.epsilon))*(r<=(1-self.epsilon))+(-self.a*r+(1+self.a)*(1+self.epsilon))*(r>=(1+self.epsilon))+r*(r<(1+self.epsilon))*(r>(1-self.epsilon))
        #r2=torch.clamp(r,1-self.epsilon,1+self.epsilon)
        advantage=advantage#-0.02*(p1_old.log()+p2_old.log())
        ploss=torch.min(r*advantage,r2*advantage)
        #ploss=ploss*(advantage>=0.0)+torch.max(ploss,2.0*advantage)*(advantage<0.0)
        ploss=-ploss.sum()/count_eff
        #print(ploss)
        act_loss = torch.sum((action_prior-outp1)**2)
        freq_loss = torch.sum((freq_prior-outp2)**2)
        prior_loss = act_loss + freq_loss
        #print(act_loss)
        #print(prior_loss)
        #prior_loss.backward()
        ploss.backward()
       
        if self.use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(),self.max_grad_norm)
        self.actor_optimizer.step()


        if self.use_linear_lr_decay:
            self.actor_scheduler.step()
            self.critic_scheduler.step()
            

        self.target_policy_net.load_state_dict(self.policy_net.state_dict())
        self.target_critic_net.load_state_dict(self.critic_net.state_dict())
    
    def action(self,state ,rest, choose_act, deterministic = False):
        '''
        use the actor_policy_net to compute the action.

        s: (np.ndarray, batch_size x state_channel x num_x) the input state  
        deterministic: (bool) if False, add exploration noise to the actions. default False. 
        '''
        ## s = (obs, act_distribution), doze = rest doze to use, choose act = chosen action = act_distribution
        if deterministic:
            self.policy_net.eval()
        else:
            self.policy_net.train()
        self.critic_net.eval()
        state = torch.tensor(state, dtype=torch.float,device=self.device).view(-1,1,self.state_dim+self.action_dim)
        #rest=torch.tensor([rest],dtype=torch.float,device=self.device).view(-1,1,1)
        rest0=torch.zeros([1,1,self.rest],dtype=torch.float,device=self.device)
        rest0[:,:,rest-1]=1
        state=torch.cat([state,rest0],dim=2)
        choose_act=torch.tensor(choose_act,dtype=torch.float,device=self.device)
        with torch.no_grad():
            angle, freq, out2= self.policy_net.act(state,self.hidden_a_pre,choose_act,deterministic)
        return angle, freq, out2

    def save(self, save_path,ep):
        print(f"save as {save_path}{ep}")
        torch.save(self.policy_net.state_dict(), save_path + '{}_actor_AC.txt'.format(ep) )
        torch.save(self.critic_net.state_dict(), save_path + '{}_critic_AC.txt'.format(ep))
        torch.save(self.actor_optimizer.state_dict(),save_path+'{}_actor_optim.txt'.format(ep))
        torch.save(self.critic_optimizer.state_dict(),save_path+'{}_critic_optim.txt'.format(ep))
        stat_fp=open('./obs_stat.txt','wb')
        pickle.dump(self.statestat,stat_fp)

    def load(self, load_path):
        self.policy_net.load_state_dict(torch.load(load_path + '_actor_AC.txt',map_location=self.device))
        self.target_critic_net.load_state_dict(torch.load(load_path + '_critic_AC.txt',map_location=self.device))
        self.target_policy_net.load_state_dict(torch.load(load_path + '_actor_AC.txt',map_location=self.device))
        self.critic_net.load_state_dict(torch.load(load_path + '_critic_AC.txt',map_location=self.device))
        self.actor_optimizer.load_state_dict(torch.load(load_path + '_actor_optim.txt'))
        self.critic_optimizer.load_state_dict(torch.load(load_path + '_critic_optim.txt'))
        

# update 10/19 17:17
# update rest from a scalar to a one-hot vector

set_seed(12)
save_path=""

agent=PPO_RB(size=64,rest=10,shape="circle",sample_size=5000,freq_bound=64,num_freq=4,state_dim=20,action_dim=360,
             hidden_cell=(256,64),n_layers=(3,2),save_path=save_path,load_path="")
load_path = "C:/Users/scott/Desktop/Projects/RL Inverse scattering/RLIS/1990"
#load_path = "C:/Users/scott/Desktop/Projects/RL Inverse scattering/RLIS/pretrain/s32r10f01/870"
agent.load(load_path)
agent.multi_train(2000)
# img1= true_img save700 better
## reconstruction

agent1=PPO_RB(size=32,rest=10,shape="oval",sample_size=500,freq_bound=32,num_freq=4,state_dim=20,action_dim=360,
             hidden_cell=(256,64),n_layers=(3,2),save_path=save_path,load_path="")
#load_path1 = "C:/Users/scott/Desktop/RLIS/pretrain/sin/nonuniform/290"
load_path1 = "C:/Users/scott/Desktop/Projects/RL Inverse scattering/RLIS/pretrain/sin/nonuniform/290"
agent1.load(load_path)

agent2=PPO_RB(size=32,rest=10,shape="oval",sample_size=500,freq_bound=32,num_freq=4,state_dim=20,action_dim=360,
             hidden_cell=(256,64),n_layers=(3,2),save_path=save_path,load_path="")
load_path2 = "C:/Users/scott/Desktop/Projects/RL Inverse scattering/RLIS/pretrain/s32r10f01/870"
agent2.load(load_path2)

compare(agent2, agent, r=10, times=10)