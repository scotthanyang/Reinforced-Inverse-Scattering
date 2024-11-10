# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 17:18:39 2023

@author: scott
"""

import time
import numpy as np
import random
import math
import pydicom as dicom
import os
import copy
import torch
import scipy
from scipy.spatial.distance import pdist, squareform
from scipy.special import hankel1
from sklearn import linear_model
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Distribution, Normal
from torch.distributions.categorical import Categorical
from torch.distributions.one_hot_categorical import OneHotCategorical
from torch.nn import init
import numpy as np
from torch.autograd import Variable

def read_img(img_path,size):
    """
    walk the dir to load all image
    """
    t=512//size
    img_list=[]
    print('image loading...')
    for _,_,files in os.walk(img_path):
        for f in files:
            if f.find('.IMA')>=0:
                tmp_img=dicom.dcmread(os.path.join(img_path,f))
                tmp_img=(tmp_img.pixel_array[0::t,0::t]-np.mean(tmp_img.pixel_array[0::t,0::t]))
                tmp_img=tmp_img/np.linalg.norm(tmp_img)
                img_list.append(tmp_img)
    img_data=np.array(img_list)
    print('done')
    return img_data

def psnr2(u_pre,u_true):
    mse=np.mean((u_pre-u_true)**2)
    if mse==0:
        return 100
    PIXEL_MAX = 4096
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


class IS():
    def __init__(self,size,rest,num_freq=4,sample_size=1,have_noise=False,shape="oval",sensor_pos="circle"):
        self.size=size
        self.width=0.5
        #self.img_data = generate_eta_batch(self.width/2, sample_size ,size, 2, 10,0.01)
        #self.img_data=generate_image(size,sample_size)
        #self.img_data = read_img("D:/data/FD_1mm/full_1mm/L067/full_1mm",size)
        self.img_data=[]
        self.sensor_pos=sensor_pos
        if shape == "oval":
            for i in range(sample_size):
                self.img_data.append(generate_tri_oval(self.width, size))
        elif shape == "circle":
            for i in range(sample_size):
                self.img_data.append(generate_circles(self.width, size))
        elif shape == "mnist":
            size = 28
            data_path = "C:/Users/scott/Desktop/RLIS/data/mnist_test.csv"
            data = np.loadtxt(data_path,delimiter=",")
            size = data.shape[0]
            s = random.sample(range(0,size),sample_size)
            for i in s:
                d=data[i,1:]
                d=d/np.linalg.norm(d)*2
                d=d.reshape((28,28))
                self.img_data.append(d)
        elif shape == "dots":
            for i in range(sample_size):
                self.img_data.append(generate_dots(self.width, size))    
        self.img_data_size=len(self.img_data)
        self.have_noise=have_noise
        self.action_num=360
        self.max_photon=1.4*180/1e5
        self.rest=rest
        self.proj_data=0
        self.detector=0
        self.freq_list=np.array([size*j//num_freq for j in range(1,num_freq+1)])
        self.freq=[]
        self.reconstruct_alg=reconstruct
        self.G=[]
        x = np.linspace(-self.width, self.width, num=self.size)
        y = np.linspace(-self.width, self.width, num=self.size)
        X, Y = np.meshgrid(x, y)
        self.gpts = np.array([X.flatten(), Y.flatten()]).T
        h = 2*self.width/(size-1)
        for freq in self.freq_list:
            self.G.append(greensfunction2(self.gpts, freq, h))

    def reset(self,rest=10,set_pic=None):
        self.detector=0
        self.state_proj_seq=[]
        self.angle_seq=[]
        self.A_seq=[]
        self.freq=[]
        self.true_img=self.img_data[random.randint(0,self.img_data_size-1)]

        if set_pic is not None:
            self.true_img=set_pic
        #self.true_img=self.img_data[10]
        init_act=random.randint(0,self.action_num)
        s1,s2=self.true_img.shape
        proj_data=np.zeros((2*self.detector**2,))
        img_size=self.true_img.shape
        self.state=np.zeros(img_size)
        #self.rest=1.+random.random()*0.2-0.1
        self.rest=rest
        self.start=True
        eta = np.reshape(self.true_img,(s1*s2,1))
        E = np.diag(eta.T[0])
        #self.ReEE=[]
        #self.ImEE=[]
        #for g in self.G:
        #    self.ReEE.append(np.multiply(np.multiply(np.diag(E)[:,None], np.real(g)),np.diag(E))+E)
        #    self.ImEE.append(np.multiply(np.multiply(np.diag(E)[:,None], np.imag(g)),np.diag(E)))
        self.proj_data=proj_data.copy()
        return proj_data,0

    def step(self,action,freq,alpha="T",stepsize=0):
        self.angle_seq = np.concatenate([self.angle_seq,action],axis=0)
        self.detector+=1
        self.freq = np.concatenate([self.freq, [freq]])
        proj_data=self.get_project_data(self.true_img,action,freq,self.have_noise)
        self.state_proj_seq.append(proj_data)
        self.old_state=self.state
        if alpha=="T":
            if stepsize==0:
                self.state=self.reconstruct_alg(self.state,self.state_proj_seq,self.angle_seq,
                                        self.A_seq,self.freq,self.freq_list,self.G,self.gpts)
            else:
                self.state=self.reconstruct_alg(self.state,self.state_proj_seq,self.angle_seq,
                                        self.A_seq,self.freq,self.freq_list,self.G,self.gpts,stepsize=stepsize)
        else:
            if stepsize==0:
                self.state=self.reconstruct_alg(self.state,self.state_proj_seq,self.angle_seq,
                                        self.A_seq,self.freq,self.freq_list,self.G,self.gpts,alpha=alpha)
            else:
                self.state=self.reconstruct_alg(self.state,self.state_proj_seq,self.angle_seq,
                                        self.A_seq,self.freq,self.freq_list,self.G,self.gpts,alpha=alpha,stepsize=stepsize)
        
        reward=psnr2(self.state,self.true_img)-psnr2(self.old_state,self.true_img)
        if self.rest == 1:
            done = True
        else:
            done = False
        self.proj_data=proj_data.copy()
        self.rest-=1
        
        return proj_data,reward,done,None
    
    def greenfunction(self, freq):
        x = np.linspace(-self.width,self.width,self.size)
        y = np.linspace(self.width,-self.width,self.size) 
        X = np.meshgrid(x,y)
        X0 = X[0].reshape((self.size**2,1))
        X1 = X[1].reshape((self.size**2,1))
        X = np.concatenate([X0,X1],axis=1)
        s = X.shape[0]
        G = np.zeros((s,s),dtype=complex)
        for i in range(s):
            tmp1 = np.sqrt(np.sum(np.power(abs(X[[i], :] - X), 2), axis=1))
            kr = tmp1 * freq
            tmp = scipy.special.hankel1(0, kr)
            G[[i], :] = -1j / 4 * tmp.transpose()
            G[i, i] = 0
        return G
   

    def get_project_data(self,img,action,freq,noise=False):
        G = self.G[freq]
        k = self.freq_list[freq]
        n = img.shape[0]
        detector=self.detector
        q = np.reshape(img,(n**2,))

        S = action[0]*np.pi/180
        R = [i*np.pi/180 for i in self.angle_seq]

        r_circ = 100

        nin = 1
        pts_receive = r_circ*np.array([np.cos(R), np.sin(R)]).T
        Gcirc = getGscat2circ(self.gpts, pts_receive, k)
        #c = -1j/4*np.sqrt(2/(k*np.pi*r_circ))*np.exp(-1j*np.pi/4)
        #c*np.exp(1j*k*np.sqrt(np.sum((gpts - pts_receive[-1,:])**2, axis=1)))
        inc = np.array([np.cos(S), np.sin(S)]).T
        uin = np.exp(1j*k*self.gpts@inc.T)
        b = q*uin
        #A = np.eye(n**2) - G@np.diag(q)#@np.eye(ndom**2)
        #sigma = np.linalg.solve(A, b)
        #sigma = np.reshape(sigma, (n**2, nin))
        sigma = b+ G@(q*b)
        FP = Gcirc@sigma
        FP = np.reshape(FP, (nin, len(R)))
        data = np.concatenate([np.real(FP),np.imag(FP)])
        return data.reshape((2*detector,)).copy()

    def show_psnr(self):
        return psnr2(self.state,self.true_img)


    def show_psnr(self):
        return psnr2(self.state,self.true_img)

'''state_seq = self.state_proj_seq
angle_seq = self.angle_seq
A_seq = self.A_seq
freq_seq = self.freq
freq_list = self.freq_list
G_list = self.G
gpts =self.gpts 
s = torch.tensor(true_img.reshape((-1)),dtype=torch.float).to(device)   '''


# for oval: alpha=3e-6
# 2/19  1e-5
# size = 32  alpha = 5e-5
# size = 64  alpha = 5e-6
# 2024/6/15  adam iter=12  
def reconstruct(state, state_seq, angle_seq, A_seq, freq_seq, freq_list, G_list,  
                gpts, alpha=1e-3, iter=3, stepsize=1e-2):  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    s1,s2=state.shape
    state0 = torch.tensor(state,dtype=torch.float).reshape((s1*s2)).to(device)
    O = opt_network(s1*s2)
    with torch.no_grad():
        O.linear.bias = nn.Parameter(state0.clone())
        O.linear.weight = nn.Parameter(O.linear.weight.to(device))
        O.linear.weight.requires_grad = False
    #optimizer = torch.optim.Adam(O.parameters(),stepsize,betas=(0.9,0.999),amsgrad=True)
    optimizer = torch.optim.LBFGS(O.parameters(),lr=1)
    O.train()
    #ReG0_list = []
    #ImG0_list = []
    #for G0 in G_list:
    #    ReG0_list.append(torch.tensor(np.real(G0),dtype=torch.float).to(device))
    #    ImG0_list.append(torch.tensor(np.imag(G0),dtype=torch.float).to(device))
    G_list = [torch.tensor(G_list[i],dtype=torch.complex64).to(device) for i in range(len(G_list))]
    ii = torch.eye(s1*s2, dtype=torch.complex64).to(device)
    for j in range(iter):
        ## this is the state (size**2)
        
        def closure():
            optimizer.zero_grad()
            s = O.forward(torch.tensor([0],dtype=torch.float).to(device))
            E = torch.diag(s)
            loss = 0
            for i in range(len(state_seq)):
                k = freq_list[int(freq_seq[i])]  
                G = G_list[int(freq_seq[i])]
                ## compute source
                S = [angle_seq[i]*np.pi/180]
                inc = np.array([np.cos(S), np.sin(S)]).T
                uin = np.exp(1j*k*gpts@inc.T)
                uin_r = torch.tensor(np.real(uin),dtype=torch.float).to(device)
                uin_i = torch.tensor(np.imag(uin),dtype=torch.float).to(device)
                ## compute receiver
                R = [ang*np.pi/180 for ang in angle_seq[:(i+1)]]
                pts_receive = 100*np.array([np.cos(R), np.sin(R)]).T
                Gcirc = getGscat2circ(gpts, pts_receive, k)
                Gcirc_r = torch.tensor(np.real(Gcirc),dtype=torch.float).to(device)
                Gcirc_i = torch.tensor(np.imag(Gcirc),dtype=torch.float).to(device)
                #A = torch.eye(s1*s2, dtype=torch.complex64).to(device) - (G@torch.diag(s))
                #B = torch.tensor(E,dtype=torch.complex64)@torch.tensor(uin, dtype=torch.complex64).to(device)
                #inv = torch.linalg.pinv(A)
                GG = (G@torch.tensor(E,dtype=torch.complex64))
                inv = ii + GG 
                real_inv = torch.tensor(np.real(inv),dtype=torch.float).to(device)
                imag_inv = torch.tensor(np.imag(inv),dtype=torch.float).to(device)
                G1 = Gcirc_r@real_inv-Gcirc_i@imag_inv
                G2 = Gcirc_i@real_inv+Gcirc_r@imag_inv
                real = G1@(E@uin_r) - G2@(E@uin_i)
                imag = G1@(E@uin_i) + G2@(E@uin_r)
                real_d = torch.tensor(state_seq[i][:len(state_seq[i])//2],dtype=torch.float).to(device)
                imag_d = torch.tensor(state_seq[i][len(state_seq[i])//2:],dtype=torch.float).to(device)
                loss += torch.norm(real_d-real.reshape((-1)))**2 + torch.norm(imag_d-imag.reshape((-1)))**2
            loss += alpha*torch.norm(s,p=1)  
            loss.backward()
            return loss
        optimizer.step(closure)
    state0 = O.linear.bias.clone().detach()
    if torch.cuda.is_available():
        state0=state0.cpu()
    img_rec=state0.numpy().reshape((s1,s2))   
    
    torch.cuda.empty_cache()   
    return img_rec.copy()


class opt_network(nn.Module):
    def __init__(self,n_hid,n_layers=1):
        super(opt_network, self).__init__()
        self.n_layers=n_layers
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.linear=nn.Linear(1,n_hid)
        self.init_weights()

    def init_weights(self):
        torch.nn.init.constant_(self.linear.weight, 0.0)
        torch.nn.init.constant_(self.linear.bias, 0.0)

    def forward(self,state):
        out=self.linear(state)
        return out
    
def generate_circles(eta_domain, n):
    num=2
    c_x = np.random.uniform(-eta_domain/2, eta_domain/2, num)
    c_y = np.random.uniform(-eta_domain/2, eta_domain/2, num)
    x = np.linspace(-eta_domain, eta_domain, n)
    y = np.linspace(-eta_domain, eta_domain, n)
    [X, Y] = np.meshgrid(x, y)
    X = X.flatten()
    Y = Y.flatten()
    X = np.mat(X)
    Y = np.mat(Y)
    eta = np.zeros([1, X.shape[1]])
    r = np.random.uniform(0.08,0.15,num)
    v = np.random.uniform(0.5,1,num)
    # print(eta.shape)
    for j in range(num):
        for i in range(X.shape[1]):
        # xx = X[0,i]
        # yy = Y[0,i]
        # print(xx,' ', yy)
           
            tmp = np.power(X[0, i] - c_x[j], 2)  + np.power(Y[0, i] - c_y[j], 2) 
            # print(tmp)
            if tmp < r[j]**2:
                eta[0, i] = v[j]                
               
    # print(tmp.shape)
    eta = eta.transpose()
    eta=eta/np.linalg.norm(eta)*2/100
    return eta.reshape((n,n))


def generate_tri_oval(eta_domain, n):
    num =1
    c_x = np.random.uniform(-eta_domain/2, eta_domain/2, num)
    c_y = np.random.uniform(-eta_domain/2, eta_domain/2, num)
    x = np.linspace(-eta_domain, eta_domain, n)
    y = np.linspace(-eta_domain, eta_domain, n)
    [X, Y] = np.meshgrid(x, y)
    X = X.flatten()
    Y = Y.flatten()
    X = np.mat(X)
    Y = np.mat(Y)
    n2 = X.shape[1]
    k = np.array([[1],[1]])
    eta = np.zeros([1, n2])
    # print(eta.shape)
    t = random.uniform(-0.4,0.4)
    for j in range(num): 
        t = random.uniform(-0.3,0.3)
        for i in range(X.shape[1]):
        # xx = X[0,i]
        # yy = Y[0,i]
        # print(xx,' ', yy)
           
            tmp = np.power(X[0, i] - c_x[j], 2) / (0.002*(j+1)) + np.power(Y[0, i] - c_y[j], 2) / (0.01/(j+1))
            # print(tmp)
            if tmp < 1:
                eta[0, i] = 1                
               
            #if (X[0, i] < 0.1+t) and (Y[0, i] > 0.05+t) and (-X[0, i] + Y[0, i] < 0.06*j):
            #    eta[0, i] = 1
            # print(tmp.shape)
            #eta[0, i] = eta[0,i]*np.sin(k.T@np.array([[X[0,i]],[Y[0,i]]]))[0][0]
    eta = eta.transpose()
    gpt = np.vstack((Y, X))
    eta=eta/np.linalg.norm(eta)
    return eta.reshape((n,n))#, gpt.transpose(), c_x, c_y
        
def generate_dots(eta_domain, n):
    num=3
    eta = np.zeros([1,n**2])
    # print(eta.shape)
    for j in range(num):
        c_x=np.random.randint(0,n**2)
        eta[0,c_x]=1
               
    # print(tmp.shape)
    eta=eta/np.linalg.norm(eta)**2/100
    return eta.reshape((n,n))

def greensfunction2(gpt, k, h):
    G = np.zeros((gpt.shape[0], gpt.shape[0]), dtype=complex)
    for i in range(gpt.shape[0]):
        kR = k*np.sqrt(np.sum((gpt - gpt[i, :])**2, axis=1))
        tmp = hankel1(0, kR)
        G[i, :] = -1j/4*tmp

    G = G
    np.fill_diagonal(G, 0)

    return G

def getGscat2circ(gpts, gpts_circ, k):
    G = np.zeros((gpts_circ.shape[0], gpts.shape[0]), dtype=complex)
    for i in range(gpts_circ.shape[0]):
        kR = k*np.sqrt(np.sum((gpts_circ[i,:]-gpts)**2, axis=1))
        tmp = hankel1(0, kR)
        tmp = -1j/4*tmp
        G[i,:] = tmp
    return G


def compare(agent1, agent2, r=10, times=50):
    win1 = 0
    img = []
    for j in range(times):
        self=agent
        if self.shape == "circle":
            true_img=generate_circles(self.env.width, self.env.size)
        if self.shape == "oval":
            true_img=generate_tri_oval(self.env.width, self.env.size)
        if self.shape == "dots":
            true_img=generate_dots(self.env.width, self.env.size)
        if self.shape == "mnist":
            true_img=self.env.img_data[random.randint(0,len(self.env.img_data))]
        img.append(true_img)
        alpha = 5e-6
        
        self=agent
        obs,_=self.env.reset(rest=r,set_pic=true_img)
        self.start_epoch(1)
        old_act=np.zeros((self.action_dim,))
        rest=r
        for i in range(self.long_time):
            if i!=0:
                obs_norm=np.clip((obs-self.statestat.mean)/(self.statestat.std+1e-8),-40,40)
            else:
                obs_norm=obs
            obs1=np.concatenate([obs_norm,np.zeros(self.state_dim-len(obs_norm)),old_act],axis=0)
            angle,freq,out=self.action(obs1,rest,old_act,True)
            angle = np.array([360//r*i])
            freq = 0
            #print(angle, freq)
            rest-=1
            next_obs,reward,done,_=self.env.step(angle,freq,alpha=alpha)
            self.next_hidden(obs1,rest+1)
            if i==(self.long_time-1):
                done=True
            old_act[angle]+=freq+1
            obs=next_obs
            if done:
                break
        '''self.env.state = reconstruct(self.env.state,self.env.state_proj_seq,self.env.angle_seq,
                                       self.env.A_seq,self.env.freq,self.env.freq_list,self.env.G,
                                       self.env.gpts,iter=20)'''
        e1 = psnr2(true_img,self.env.state)
        print("round ",j," agent 1:",e1,alpha)
        s1 = self.env.state
        p1.append(e1)
        
        self=agent
        obs,_=self.env.reset(rest=r,set_pic=true_img)
        self.start_epoch(1)
        old_act=np.zeros((self.action_dim,))
        rest=r
        for i in range(self.long_time):
            if i!=0:
                obs_norm=np.clip((obs-self.statestat.mean)/(self.statestat.std+1e-8),-40,40)
            else:
                obs_norm=obs
            obs1=np.concatenate([obs_norm,np.zeros(self.state_dim-len(obs_norm)),old_act],axis=0)
            angle,freq,out=self.action(obs1,rest,old_act,True)
            print(angle,freq)
            rest-=1
            next_obs,reward,done,_=self.env.step(angle,freq,alpha=alpha)
            self.next_hidden(obs1,rest+1)
            if i==(self.long_time-1):
                done=True
            old_act[angle]+=freq+1
            obs=next_obs
            '''self.env.state = reconstruct(self.env.state,self.env.state_proj_seq,self.env.angle_seq,
                                       self.env.A_seq,self.env.freq,self.env.freq_list,self.env.G,
                                       self.env.gpts)'''
            if done:
                break
        e2 = psnr2(true_img,self.env.state)
        p2.append(e2)
        print("round ",j," agent 2:",e2, alpha)
        fig, (ax1,ax2,ax3) = plt.subplots(1,3)  
        ax1.imshow(true_img)
        ax2.imshow(s1)
        ax3.imshow(self.env.state)
        ax1.set_title('true_img')
        ax2.set_title("psnr="+str(round(e1,4)))
        ax3.set_title("psnr="+str(round(e2,4)))
        ax1.axis('off')
        ax2.axis('off')
        ax3.axis('off')
        plt.show()
        if e2 > 128:
            break
        if e1 > e2:
            win1 += 1
        
    print("final agent 1 wins ", win1," times")

