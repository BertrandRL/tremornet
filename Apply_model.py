import numpy as np
import pandas as pd
import time
import os
import sys
import glob

import obspy
from obspy.core.trace import Trace
from obspy.core.stream import Stream
from obspy import UTCDateTime

import pickle
import torch
import torch.nn
from torch.nn import functional as F


class Net(nn.Module):
    ################
    # Import model #
    ################
    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv1 = nn.Conv2d(3, nf1, ks1,dilation=dilation)
        self.conv2 = nn.Conv2d(nf1, nf2, ks2,dilation=dilation)
        self.conv3 = nn.Conv2d(nf2, nf3, ks3,dilation=dilation)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(n_features, fcn1)  # (((124-3)//3-2)//3)* (((94-3)//3-2)//3)*16
        self.fc2 = nn.Linear(fcn1, fcn2)
        self.fc3 = nn.Linear(fcn2, 1)
        self.dropout1 = nn.Dropout2d(drop)

    def forward(self, x):
        x=self.dropout1(x)
        x = F.max_pool2d(F.leaky_relu(self.conv1(x), negative_slope=0.1), ps1)
        x=self.dropout1(x)
        x = F.max_pool2d(F.leaky_relu(self.conv2(x), negative_slope=0.1), ps2)
        x=self.dropout1(x)
        x = F.max_pool2d(F.leaky_relu(self.conv3(x), negative_slope=0.1), ps3)
        x=self.dropout1(x)
        x = x.view(-1, n_features)
        x = F.relu(self.fc1(x))       
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        x=torch.flatten(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
        
        
#########################################
#           Main program                #
#########################################
if __name__ == "__main__":

    # Device to use
    gpu=0 
    device=torch.device("cuda:%i"%gpu)

    # NN model and interpretor
    nx=124
    ny=94
    #Filters
    nf1=32
    nf2=64
    nf3=128
    #Kernel sizes
    ks1=3
    ks2=3
    ks3=3
    dilation=1
    #Pooling size
    ps1=3
    ps2=2
    ps3=2
    #Number of fully connected neurons
    fcn1=180
    fcn2=90
    #Dropout
    drop=0.1
    x = F.max_pool2d(F.leaky_relu(nn.Conv2d(3, nf1, ks1,dilation=dilation)(torch.empty([1, 3, 124, 94]).cpu()), negative_slope=0.1), ps1)
    x = F.max_pool2d(F.leaky_relu(nn.Conv2d(nf1, nf2, ks2,dilation=dilation)(x), negative_slope=0.1), ps2)
    x = F.max_pool2d(F.leaky_relu(nn.Conv2d(nf2, nf3, ks3,dilation=dilation)(x), negative_slope=0.1), ps3)
    n_features=x.shape[1]*x.shape[2]*x.shape[3]
    # Import model
    tremor_model = Net()
    tremor_model.load_state_dict(torch.load('working_tremornet_robust_small_kernel_normalized_fgsm_HH.tar'))
    tremor_model.cuda().to(device)
    tremor_model.eval()
    
    ###########################
    # Read data here... -> st #
    ###########################
    
    # To apply once the data has been read
    nfft=int(n_fft*sampling/s_rate)
    window=torch.hamming_window(nfft).cuda()
    #Compute stft   
    stft=torch.stft(st,n_fft=nfft,hop_length=nfft//2,window=window)[:,dim1:dim2,:]
    stft_norm=torch.view_as_complex(stft).abs()
    stft_angle=torch.view_as_complex(stft).angle()
    stft_norm=(stft_norm-torch.mean(stft_norm,dim=(1,2)).reshape((stft_norm.shape[0],1,1)))/torch.std(stft_norm,dim=(1,2)).reshape((stft_norm.shape[0],1,1))
    stft_norm=stft_norm.reshape(n_stations, 3, 124, 94)
    stft_angle=stft_angle.reshape(n_stations, 3, 124, 94)
    #Compute tremorness
    tremorness=[]
    analyses=[]
    with torch.no_grad():
        tremorness=tremor_model(stft_norm[:,:,:,:])      
