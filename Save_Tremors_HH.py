import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import obspy
from obspy.core.trace import Trace
from obspy import UTCDateTime
from obspy.core.stream import Stream
from obspy.clients.fdsn import Client
from obspy.clients.fdsn import Client

import glob
import time
import scipy
import os.path
import h5py
from geopy import distance
import torch
#import torchaudio

client=Client("IRIS")
#client=Client('NCEDC')

def build_database_tremors(df,indices,network,station,channels,nfft,s_rate,l,dim1,dim2,dim3,path,latitude,longitude):

    spectrograms=[]
    times=[]
    i=0

    # Loop over tremors
    for j in indices: #(len(times_cat)):
        try:
            t=pd.to_datetime(df['time'].values[j]) #times_cat[i]
            #r=np.random.uniform(-5,l-5)
            #t_start=UTCDateTime(t)-r
            #t_end=UTCDateTime(t)-r+l
            d=distance.distance((latitude,longitude),(df['lat'].values[j],df['lon'].values[j])).km
            t_start=UTCDateTime(t)+d/3
            t_end=UTCDateTime(t)+d/3+30
            try: n_channels=len(channels.split(','))
            except: n_channels=1
            try: st=client.get_waveforms(network,station,"*",channels,t_start,t_end,attach_response=True)
            except: st=None
        
            # Verify that all channels are available
            tr=[]
            #if st is not None and len(st)>len(channels.split(',')): st2=st.merge(method=-1)
            if st is not None and len(st)==n_channels:
                for comp in channels.split(','):
                    st3 = st.select(station=station,channel=comp) 
                    tr.append(st3[0].data/st3[0].stats.response._get_overall_sensitivity_and_gain()[1])
        
            # Compute normalized stft
            if len(tr)>0:
                with torch.no_grad():
                    tensor=torch.Tensor(tr) #.cuda()
                    n_fft=nfft*st[0].stats.sampling_rate/s_rate
                    assert int(nfft*st[0].stats.sampling_rate/s_rate)-n_fft==0
                    n_fft=int(n_fft)
                    Fourier_trans=torch.norm(torch.stft(tensor,n_fft=n_fft,hop_length=n_fft//2,window=torch.hamming_window(n_fft)),dim=3)
                    #Fourier_trans=(Fourier_trans-torch.mean(Fourier_trans))/torch.std(Fourier_trans)
                    Fourier_trans=Fourier_trans[:,dim1:dim2,:]
                    #print(Fourier_trans.shape)
    
                #times.append(t)
                if Fourier_trans.shape[0]==len(channels.split(',')) and Fourier_trans.shape[1]==dim2-dim1 and Fourier_trans.shape[2]==dim3: 
                    #print('True')
                    spectrograms.append(Fourier_trans.data.cpu().numpy())
                    times.append(t)
                    
        except: pass
        i=i+1
        
        # Save spectrograms in h5 file
        if len(spectrograms)>0 and (i%100==0 or i==len(times_cat)):
            tremors=np.array(spectrograms).reshape((len(spectrograms),len(channels.split(',')), Fourier_trans.shape[1], Fourier_trans.shape[2]))
            if os.path.exists(path+'/Database_HH_tremors_%s.hdf5'%station):
                with h5py.File(path+'/Database_HH_tremors_%s.hdf5'%station, 'a') as f:
                    f["tremors"].resize((f["tremors"].shape[0] + tremors.shape[0]), axis=0)    
                    f["tremors"][-tremors.shape[0]:] = tremors     
            else: 
                with h5py.File(path+'/Database_HH_tremors_%s.hdf5'%station, 'w') as f:
                    dsetX = f.create_dataset("tremors", data=tremors,maxshape=(None,None,None,None),chunks=True, dtype='float32')

            # Save times
            if os.path.exists(path+'/times_tremors_HH_%s.csv'%station):
                tt=pd.Series(np.array(times))
                tt.to_csv(path+'/times_tremors_HH_%s.csv'%station, mode='a', header=False, index=False)
            else: 
                tt=pd.Series(np.array(times))
                tt.to_csv(path+'/times_tremors_HH_%s.csv'%station, header=False, index=False)
            spectrograms=[]
            times=[]            
    return


##############################
#        Main program        #
##############################

catalog=pd.read_csv('Wesh_catalog_new.csv') #('WechCatalog_all_times.csv')
network='UW' #'NC' #'UO' #'UW'
#station_list= ['KSXB','KHBB','KHMB'] #NC
#station_list=['MINN','RAIN'] #'VERN' #UO
station_list=['FISH','LCCR','LEBA','SP2','STOR'] #'DDRF','PASS','PHIN','TUCA','CCRK','KENT','MRBL'] #UW  
channels='HHE,HHN,HHZ'
s_rate=40
nfft=64 #256
#path='/ram/mnt/local/projects/criticalstress-ldrd/bertrandrl/DB_tremor'
path='/ram/mnt/local/projects/criticalstress-ldrd/bertrandrl/DB_tremor_30'

l=300 #300
v=torch.ones(l*s_rate)
test=torch.stft(v,n_fft=int(nfft),hop_length=nfft//2,window=torch.hamming_window(nfft))
dim1=int(1/(s_rate/2/test.shape[1]))+1
dim2=test.shape[0]
dim3=test.shape[1]
#print(dim1,dim2,dim3)


for station in station_list:
    
    inventory = client.get_stations(network=network,station=station,location='*',channel=channels.split(',')[0],starttime='2009-01-01',endtime='2020-01-01')
    latitude =  inventory[0][0].latitude
    longitude = inventory[0][0].longitude
    df=catalog[catalog['lat']<=latitude+0.5]
    df=df[df['lat']>=latitude-0.5] 
    df=df[df['lon']<=longitude+0.5] 
    df=df[df['lon']>=longitude-0.5]  
    times_cat=pd.DataFrame(pd.to_datetime(df['time'])).values[:,0]

    try:
        saved_events=pd.read_csv(path+'/times_tremors_HH_%s.csv'%station,header=None)
        saved_events=pd.DataFrame(pd.to_datetime(saved_events.values[:,0])).values[:,0]
    except: saved_events=[]

    times_cat=list(set(list(times_cat)).difference(set(list(saved_events))))
    times_cat=pd.to_datetime(times_cat)
    indices=[(times_cat==pd.to_datetime(df['time'].values[i])).nonzero()[0] for i in range(len(df))]
    indices = [item for sublist in indices for item in sublist]
    print ("Saving seismic waveforms for station %s..."%station)
    print('Number of tremors: ', len(times_cat))
    build_database_tremors(df,indices,network,station,channels,nfft,s_rate,l,dim1,dim2,dim3,path,latitude,longitude)
