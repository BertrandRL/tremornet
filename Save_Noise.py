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

import torch

client=Client("IRIS")

def build_database_noise(times_cat,n,network,station,channels,nfft,s_rate,l,dim1,dim2,dim3):

    spectrograms=[]
    times=[]
    i=0
    
    # Loop over times
    for j in range(n): 
        try:
            t2=0 
            while t2<l+30:
                # Draw a time at random
                t="%i-%.2i-%.2iT%.2i:%.2i:%.2i"%(np.random.randint(2010,2020),np.random.randint(1,13),np.random.randint(1,32),np.random.randint(0,24),np.random.randint(0,61),np.random.randint(0,61))
                # Verify that this time does not contain a tremor
                try:
                    test=np.searchsorted(times_cat, pd.to_datetime(t))
                    t2=(times_cat[test]-pd.to_datetime(t)).total_seconds()
                except: pass
            t_start=UTCDateTime(t)+30
            t_end=UTCDateTime(t)+l+30
            try: n_channels=len(channels.split(','))
            except: n_channels=1
            try: st=client.get_waveforms(network,station,"*",channels,t_start,t_end,attach_response=True)
            except: st=None
        
            # Verify that all channels are available
            tr=[]
            if st is not None and len(st)==n_channels:
                for comp in channels.split(','):
                    st3 = st.select(station=station,channel=comp) 
                    tr.append(st3[0].data/st3[0].stats.response._get_overall_sensitivity_and_gain()[1])
        
            # Compute normalized stft
            if len(tr)>0:
                with torch.no_grad():
                    tensor=torch.Tensor(tr) 
                    assert int(nfft*st[0].stats.sampling_rate/s_rate)-nfft*st[0].stats.sampling_rate/s_rate==0
                    Fourier_trans=torch.norm(torch.stft(tensor,n_fft=int(nfft*st[0].stats.sampling_rate/s_rate),hop_length=nfft//2,window=torch.hamming_window(nfft)),dim=3)
                    Fourier_trans=Fourier_trans[:,dim1:dim2,:]
                if Fourier_trans.shape[0]==len(channels.split(',')) and Fourier_trans.shape[1]==dim2-dim1 and Fourier_trans.shape[2]==dim3: 
                    spectrograms.append(Fourier_trans.data.cpu().numpy())
                    times.append(t)
                    
        except: pass
        i=i+1
        
        # Save spectrograms in h5 file
        if len(spectrograms)>0 and (i%100==0 or i==len(times_cat)):
            earthquakes=np.array(spectrograms).reshape((len(spectrograms),len(channels.split(',')), Fourier_trans.shape[1], Fourier_trans.shape[2]))
            if os.path.exists('DB_tremor/Database_BH_noise_%s.hdf5'%station):
                with h5py.File('DB_tremor/Database_BH_noise_%s.hdf5'%station, 'a') as f:
                    f["noise"].resize((f["noise"].shape[0] + earthquakes.shape[0]), axis=0)    
                    f["noise"][-earthquakes.shape[0]:] = earthquakes     
            else: 
                with h5py.File('DB_tremor/Database_BH_noise_%s.hdf5'%station, 'w') as f:
                    dsetX = f.create_dataset("noise", data=earthquakes,maxshape=(None,None,None,None),chunks=True, dtype='float32')
      
            # Save times
            if os.path.exists('DB_tremor/times_noise_BH_%s.csv'%station):
                tt=pd.Series(np.array(times))
                tt.to_csv('DB_tremor/times_noise_BH_%s.csv'%station, mode='a', header=False, index=False)
            else: 
                tt=pd.Series(np.array(times))
                tt.to_csv('DB_tremor/times_noise_BH_%s.csv'%station, header=False, index=False)
            spectrograms=[]
            times=[]

    return


##############################
#        Main program        #
##############################

if __name__ == "__main__":

    catalog=pd.read_csv('WechCatalog_all_times.csv')
    network='UW'
    station_list=['DOSE','GNW','LEBA','LRIV']
    channels='BHE,BHN,BHZ'
    s_rate=40
    nfft=256

    l=300
    v=torch.ones(l*s_rate)
    test=torch.stft(v,n_fft=int(nfft),hop_length=nfft//2,window=torch.hamming_window(nfft))
    dim1=int(1/(s_rate/2/test.shape[1]))+1
    dim2=test.shape[0]
    dim3=test.shape[1]

    for station in station_list:

        inventory = client.get_stations(network=network,station=station,location='*',channel=channels.split(',')[0],starttime='2009-01-01',endtime='2020-01-01')
        latitude =  inventory[0][0].latitude
        longitude = inventory[0][0].longitude
        df=catalog[catalog['lat']<=latitude+0.5]
        df=df[df['lat']>=latitude-0.5]
        df=df[df['lon']<=longitude+0.5]
        df=df[df['lon']>=longitude-0.5]

        df2=catalog[catalog['lat']<=latitude+2]
        df2=df2[df2['lat']>=latitude-2]
        df2=df2[df2['lon']<=longitude+2]
        df2=df2[df2['lon']>=longitude-2]
        df2=df2.sort_values(by=['time'])
        times_cat2=pd.to_datetime(df2['time'].values)

        print ("Saving seismic waveforms for station %s..."%station)
        print('Number of noise: ', len(df))
        build_database_noise(times_cat2,len(df)*2,network,station,channels,nfft,s_rate,l,dim1,dim2,dim3)
