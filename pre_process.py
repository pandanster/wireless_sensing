'''
Source g1A for extracting data from raw signal
@author Panneer Selvam Santhalingam
2/5/2018
'''

import cmath
import numpy as np
from scipy.signal import savgol_filter
from scipy import signal
import pandas as pd
from scipy.signal import spectrogram
import matplotlib.pyplot as plt
import copy as cp

sample_rate=10e6 

'''
File naming convention s-sine, a-ahosain, r-reflection, g-gesture, cal - standing calibration, calW - wall calibration
''' 
def getAmpPhase(file):
	time_slot=1/sample_rate
	sample_time=0
	ampArray=[]
	phaseArray=[]
	count=1
	ampFile=open(file+'A','w')
	#phaseFile=open("sarg41P",'w')
	f=open(file,'rb')
	data=np.fromfile(f,"complex64",-1,"")
	for val in data:
		sample_time+=time_slot
		amp,phase=cmath.polar(val)
		ampFile.write(str(sample_time)+','+str(amp)+'\n')
	#	phaseFile.write(str(sample_time)+','+str(phase)+'\n')

def movingWinFilter(inFile,window):
	file=pd.read_csv(inFile)
	file.columns=['time','val']
	file['mean']=file['val'].rolling(window=window).mean()
	file=file.dropna()
	ampFiltered=open(inFile+'_mwinfilt','w')
	for i in range(file.shape[0]):
		ampFiltered.write(str(file.iloc[i]['time'])+','+str(file.iloc[i]['mean'])+'\n')

def windowAverage(inFile,window):
	file=pd.read_csv(inFile)
	file.columns=['time','val']
	time_slot=(1/sample_rate)*window
	file_count=file.shape[0]
	current_index=0
	sample_time=file.iloc[0]['time']
	outFile=open(inFile+'_wmean_filt','w')
	while current_index<file_count:
		sample_time+=time_slot
		outFile.write(str(sample_time)+','+str(file.iloc[current_index:(current_index+window)].mean()['val'])+'\n')
		current_index+=window

def windowMedian(inFile,window):
	file=pd.read_csv(inFile)
	file.columns=['time','val']
	time_slot=(1/sample_rate)*window
	file_count=file.shape[0]
	current_index=0
	sample_time=file.iloc[0]['time']
	outFile=open(inFile+'_wmed_filt','w')
	while current_index<file_count:
		sample_time+=time_slot
		outFile.write(str(sample_time)+','+str(file.iloc[current_index:(current_index+window)].median()['val'])+'\n')
		current_index+=window

def build_spectrogram(inFile,window):
	file=pd.read_csv(inFile)
	file.columns=['time','val']
	a=[]
	fs=sample_rate/1000
	j=0
	minAmp=0
	maxAmp=0
	for i in range(0,file.shape[0],window):
		if i+window > file.shape[0]:
			break
		x=abs(np.fft.rfft(file['val'].iloc[i:(i+(window))].values).real).tolist()
		if maxAmp<max(x):
			maxAmp=max(x)
		a.append(x)
		
	#x,y=np.meshgrid(f,t)
	t=np.linspace(file['time'].iloc[0],file['time'].iloc[-1],len(a)).tolist()
	f=np.fft.rfftfreq(window,1/fs).tolist()
	amplitude=np.array(a)
	print(len(t))
	print(len(f))
	print(amplitude.shape)
	plt.pcolormesh(f,t,amplitude,cmap='hot')
	plt.ylabel('Time')
	plt.xlabel('Frequency')
	plt.colorbar(ticks=np.linspace(0,maxAmp,20))
	plt.show()


	
#Size of file being used 33748110
#din1ca 134918751
#psanthal 134847240
#getAmpPhase('psanthal_1_c')
#windowAverage('psanthal_1_cA_4',1000)
#movingWinFilter('psanthal_1_cA_4_wmean_filt',101):
build_spectrogram('ahos_1_hA_wmean_filt',1000)