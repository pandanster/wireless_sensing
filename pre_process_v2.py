'''
Source code for extracting data from raw signal
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
time_slot=1/sample_rate

	
'''
File naming convention s-sine, a-ahosain, r-reflection, g-gesture, cal - standing calibration, calW - wall calibration
''' 

def getAmpPhase(file):
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
		ampFile.write(str(round(amp,10))+'\n')
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


'''
	Builds  spectrogram based from the given timseries data
	for the given window and  from the given location of the file
	based on the sampling rate
'''

def build_spectrogram(inFile,window,start,end,fs):
	file=pd.read_csv(inFile)
	file.columns=['val']
	a=[]
	j=0
	minAmp=0
	maxAmp=0
	count=0
	gfile=open('gplot_file','w')
	time_range=fs/window
	for i in range(0,file.shape[0],window):
		if i+window > file.shape[0]:
			break
		x=abs(np.fft.rfft(file['val'].iloc[i:(i+(window))].values).real).tolist()
		if maxAmp<max(x):
			maxAmp=max(x)
		if count > (start * time_range):
			a.append(x)
		if end != None and count > (end*time_range):
			break
		count+=1
	t=np.linspace(start,start+time_slot*len(a)*window,len(a)).tolist()
	#t=np.linspace(0,time_slot*file.shape[0],len(a)).tolist()
	f=np.fft.rfftfreq(window,1/sample_rate).tolist()
	x,y=np.meshgrid(t,f)
	amplitude=np.array(a)
	print(len(t))
	print(len(f))
	print(amplitude.shape)
	#plt.plot(f,x)
	ax=plt.gca()
	#ax.set_yscale('log')
	plt.pcolormesh(x,y,np.transpose(amplitude),cmap='hot')
	plt.xlabel('Time')
	plt.ylabel('Frequency')
	plt.colorbar(ticks=np.linspace(0,maxAmp,20))
	plt.show()
	
	
#Size of file being used 33748110
#din1ca 134918751
#psanthal 134847240
#getAmpPhase('ahos_1_c')
#windowAverage('psanthal_1_cA_4',1000)
#movingWinFilter('psanthal_1_cA_4_wmean_filt',101):
build_spectrogram('ahos_1_cA',1000000,0,None,sample_rate)
