'''
Source code for extracting data from raw signal
@author Panneer Selvam Santhalingam
2/5/2018


'''

import cmath
import numpy as np
import pandas as pd
from scipy.signal import spectrogram, iirfilter,freqz,decimate,filtfilt
import matplotlib.pyplot as plt
import copy as cp
import glob

sample_rate=10e6 
time_slot=1/sample_rate

	

#File naming convention s-sine, a-ahosain, r-reflection, g-gesture, cal - standing calibration, calW - wall calibration


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

def movingWinFilter(input,window,outFile=True,inFile=True):
	if inFile==True:
		file=pd.read_csv(input,names=['time','val'])
	else:
		file=pd.DataFrame(input,columns=['val'])
	file['mean']=file['val'].rolling(window=window).mean()
	file=file.dropna()
	if outFile==True:
		ampFiltered=open(inFile+'_mwinfilt','w')
	else:
		ampFiltered=[]
	for i in range(file.shape[0]):
		if outFile==True:
			ampFiltered.write(str(file.iloc[i]['time'])+','+str(file.iloc[i]['mean'])+'\n')
		else:
			ampFiltered.append(file.iloc[i]['mean'])
	if outFile==False:
		return ampFiltered

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

def build_spectrogram(input,window,start,end,fs,inFile=True):
	if inFile==True:
		file=pd.read_csv(input,names=['val'])
	else:
		file=pd.DataFrame(input,columns=['val'])
	a=[]
	j=0
	minAmp=0
	maxAmp=0
	count=0
	gfile=open('gplot_file','w')
	time_slot=fs/window
	for i in range(0,file.shape[0],window):
		if i+window > file.shape[0]:
			break
		x=abs(np.fft.rfft(file['val'].iloc[i:(i+(window))].values).real).tolist()
		if maxAmp<max(x):
			maxAmp=max(x)
		if count > (start * time_slot):
			a.append(x)
		if end != None and count > (end*time_slot):
			break
		count+=1
	t=np.linspace(start,start+time_slot*len(a)*window,len(a)).tolist()
	#t=np.linspace(0,time_slot*file.shape[0],len(a)).tolist()
	f=np.fft.rfftfreq(window,1/fs).tolist()
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


def build_filter(Order,pass_band,stop_band,band,fs,filter,ripple,attenuation):
	nyq=fs/2
	if band== 'bandpass':
		pass_low=pass_band/nyq
		pass_high=stop_band/nyq
		stop_low=pass_low*0.8
		stop_high=pass_high/0.8
		wn=[pass_low,pass_high]
	elif band=='lowpass':
		wn=pass_band/nyq
	elif band=='highpass':
		wn=pass_band/nyq
	else:
		return None

	if attenuation!=None:
		b,a = iirfilter(Order,Wn=wn,btype=band,rp=ripple,rs=attenuation,ftype=filter)
		w,h =freqz(b,a)
	elif Order !=None:
		for i in [10,30,40,60,80]:
			b,a = iirfilter(Order,Wn=wn,btype=band,rp=.05,rs=i,ftype=filter)
			w,h =freqz(b,a)
			plt.plot((nyq/np.pi)*w,abs(h),label='stop band attenuation= %d' % i)
		plt.title(filter+' filter frequency response')
		plt.xlabel('Frequency')
		plt.ylabel('Amplitude')
		plt.grid(True)
		plt.legend(loc='best')
		plt.show()	
	elif attenuation==None:
		for i in [2,4,6]:
			b,a = iirfilter(i,Wn=wn,btype=band,rp=.01,rs=40,ftype=filter)
			w,h =freqz(b,a)
			plt.plot((nyq/np.pi)*w,abs(h),label='Order= %d' % i)
		plt.title(filter+' filter frequency response')
		plt.xlabel('Frequency')
		plt.ylabel('Amplitude')
		plt.grid(True)
		plt.legend(loc='best')
		plt.show()
	return (b,a)

def down_sample(sampleFile,factor,order,file=True):
	if file!=True:
		samples=sampleFile
	else:
		samples=pd.read_csv(sampleFile,names=['val']).round(9)
		samples=samples['val'].iloc[:].values
		samples=samples.round(9)
	if factor>9:
		iterations=int(factor/10)
	else:
		iterations=0
	final_iteration=factor%10
	if iterations>0:
		for i in range(0,iterations):
			print(samples.shape)
			samples=decimate(samples,10,order)
	if final_iteration>0:
		samples=decimate(samples,final_iteration,order)
	return samples

def apply_filter(b,a,data):
	return filtfilt(b,a,data)

def plotData(dataFile,filtered,fs1,fs2,file=True):
	time_slot1=1/fs1
	time_slot2=1/fs2
	if file==True:
		data=pd.read_csv(dataFile,names=['val']).round(9)
		data=data['val'].iloc[:].values
	else:
		data=dataFile
	t1=np.linspace(0,time_slot1*len(data),len(data))
	plt.title('time vs amplitude plot')
	plt.plot(t1,data,label='filtered 200Hz')
	t2=np.linspace(0,time_slot2*len(filtered),len(filtered))
	plt.plot(t2,filtered,label='filtered 10Hz to 200Hz')
	plt.legend(loc='best')		
	plt.grid(True)
	plt.show()

def plotDatafromDict(pltDict):
	plt.title('time vs amplitude plot')
	for title,values in pltDict:
		if values['file'] == True:
			data=pd.read_csv(values['data'],names=['val']).round(9)
			data=data['val'].iloc[:].values
		else:
			data=values['data']
		time_slot=1/values['fs']
		time=np.linspace(0,time_slot*len(data),len(data))
		plt.plot(time,data,label=key)
	plt.legend(loc='best')
	plt.grid(True)
	plt.show()

def buildAmplitudes():
	files=glob.glob('*_*')
	for file in files:
		if '.py' not in file:
			getAmpPhase(file)

#Size of file being used 33748110
#din1ca 134918751
#psanthal #134847240
'''
Low pass at 200 Hz order 6 rp=.01, rs=40
High pass at 10 Hz 6 rp=.01 and rs =80


b1,a1=build_filter(6,150.0,None,'lowpass',100000.0,'ellip',.01,40)
b2,a2=build_filter(6,10.0,None,'highpass',5000.0,'ellip',.01,80)
sampled1=down_sample('ahos_1_hA',20,6)
filtered1=apply_filter(b1,a1,sampled1)
sampled2=down_sample(filtered1,12,6,file=False)
filtered2=apply_filter(b2,a2,sampled2)

#noiseRemoved=movingWinFilter(filtered2,100,False,False)
plotData(filtered1,filtered2,100000,5000,file=False)
'''
#getAmpPhase('ahos_1_r')
#windowAverage('psanthal_1_cA_4',1000)
#movingWinFilter('psanthal_1_cA_4_wmean_filt',101):
#plt_dict={'0-100Hz': {'data':filtered1,'file':False,'fs':100000},'10 Hz- 100 Hz': {'data':filtered2,'file':False,'fs':5000}}
#plotDatafromDict(plt_dict)
#build_spectrogram(filtered2,500,0,None,5000,False)

buildAmplitudes()
