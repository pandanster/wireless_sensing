import sys


sys.settrace
'''
Source code for extracting data from raw signal
@author Panneer Selvam Santhalingam
2/5/2018
'''

import cmath
import numpy as np
import pandas as pd
from scipy.signal import spectrogram, iirfilter,freqz,decimate,filtfilt,correlate
import matplotlib.pyplot as plt
import copy as cp
import glob
import multiprocessing as mp
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
import time
import os
#import mlpy
import get_features as gf
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
import io
import pydotplus

sample_rate=10e6 
time_slot=1/sample_rate

	

#File naming convention s-sine, a-ahosain, r-reflection, g-gesture, cal - standing calibration, calW - wall calibration


def getAmpPhase(file,b1=None,a1=None,b2=None,a2=None,low=None,high=None,mode=0):
	f=open(file,'rb')
	data=np.fromfile(f,"complex64",-1,"")
	file_amp=[]
	for val in data:
		amp,phase=cmath.polar(val)
		file_amp.append(amp)
	if mode==1:
		print('came to mode')
		sampled1=down_sample(np.array(file_amp),20,6,file=False)
		filtered1=apply_filter(b1,a1,sampled1)
		sampled2=down_sample(filtered1,12,6,file=False)
		filtered2=apply_filter(b2,a2,sampled2)
		output=open(file+'_'+str(low)+'-'+str(high)+'Hz','w')
		time_slot=1/5000
		time=np.linspace(0,time_slot*len(filtered2),len(filtered2))
		for i in range(len(time)):
			output.write(str(time[i].round(9))+','+str(filtered2[i].round(9))+'\n')
		output.close()
		f.close()
		return
	else:
		return np.array(file_amp)
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
		file=pd.read_csv(input,names=['time','val'])
	else:
		file=pd.DataFrame(input,columns=['val'])
	a=[]
	j=0
	minAmp=0
	maxAmp=0
	count=0
	gfile=open('gplot_file','w')
	time_slot=window/fs
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
	t=np.linspace(start,start+time_slot*len(a),len(a)).tolist()
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

def buildPreProcessed(low,high):
	files=glob.glob('*ahosain*')
	#mp.set_start_method('fork')
	b1,a1=build_filter(6,high,None,'lowpass',100000.0,'ellip',.01,40)
	b2,a2=build_filter(6,low,None,'highpass',5000.0,'ellip',.01,80)	
	for file in files:
		if '.py' not in file and 'Hz' not in file:
			p=mp.Process(target=getAmpPhase,args=(file,b1,a1,b2,a2,low,high,1,))
			p.start()
			p.join()
	return

def buildFilteredFile(b1,a1,b2,a2,file,low,high):
	sampled1=down_sample(file,20,6)
	filtered1=apply_filter(b1,a1,sampled1)
	sampled2=down_sample(filtered1,12,6,file=False)
	filtered2=apply_filter(b2,a2,sampled2)
	output=open(file+'_'+str(low)+'-'+str(high)+'Hz','w')
	time_slot=1/5000
	time=np.linspace(0,time_slot*len(filtered2),len(filtered2))
	for i in range(len(time)):
		output.write(str(time[i].round(9))+','+str(filtered2[i].round(9))+'\n')
	output.close()
	return

def buildFiltered(low,high):
	files=glob.glob('*ding*')
	mp.set_start_method('fork')
	b1,a1=build_filter(6,high,None,'lowpass',100000.0,'ellip',.01,40)
	b2,a2=build_filter(6,low,None,'highpass',5000.0,'ellip',.01,80)
	for file in files:
		if '.py' not in file and 'Hz' not in file and 'A' in file:
			p=mp.Process(target=buildFilteredFile,args=(b1,a1,b2,a2,file,low,high,))
			p.start()
			p.join()
	return

def buildLabels(file,train,test):
	data=pd.read_csv(file,names=['name','start','end','label'])
	trainData=data.iloc[:train]
	testData=data.iloc[test:]
	output=open('results','w')
	database=[]
	for i in range(train):
		file=pd.read_csv(trainData.iloc[i]['name'],names=('time','val'))
		file=file['val'].iloc[trainData.iloc[i]['start']:trainData.iloc[i]['end']]
		database.append([file.tolist(),trainData.iloc[i]['label'],trainData.iloc[i]['name']])
	for i in range(testData.shape[0]):
		predictions={'h1':0,'h2':0,'h3':0,'5':0,'m':0}
		distances=[]
		testFile=pd.read_csv(testData.iloc[i]['name'],names=('time','val'))
		testFile=testFile['val'].iloc[testData.iloc[i]['start']:testData.iloc[i]['end']]
		for j in range(train):
			distance,path=fastdtw(testFile.tolist(),database[j][0],dist=euclidean)
			distances.append([distance,database[j][1],database[j][2]])
#			print(str(j)+','+str(time.time()))
		sortedDistances=sorted(distances,key=lambda x: x[0])
		output.write(str(sortedDistances))
		output.write('\n')
		for k in range(10):
			predictions[sortedDistances[k][1]]+=1
		max=-1
		predicted=None
		for key in predictions.keys():
			if max< predictions[key]:
				max=predictions[key]
				predicted=key
#		print("Predicted: "+predicted+" Actual: "+testData.iloc[i]['label']+'\n')
		output.write("Predicted: "+predicted+" Actual: "+testData.iloc[i]['label']+' File: '+testData.iloc[i]['name']+'\n')
		

def getCorrelated(file1,file2):
	sig1=pd.read_csv(file1,names=['time','val'])
	sig2=pd.read_csv(file2,names=['time','val'])
	correlated=correlate(sig1['val'].values,sig2['val'].values,mode='same')
	output=open(file1[:-9]+'_corr','w')
	for i in range(correlated.shape[0]):
		output.write(str(correlated[i])+'\n')

def buildCorrelated():
	files=glob.glob('*Hz*')
	for file in files:
		if '.py' not in file and 'Hz' in file and os.path.isdir(file) == False:
			getCorrelated(file[:-9]+'_10-100Hz',file[:-9]+'_10-150Hz')
				

def getMeanDist(database,inData,inFile,label):
	data=pd.read_csv(database,names=['name','start','end','label'])
	data=data[data['label']==label]
	totalDist=0
	totalcount=0
	for i in range(data.shape[0]):
		if data.iloc[i]['name']==inFile:
			continue
		compFile=pd.read_csv(data.iloc[i]['name'],names=['time','val'])
		compFile=compFile['val'].iloc[data.iloc[i]['start']:data.iloc[i]['end']]
		totalDist+=gf.getDTWDist(inData,compFile.tolist())
		totalcount+=1
	return totalDist/totalcount

def getFeatures(database,inData,inFile,inLabel,labels):
	featureVector=[]
	outFile=open('feautreFile','a')
	for label in labels:
		featureVector.append(getMeanDist(database,inData,inFile,label))
	featureVector.append(gf.getMean(inData))
	featureVector.append(gf.getArea(inData))
	featureVector.append(gf.getAbsMean(inData))
	featureVector.append(gf.getAbsArea(inData))
	featureVector.append(gf.getEntropy(inData))
	featureVector.append(gf.getSkew(inData))
	featureVector.append(gf.getKur(inData))
	quartiles=gf.getQuartiles(inData)
	featureVector.append(gf.getIQR(quartiles[2],quartiles[1]))
	featureVector.append(','.join([str(x) for x in gf.getFFTPeaks(inData)]))
	featureVector.append(gf.getEnergy(inData))
	featureVector=[str(x) for x in featureVector]
	outFile.write(','.join(featureVector)+','+inFile+','+inLabel+'\n')
	return

def buildFeatures(dataFile,labels):
	data=pd.read_csv(dataFile,names=['name','start','end','label'])
	mp.set_start_method('fork')
	for i in range(data.shape[0]):
		inData=pd.read_csv(data.iloc[i]['name'],names=('time','val'))
		inData=inData['val'].iloc[data.iloc[i]['start']:data.iloc[i]['end']]
		p=mp.Process(target=getFeatures,args=(dataFile,inData,data.iloc[i]['name'],data.iloc[i]['label'],labels,))
		p.start()
		p.join()
	return

def buildTree(trainData,trainLabels,depth):
	dtc= DecisionTreeClassifier(max_depth=depth)
	dtc.fit(trainData.values,trainLabels.values)
	return dtc

def printTree(dtree,features):
	dotfile=io.StringIO()
	export_graphviz(decision_tree=dtree,feature_names=features,out_file=dotfile,class_names=['5','h1','h2','h3','m'])
	graph=pydotplus.graph_from_dot_data(dotfile.getvalue())
	graph.write_png("dtree.png")

def makePredictions(dataFile,train,test):
	data=pd.read_csv(dataFile,header=None)
	trainData=data.iloc[:train]
	testData=data.iloc[test:]
	print(data.shape,trainData.shape,testData.shape)
	output=open('results','w')
	dtc=buildTree(trainData.iloc[:,0:19],trainData.iloc[:,20:21],5)
	printTree(dtc,['h1','h2','h3','5','m','mean','area','abs_mean','abs_area','entropy','skew','kur','iqr','fft1','fft2','fft3','fft4','fft5','energy'])
	'''
	for i in range(testData.shape[0]):
		predicted=dtc.predict(testData.iloc[i,0:19].values.reshape(1,-1))
		print('predicted')
		print(predicted)
		print('Actual')
		print(testData.iloc[i,20:21].values)
		print('next\n')
		'''
#Size of file being used 33748110
#din1ca 134918751
#psanthal #134847240
'''
Low pass at 200 Hz order 6 rp=.01, rs=40
High pass at 10 Hz 6 rp=.01 and rs =80
'''
#windowAverage('psanthal_1_cA_4',1000)
#movingWinFilter('psanthal_1_cA_4_wmean_filt',101):
#plt_dict={'0-100Hz': {'data':filtered1,'file':False,'fs':100000},'10 Hz- 100 Hz': {'data':filtered2,'file':False,'fs':5000}}
#plotDatafromDict(plt_dict)
#build_spectrogram('ahosain_5_1_10-100Hz',500,0,None,5000)
#buildFeatures('segmentations.csv',['h1','h2','h3','5','m'])
#makePredictions('feautreFile',70,69)
#buildPreProcessed(10,100)