'''
Source code for extracting data from raw signal
@author Panneer Selvam Santhalingam
2/5/2018
'''
import sys
import cmath
import numpy as np
import pandas as pd
from scipy.signal import spectrogram, iirfilter,freqz,decimate,filtfilt,correlate
#import matplotlib.pyplot as plt
import copy as cp
import glob
import multiprocessing as mp
#from scipy.spatial.distance import euclidean
import time
import os
#import mlpy
import get_features as gf
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
import io
#import pydotplus
from numpy import trapz
import math
sample_rate=10e6 
time_slot=1/sample_rate

	

#File naming convention s-sine, a-ahosain, r-reflection, g-gesture, cal - standing calibration, calW - wall calibration


def getFiltered(file,b1=None,a1=None,b2=None,a2=None,low=None,high=None,mode=0):
	f=open(file,'rb')
	data=np.fromfile(f,"complex64",-1,"")
	file_amp=[]
	for val in data:
		amp,phase=cmath.polar(val)
		file_amp.append(amp)
	if mode==1:
		dcMean=gf.getAbsMean(np.array(file_amp))
		file_amp=np.array(file_amp)-dcMean
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

def getAmp(file,sample_rate):
	time_slot=1/sample_rate
	sample_time=0
	f=open(file,'rb')
	output=open(file+'_'+'A','w')
	data=np.fromfile(f,"complex64",-1,"")
	for val in data:
		sample_time+=time_slot
		amp,phase=cmath.polar(val)
		output.write(str(sample_time)+','+str(round(amp,9))+'\n')
	return

def getPhase(file,sample_rate):
	time_slot=1/sample_rate
	sample_time=0
	f=open(file,'rb')
	output=open(file+'_'+'P','w')
	data=np.fromfile(f,"complex64",-1,"")
	for val in data:
		sample_time+=time_slot
		amp,phase=cmath.polar(val)
		output.write(str(round(sample_time,9))+','+str(round(phase,9))+'\n')
	return

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

def plotAvgfreq(input,window,start,end,fs,time=None,inFile=True):
	if inFile==True:
		file=pd.read_csv(input,names=['time','val'])
	else:
		file=pd.DataFrame(input,columns=['val'])
	a=[]
	j=0
	minAmp=0
	maxAmp=0
	count=0
	time_slot=window/fs
	f=np.fft.rfftfreq(window,1/fs).tolist()
	for i in range(0,file.shape[0],window):
		if i+window > file.shape[0]:
			break
		x=abs(np.fft.rfft(file['val'].iloc[i:(i+(window))].values).real).tolist()
		y=[x[i]*f[i] for i in range(len(x))]
		if maxAmp<max(x):
			maxAmp=max(x)
		if count > (start * time_slot):
			a.append(np.mean(y)+np.std(y))
		if end != None and count > (end*time_slot):
			break
		count+=1
	t=np.linspace(start,start+time_slot*len(a),len(a)).tolist()
	if time == None:
		outfile=open('seg-5','a')
		currMax=0
		currStart=0
		currEnd=0
		currK=0
		currIndex1=0
		currIndex2=0
		for k in [15,20,25,30,35,40]:
			#[13,16,19,22,25,28,31,34,37,40]:
			#[12,14,16,18,20,22,24,26,28,30,32,34,36,38,40]:
			maxArea=0
			start=0
			end=0
			index1=0
			index2=0
			for i in range(10,len(a)-k):
				if (trapz(a[i:i+k],dx=time_slot)**2)*math.sqrt(np.var(a[i:i+k]))/k > maxArea:
					maxArea=(trapz(a[i:i+k],dx=time_slot)**2)*math.sqrt(np.var(a[i:i+k]))/k
					start=t[i]
					end=t[i+k]
					index1=i
					index2=i+k
			if maxArea>currMax:
				currMax=maxArea
				currStart=start
				currEnd=end
				currK=k
				currIndex1=index1
				currIndex2=index2
		compArea = (trapz(a[:currIndex1]+a[currIndex2:],dx=time_slot)**2)*math.sqrt(np.var(a[:currIndex1]+a[currIndex2:]))/len(a[:currIndex1])+len(a[currIndex2:])
		if compArea >currMax:
			status='signal not ok'
		elif currK/len(a) > .60 or currK/len(a) < .25 :
			status='signal not ok'
		else:
			status='signal ok'
		outfile.write(input+','+str(currStart)+','+str(currEnd)+','+str(round(currStart*5000))+','+str(round(currEnd*5000))+','+str(currK)+','+status+'\n')
	if time !=None:
		start=0
		end=0
		maxArea=0
		k=time
		outfile=open('segmentations','a')
		for i in range(len(a)-k):
			if (trapz(a[i:i+k],dx=time_slot)**2)*math.sqrt(np.var(a[i:i+k]))/k > maxArea:
				maxArea=(trapz(a[i:i+k],dx=time_slot)**2)*math.sqrt(np.var(a[i:i+k]))/k 
				start=t[i]
				end=t[i+k]
		outfile.write(input+','+str(start)+','+str(end)+','+str(round(start*5000))+','+str(round(end*5000))+'\n')


	'''		
	plt.plot(t,a)
	plt.xlabel('Time')
	plt.ylabel('Amplitude')
	plt.show()
	'''

def buildAvgfreq(input,window,start,end,fs,time=None,inFile=True):
	if inFile==True:
		file=pd.read_csv(input,names=['time','val'])
	else:
		file=pd.DataFrame(input,columns=['val'])
	a=[]
	j=0
	minAmp=0
	maxAmp=0
	count=0
	time_slot=window/fs
	f=np.fft.rfftfreq(window,1/fs).tolist()
	for i in range(0,file.shape[0],window):
		if i+window > file.shape[0]:
			break
		x=abs(np.fft.rfft(file['val'].iloc[i:(i+(window))].values).real).tolist()
		y=[x[i]*f[i] for i in range(len(x))]
		if maxAmp<max(x):
			maxAmp=max(x)
		if count > (start * time_slot):
			a.append(np.mean(y)+np.std(y))
		if end != None and count > (end*time_slot):
			break
		count+=1
	t=np.linspace(start,start+time_slot*len(a),len(a)).tolist()
	outfile=open('../spectrogram/'+input+'_spec','w')
	for i in range(len(a)):
		outfile.write(str(t[i])+','+str(a[i])+'\n')


def build_phaseogram(input,window,start,end,fs,inFile=True):
	if inFile==True:
		file=pd.read_csv(input,names=['time','val'])
	else:
		file=pd.DataFrame(input,columns=['val'])
	a=[]
	j=0
	minAmp=0
	maxAmp=0
	count=0
	time_slot=window/fs
	print(file.shape[0])
	for i in range(0,file.shape[0],window):
		if i+window > file.shape[0]:
			break
		x=np.angle(np.fft.fft(file['val'].iloc[i:(i+(window))].values)).tolist()
		if maxAmp<max(x):
			maxAmp=max(x)
		if count > (start * time_slot):
			a.append(np.mean(x))
		if end != None and count > (end*time_slot):
			break
		count+=1
	print(count)
	print(time_slot*len(a))
	print(len(a))
	t=np.linspace(start,start+time_slot*len(a),len(a)).tolist()
	'''
	plt.plot(t,a)
	plt.xlabel('Time')
	plt.ylabel('Frequency')
	plt.show()
	'''

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
	files=glob.glob('*astand*')
	#mp.set_start_method('fork')
	b1,a1=build_filter(6,high,None,'lowpass',100000.0,'ellip',.01,40)
	b2,a2=build_filter(6,low,None,'highpass',5000.0,'ellip',.01,80)	
	for file in files:
		if '.py' not in file and 'Hz' not in file:
			p=mp.Process(target=getAmpPhase,args=(file,b1,a1,b2,a2,low,high,1,))
			p.start()
			p.join()
	return

def buildSegments(blob,window,time):
	files=glob.glob(blob)
	#mp.set_start_method('fork')
	for file in files:
		if '.py' not in file:
			p=mp.Process(target=plotAvgfreq,args=(file,window,0,None,5000,time,))
			p.start()
			p.join()
	return

def buildAvgSpectrogram(blob,window,time):
	files=glob.glob(blob)
	#mp.set_start_method('fork')
	for file in files:
		if '.py' not in file:
			p=mp.Process(target=buildAvgfreq,args=(file,window,0,None,5000,time,))
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
	data=pd.read_csv(database,names=['name','starttime','endtime','start','end','len','signal-qual','label'])
	data=data[data['label']==label]
	totalDist=0
	totalcount=0
	f=open('time-dtw-5','a')
	for i in range(data.shape[0]):
		if data.iloc[i]['name']==inFile:
			continue
		compFile=pd.read_csv(data.iloc[i]['name'],names=['time','val'])
		compFile=compFile['val'].iloc[data.iloc[i]['start']:data.iloc[i]['end']]
		totalDist+=gf.getDTWDist(inData,compFile.tolist())
		totalcount+=1
	f.write(str(totalDist/totalcount)+','+inFile+','+label+'\n')

def getMeanSpectrogramDist(database,inData,inFile,label):
	data=pd.read_csv(database,names=['name','starttime','endtime','start','end','len','signal-qual','label'])
	data=data[data['label']==label]
	totalDist=0
	totalcount=0
	f=open('spec-dtw-5','a')
	for i in range(data.shape[0]):
		if data.iloc[i]['name']==inFile:
			continue
		compFile=pd.read_csv('../spectrogram/'+data.iloc[i]['name']+'_spec',names=['time','val'])
		compFile=compFile[compFile['time']>=data.iloc[i]['starttime']]
		compFile=compFile[compFile['time']<=data.iloc[i]['endtime']]
		compFile=compFile['val']
		totalDist+=gf.getDTWDist(inData,compFile.tolist())
		totalcount+=1
	f.write(str(totalDist/totalcount)+','+inFile+','+label+'\n')

def getFeatures(database,inData,specData,inFile,inLabel,labels):
	featureVector=[]
	for label in labels:
		p=mp.Process(target=getMeanDist,args=(database,inData,inFile,label,))
		p.start()
		p.join()
	for label in labels:
		p=mp.Process(target=getMeanSpectrogramDist,args=(database,specData,inFile,label,))
		p.start()
		p.join()
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

def writeDistance(dataBase,data,data1,file1,index,start,end,startI,endI):
	outFile=open('/scratch/psanthal/seg2-'+str(startI)+'-'+str(endI-1)+'/seg2'+'_'+str(index)+'_'+str(end),'w')
	outputs=[]
	for i in range(start,end):
		data2=dataBase[data.iloc[i]['name']]
		data2=data2.tolist()
		label=data.iloc[i]['label']
		file2=data.iloc[i]['name']
		outputs.append(file1+','+file2+','+str(gf.getDTWDist(data1,data2))+','+label+'\n')
#		print(file1+','+file2+','+str(gf.getDTWDist(data1,data2))+','+label+'\n')
	for line in outputs:
		outFile.write(line)

def computeDistances(dataFile,startI,endI):
	data=pd.read_csv(dataFile,names=['name','starttime','endtime','start','end','len','signal-qual','label'])
	dataBase={}
	for i in range(data.shape[0]):
		inputData=pd.read_csv(data.iloc[i]['name'],names=['time','val'])
		inputData=inputData['val'].iloc[data.iloc[i]['start']:data.iloc[i]['end']]
		dataBase[data.iloc[i]['name']]=inputData
	processes=[]
	count=0
	toWrite=[]
	#outQueue=mp.Queue()
	for i in range(startI,endI):
		input1=dataBase[data.iloc[i]['name']]
		for j in range(i+1,data.shape[0],100):
			if j+100>data.shape[0]:
				end=data.shape[0]
			else:
				end=j+100
			p=mp.Process(target=writeDistance,args=(dataBase,data,input1.tolist(),data.iloc[i]['name'],i,j,end,startI,endI,))
			if count>16:
				for k in processes:
					k.join()
					processes=[]
					count=0
			p.start()
			processes.append(p)
	#		if not outQueue.empty():
	#			toWrite.append(outQueue.get())
			count+=1
	#f=open('temp','w')
	#for line in toWrite:
	#	f.write(line)
	if count > 0:
		for k in processes:
			k.join()
	return

def writeSpectrogramDistance(dataBase,data,data1,file1,index,start,end,startI,endI):
	outFile=open('/scratch/psanthal/spect_distances/seg2-'+str(startI)+'-'+str(endI-1)+'/seg2'+'_'+str(index)+'_'+str(end),'w')
	outputs=[]
	for i in range(start,end):
		data2=dataBase[data.iloc[i]['name']]
		data2=data2.tolist()
		label=data.iloc[i]['label']
		file2=data.iloc[i]['name']
		outputs.append(file1+','+file2+','+str(gf.getDTWDist(data1,data2))+','+label+'\n')
#		print(file1+','+file2+','+str(gf.getDTWDist(data1,data2))+','+label+'\n')
	for line in outputs:
		outFile.write(line)

def computeSpectrogramDistances(dataFile,startI,endI):
	data=pd.read_csv(dataFile,names=['name','starttime','endtime','start','end','len','signal-qual','label'])
	dataBase={}
	for i in range(data.shape[0]):
		inputData=pd.read_csv(data.iloc[i]['name'],names=['time','val'])
		inputData=inputData[inputData['time']>=data.iloc[i]['starttime']]
		inputData=inputData[inputData['time']<=data.iloc[i]['endtime']]
		inputData=inputData['val']
		dataBase[data.iloc[i]['name']]=inputData
	processes=[]
	count=0
	toWrite=[]
	#outQueue=mp.Queue()
	for i in range(startI,endI):
		input1=dataBase[data.iloc[i]['name']]
		for j in range(i+1,data.shape[0],100):
			if j+100>data.shape[0]:
				end=data.shape[0]
			else:
				end=j+100
			p=mp.Process(target=writeSpectrogramDistance,args=(dataBase,data,input1.tolist(),data.iloc[i]['name'],i,j,end,startI,endI,))
			if count>16:
				for k in processes:
					k.join()
					processes=[]
					count=0
			p.start()
			processes.append(p)
	#		if not outQueue.empty():
	#			toWrite.append(outQueue.get())
			count+=1
	#f=open('temp','w')
	#for line in toWrite:
	#	f.write(line)
	if count > 0:
		for k in processes:
			k.join()
	return

def buildFeatures(dataFile,labels):
	data=pd.read_csv(dataFile,names=['name','starttime','endtime','start','end','len','signal-qual','label'])
#	mp.set_start_method('fork')
	#for i in range(data.shape[0]):
	for i in range(0,1):
		inData=pd.read_csv(data.iloc[i]['name'],names=('time','val'))
		specData=pd.read_csv('../spectrogram/'+data.iloc[i]['name']+'_spec',names=('time','val'))
		inData=inData['val'].iloc[data.iloc[i]['start']:data.iloc[i]['end']]
		specData=specData[specData['time']>=data.iloc[i]['starttime']]
		specData=specData[specData['time']<=data.iloc[i]['endtime']]
		specData=specData['val']
		p=mp.Process(target=getFeatures,args=(dataFile,inData,specData,data.iloc[i]['name'],data.iloc[i]['label'],labels,))
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
def createLabels(dataFile):
	f=open(dataFile,'r')
	outfile=open(dataFile+'_labeled','w')
	lines=f.readlines()
	for line in lines:
		tokens=line.strip().split(',')
		label=tokens[0].split('_')[1]
		outfile.write(line.strip()+','+label+'\n')

def computeLabelAverageDistances(labels):	
	files=glob.glob('*_*')
	average_distances={}
	for file in files:
		if 'merged' in file:
			f=open('file','r')
			lines=f.readlines()
			for line in lines:
				file1,file2,distance,label=lines.strip().split(',')
				label1=file1.split('_')[1]
				label2=file2.split('_')[1]
				try:
					average_distances[file1][label2]+=distance
				except:
					average_distances[file1]={}
					for label in labels:
						average_distances[file1][label]=0
					average_distances[file1][label2]+=distance
				try:
					average_distances[file2][label1]+=distance
				except:
					average_distances[file2]={}
					for label in labels:
						average_distances[file2][label]=0
					average_distances[file2][label1]+=distance
	outfile=open('dtw_time_seg-5','w')
	for key in average_distances.keys():
		towrite=[]
		towrite.append(label)
		for label in labels:
			towrite.append(str(average_distances[key][label]))
		outfile.write(','.join(towrite)+'\n')
	return	

	
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
#print(plotAvgfreq('alamin_alarm_3_10-100Hz',500,0,None,5000,20))
#build_spectrogram('ahosain_5_1_10-100Hz',500,0,None,5000)
#buildFeatures('seg-5_labeled',['alarm','call','snow','rain','turnon','email','time','therm','open','close','wakeup','interpreter','ac'])
#makePredictions('feautreFile',70,69)
#buildPreProcessed(10,100)
#getAmp('ahosain_5_1',sample_rate)
#buildSegments('*10-100*',500,None)
#buildAvgSpectrogram('*10-100*',500,None)
#plotAvgfreq('ding_turnon_12_10-100Hz',500,0,None,5000)
#createLabels('seg-5')
#if __name__ == '__main__':
#start=int(sys.argv[1])
#end=int(sys.argv[2])
#computeDistances('seg-5_labeled',start,end)
#computeSpectrogramDistances('seg-5_labeled',start,end)
computeLabelAverageDistances(['alarm','call','snow','rain','turnon','email','time','therm','open','close','wakeup','interpreter','ac'])