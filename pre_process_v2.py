import sys


sys.settrace
'''
Source code for extracting data from raw signal
@author Panneer Selvam Santhalingam
2/5/2018
'''
import numpy
import cmath
import numpy as np
import pandas as pd
from scipy.signal import spectrogram, iirfilter,freqz,decimate,filtfilt,correlate
import matplotlib.pyplot as plt
import copy as cp
import glob
import multiprocessing as mp
from scipy.spatial.distance import euclidean
#from fastdtw import fastdtw
import time
import os
import mlpy
import get_features as gf
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from matplotlib.ticker import NullLocator
#from scipy.signal import flattop
import io
import pydotplus
from numpy import trapz
import math
import time
from scipy.io.wavfile import write
import warnings
import pickle
from sklearn.decomposition import PCA
from PIL import Image
import random
from scipy.spatial import Delaunay
from scipy.spatial import ConvexHull
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
		file=pd.read_csv(input,names=['val'])
	else:
		file=pd.DataFrame(input,columns=['val'])
	file['mean']=file['val'].rolling(window=window).mean()
	file=file.dropna()
	if outFile==True:
		ampFiltered=open(input+'_mwinfilt','w')
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
	file.columns=['val']
	time_slot=(1/sample_rate)*window
	file_count=file.shape[0]
	current_index=0
	outFile=open(inFile+'_wmean_filt','w')
	while current_index<file_count:
		outFile.write(str(file.iloc[current_index:(current_index+window)].mean()['val'])+'\n')
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
def safe_ln(x,min_val=.0000000001):
	return np.log(x.clip(min=min_val))

'''
	Builds  spectrogram based from the given timseries data
	for the given window and  from the given location of the file
	based on the sampling rate
'''

def build_spectrogram(input,npoint,window,start,end,fs,padding=None,inFile=True,saveFile=False,title=None,plotShort=False,logScale=False,saveDir=None,Seg=False,chop=False):
	if inFile==True:
		file=pd.read_csv(input,names=['val'])
	else:
		file=pd.DataFrame(input,columns=['val'])
#	file=file[35381700:59861700]
	a=[]
	j=0
	minAmp=0
	maxAmp=0
	count=0
	time_slot=window/fs
	window_func=np.hamming(npoint)
	#window_func=flattop(npoint)
	f=np.fft.rfftfreq(npoint,1/fs).tolist()
	#f=np.linspace(-500,500,400)
	m=0
	n=0
	for freq in f:
                if freq < 10:
                        m+=1
                elif freq <= 1000:
                        n+=1
                else:
                        break
	data=np.concatenate((np.zeros(npoint-padding),file['val'].values,np.zeros(npoint-padding)))
	for i in range(0,data.shape[0],window):
		if i+(npoint-padding) > data.shape[0]:
			break
		if padding ==0:
			if Seg:
				x=abs(np.fft.rfft(data[i:(i+(npoint))]).real).tolist()
			elif logScale:
				with warnings.catch_warnings():
					warnings.simplefilter("ignore")
					x=abs(np.fft.rfft(data[i:(i+(npoint))]).real)
					x=np.where(x>0,np.log(x),0).tolist()
			else:
				x=abs(np.fft.rfft(data[i:(i+(npoint))]).real).tolist()
		
		else:
			if Seg:
				x=abs(np.fft.rfft(np.concatenate((data[i:(i+(npoint-padding))],np.zeros(padding)))).real).tolist()
			elif logScale:
				with warnings.catch_warnings():
					warnings.simplefilter("ignore")
					x=abs(np.fft.rfft(np.concatenate((data[i:(i+(npoint-padding))],np.zeros(padding)))).real)
					x=np.where(x>0.0,np.log(x),0).tolist()
			else:
				x=abs(np.fft.rfft(np.concatenate((data[i:(i+(npoint-padding))],np.zeros(padding)))).real).tolist()
		if maxAmp<max(x):
			maxAmp=max(x)
		if minAmp>min(x):
			minAmp=min(x[0:n])
		if count > (start * time_slot):
			if Seg:
				x=x[:n]
				'''
				temp=[]
				for i in range(len(x)):
					if x[i] > 1:
						temp.append(f[i])
				if len(temp) >0:
					a+=[max(temp)]
				else:
					a+=[0]
				'''
				y=[x[i]*f[i] for i in range(len(x))]
				a+=[np.mean(y)+np.std(y)]
				continue
			elif saveFile:
				a.append(x[:n])
			elif plotShort:
				a.append(x[:n])
			else:
				a.append(x)
		if end != None and count > (end*time_slot):
			break
		count+=1
#	a.append((np.zeros(len(x[:n]))+20).tolist())
	if Seg:
		a=smooth(a)
	t=np.linspace(start,start+time_slot*len(a),len(a)).tolist()
	if saveFile:
		f=f[:n]
	if plotShort:
		f=f[:n]
	

	if saveFile and not Seg:
		plt.rcParams["figure.figsize"] = [1,4]
		x,y=np.meshgrid(t,f)
		amplitude=np.array(a)
		ax=plt.gca()
		ax.set_axis_off()
		plt.margins(0,0)
		ax.xaxis.set_major_locator(NullLocator())
		ax.yaxis.set_major_locator(NullLocator())
		plt.ioff()
		plt.axis('off')
		plt.pcolormesh(x,y,np.transpose(amplitude),cmap='gist_heat')
		if title is None:
			plt.savefig(saveDir+'/'+input+'.jpeg',bbox_inches='tight',pad_inches=0)
		else:
			plt.savefig(saveDir+'/'+title+'.jpeg',bbox_inches='tight',pad_inches=0)
	elif Seg:
		'''
		points=[[x,y] for x,y in zip(t,a)]
		points=np.array(points)
		hull=ConvexHull(points)
		for simplex in hull.simplices:
			plt.plot(points[simplex, 0], points[simplex, 1], 'k-')
		plt.plot(points[hull.vertices,0], points[hull.vertices,1], 'r--', lw=2)
		plt.plot(points[hull.vertices[0],0], points[hull.vertices[0],1], 'ro')
		'''
		print(len(t))
		plt.plot(t,a)
		plt.show()
		#plt.plot(t,a)
		#plt.show()		
	else:
		x,y=np.meshgrid(t,f)
		amplitude=np.array(a)
		if title is not None:
			plt.title(title)
		plt.xlabel('Time')
		plt.ylabel('Frequency')
		plt.pcolormesh(x,y,np.transpose(amplitude),cmap='gist_heat')
		plt.colorbar()
		plt.show()

def smooth(x,window_len=70,window='hanning'): 
    #print("came here")
    #if x.ndim != 1:
    #    raise ValueError, "smooth only accepts 1 dimension arrays."

    #if x.size < window_len:
    #    raise ValueError, "Input vector needs to be bigger than window size."


    #if window_len<3:
    #    return x


    #if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
    #    raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"


    s=numpy.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=numpy.ones(window_len,'d')
    else:
        w=eval('numpy.'+window+'(window_len)')

    y=numpy.convolve(w/w.sum(),s,mode='valid')
    return y

def build_spectrogram_seg(input,npoint,window,start,end,fs,a1,b1,a2,b2,padding=None,inFile=True,saveFile=False,title=None,plotShort=False,logScale=False,saveDir=None,Eng=False,compArea=False):
	time_slot=window/fs
	window_func=np.hamming(npoint)
	a=[]
	minAmp=0

	maxAmp=0
	count=0
	f=np.fft.rfftfreq(npoint,1/fs).tolist()
	area=[]
	m=0
	n=0
	trainDir='/home/gmuadmin/Desktop/Research Experimets/code/trainImages-low/'
	testDir='/home/gmuadmin/Desktop/Research Experimets/code/testImages-low/'
	evalMultiDir='/home/gmuadmin/Desktop/Research Experimets/code/evalImages-multi/'
	evalProfDir='/home/gmuadmin/Desktop/Research Experimets/code/evalImages-prof/'
	evalDingDir='/home/gmuadmin/Desktop/Research Experimets/code/evalImages-ding/'
	evalPanneerDir='/home/gmuadmin/Desktop/Research Experimets/code/evalImages-panneer/'
	evalLabDir='/home/gmuadmin/Desktop/Research Experimets/code/evalImages-lab2/'
	evalClsDir='/home/gmuadmin/Desktop/Research Experimets/code/evalImages-cls/'
	trainFiles=glob.glob('/home/gmuadmin/Desktop/Research Experimets/code/trainImages-low/*')
	testFiles=glob.glob('/home/gmuadmin/Desktop/Research Experimets/code/testImages-low/*')
	evalMultiFiles=glob.glob('/home/gmuadmin/Desktop/Research Experimets/code/evalImages-multi/*')
	evalProfFiles=glob.glob('/home/gmuadmin/Desktop/Research Experimets/code/evalImages-prof/*')
	evalDingFiles=glob.glob('/home/gmuadmin/Desktop/Research Experimets/code/evalImages-ding/*')
	evalPanneerFiles=glob.glob('/home/gmuadmin/Desktop/Research Experimets/code/evalImages-panneer/*')
	evalLabFiles=glob.glob('/home/gmuadmin/Desktop/Research Experimets/code/evalImages-lab2/*')
	evalClsFiles=glob.glob('/home/gmuadmin/Desktop/Research Experimets/code/evalImages-cls/*')
	trainOut=open('trainData','a')
	testOut=open('testData','a')
	evalMultiOut=open('evalMultiData','a')
	evalProfOut=open('evalProfData','a')
	evalDingOut=open('evalDingData','a')
	evalPanneerOut=open('evalPanneerData','a')
	evalLabOut=open('evalLabData','a')
	evalClsOut=open('evalClsData','a')
	for freq in f:
		if freq < 10:
			m+=1
		elif freq <= 1000:
			n+=1
		else:
			break
	for inData in input:
		file=pd.DataFrame(inData,columns=['val'])
		filt1=apply_filter(b1,a1,file['val'].values)
		filt2=apply_filter(b2,a2,filt1)
		data=np.concatenate((np.zeros(npoint-padding),filt2,np.zeros(npoint-padding)))
		for i in range(0,data.shape[0],window):
			if i+(npoint-padding) > data.shape[0]:
				break
			if padding is None:
				if Eng:
					x=abs(np.fft.rfft(data[i:(i+(npoint))]).real).tolist()

				elif logScale:
					with warnings.catch_warnings():
						warnings.simplefilter("ignore")
						x=abs(np.fft.rfft(data[i:(i+(npoint))]).real)
						x=np.where(x>0,np.log(x),0).tolist()
				else:
					x=abs(np.fft.rfft(data[i:(i+(npoint))]).real).tolist()
			else:
				if Eng:
					x=abs(np.fft.rfft(np.concatenate((data[i:(i+(npoint-padding))],np.zeros(padding)))).real).tolist()
				elif logScale:
					with warnings.catch_warnings():
						warnings.simplefilter("ignore")
						x=abs(np.fft.rfft(np.concatenate((data[i:(i+(npoint-padding))],np.zeros(padding)))).real)
						x=np.where(x>0,np.log(x),0).tolist()
				else:
					x=abs(np.fft.rfft(np.concatenate((data[i:(i+(npoint-padding))],np.zeros(padding)))).real).tolist()
			if maxAmp<max(x):
				maxAmp=max(x)
			if minAmp>min(x):
				minAmp=min(x[0:n])
			if count > (start * time_slot):
				if all(v==0 for v in x):
					continue
				if Eng:
					x=x[:n]
					#y=[x[i]*(f[i]/(len(x)-i+1)) for i in range(len(x))]
					y=[x[i]*f[i] for i in range(len(x))]
					a+=[np.mean(y)+np.std(y)]
				elif saveFile:
					a.append(x[:n])
				elif plotShort:
					a.append(x[:n])
				else:
					a.append(x)
			if end != None and count > (end*time_slot):
				break
			count+=1
		if compArea:
			area.append(trapz(a,dx=time_slot))
			a=[]


	if not compArea and saveFile and not Eng:
		f=f[:n]
		t=np.linspace(start,start+time_slot*len(a),len(a)).tolist()
		x,y=np.meshgrid(t,f)
		amplitude=np.array(a)
		ax=plt.gca()
		ax.set_axis_off()
		plt.margins(0,0)
		ax.xaxis.set_major_locator(NullLocator())
		ax.yaxis.set_major_locator(NullLocator())
		plt.ioff()
		plt.axis('off')
		plt.pcolormesh(x,y,np.transpose(amplitude),cmap='gist_heat')
		#plt.colorbar()
		if title is None:
			plt.savefig(saveDir+'/'+input+'.jpeg',bbox_inches='tight',pad_inches=0)
		else:
			plt.savefig(saveDir+'/'+title+'.jpeg',bbox_inches='tight',pad_inches=0)
	elif not compArea and not saveFile and not Eng:
		f=f[:n]
		t=np.linspace(start,start+time_slot*len(a),len(a)).tolist()
		x,y=np.meshgrid(t,f)
		amplitude=np.array(a)
		ax=plt.gca()
		ax.set_axis_off()
		plt.margins(0,0)
		ax.xaxis.set_major_locator(NullLocator())
		ax.yaxis.set_major_locator(NullLocator())
		plt.ioff()
		plt.axis('off')
		plt.pcolormesh(x,y,np.transpose(amplitude),cmap='brg')
		plt.show()
		#plt.colorbar()
	elif saveFile and not compArea and Eng:
		ax=plt.gca()
		ax.set_axis_off()
		plt.margins(0,0)
		ax.xaxis.set_major_locator(NullLocator())
		ax.yaxis.set_major_locator(NullLocator())
		plt.ioff()
		plt.axis('off')
		y=smooth(a)
		t=np.linspace(start,start+time_slot*len(y),len(y)).tolist()
		f=open(title+'.dat','w')
		for i in range(len(t)):
			f.write(str(t[i])+','+str(y[i])+'\n')
			
		'''
		plt.plot(t,y)
		if title is None:
			plt.savefig(saveDir+'/'+input+'.png',bbox_inches='tight',pad_inches=0)
		else:
			plt.savefig(saveDir+'/'+title+'.png',bbox_inches='tight',pad_inches=0)
		'''
		return
	elif not saveFile and not compArea and Eng:
		ax=plt.gca()
	#	ax.set_axis_off()
	#	plt.margins(0,0)
	#	ax.xaxis.set_major_locator(NullLocator())
	#	ax.yaxis.set_major_locator(NullLocator())
	#	plt.ioff()
	#	plt.axis('off')
		y=smooth(a)
		t=np.linspace(start,start+time_slot*len(y),len(y)).tolist()
		plt.plot(t,y)
		plt.show()
	elif Eng and saveFile and compArea:
		area=[str(are) for are in area]
		if trainDir+title+'.jpeg' in trainFiles:
			trainOut.write(title+','+','.join(area)+'\n')
		elif testDir+title+'.jpeg' in testFiles:
			testOut.write(title+','+','.join(area)+'\n')
		elif evalMultiDir+title+'.jpeg' in evalMultiFiles:
			evalMultiOut.write(title+','+','.join(area)+'\n')
		elif evalProfDir+title+'.jpeg' in evalProfFiles:
			evalProfOut.write(title+','+','.join(area)+'\n')
		elif evalDingDir+title+'.jpeg' in evalDingFiles:
			evalDingOut.write(title+','+','.join(area)+'\n')
		elif evalPanneerDir+title+'.jpeg' in evalPanneerFiles:
			evalPanneerOut.write(title+','+','.join(area)+'\n')
		elif evalLabDir+title+'.jpeg' in evalLabFiles:
			evalLabOut.write(title+','+','.join(area)+'\n')
		elif evalClsDir+title+'.jpeg' in evalClsFiles:
			evalClsOut.write(title+','+','.join(area)+'\n')
		else:
			print('File::'+title+' not found\n')
	return	

def build_spectrogram_complex(input,npoint,window,start,end,fs,padding=None,inFile=True,saveFile=False,title=None):
	if inFile==True:
		file=pd.read_csv(input,names=['val'])
	else:
		file=pd.DataFrame(input,columns=['val'])
#	file=file[35381700:59861700]
	real_spect=[]
	im_spect=[]
	j=0
	minAmp=0
	maxAmp=0
	count=0
	time_slot=window/fs
	window_func=np.hamming(npoint)
	#window_func=flattop(npoint)
	f=np.fft.fftfreq(npoint,1/fs).tolist()
	#f=np.linspace(-500,500,400)
	m=0
	n=0
	for freq in f:
                if freq < 10:
                        m+=1
                elif freq <= 500:
                        n+=1
                else:
                        break
	data=np.concatenate((np.zeros(npoint),file['val'].values,np.zeros(npoint)))
	for i in range(0,data.shape[0],window):
		if i+npoint > data.shape[0]:
			break
		if padding is None:
			x=np.fft.fft(data[i:(i+(npoint))])
		else:
			x=np.fft.fft(np.concatenate((data[i:(i+(npoint-padding))],np.zeros(padding))))

		if maxAmp<max(x):
			maxAmp=max(x)
		if minAmp>min(x):
			minAmp=min(x[0:n])
		if count > (start * time_slot):
			if saveFile:
				a.append(x[:n])
			else:
				real_spect.append(abs(x.real))
				im_spect.append(abs(x.imag))
		if end != None and count > (end*time_slot):
			break
		count+=1
	t=np.linspace(start,start+time_slot*len(real_spect),len(real_spect)).tolist()
	if saveFile:
		f=f[:n]
	x,y=np.meshgrid(t,f)
	real_amplitude=np.array(real_spect)
	img_amplitude=np.array(im_spect)
	if saveFile:
		ax=plt.gca()
		ax.set_axis_off()
		plt.margins(0,0)
		ax.xaxis.set_major_locator(NullLocator())
		ax.yaxis.set_major_locator(NullLocator())
		plt.ioff()
		plt.axis('off')
		plt.pcolormesh(x,y,np.transpose(real_amplitude),cmap='hot')
		if title is None:
			plt.savefig('./panneeer-spect-images/'+input+'.jpeg',bbox_inches='tight',pad_inches=0)
		else:
			plt.savefig('./panneer-spect-images/'+title+'.jpeg',bbox_inches='tight',pad_inches=0)
	else:
		if title is not None:
			plt.title(title)
		plt.xlabel('Time')
		plt.ylabel('Frequency')
	#	plt.pcolormesh(x,y,np.transpose(real_amplitude),cmap='hot')
		plt.pcolormesh(x,y,np.transpose(img_amplitude),cmap='hot')
		plt.colorbar()
		plt.show()


def buildDominantFreq(input,window,start,end,fs,inFile=True):
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
			a.append(x.index(max(x)))
		if end != None and count > (end*time_slot):
			break
		count+=1
	t=np.linspace(start,start+time_slot*len(a),len(a)).tolist()
	f=np.fft.rfftfreq(window,1/fs).tolist()
	#t=np.linspace(0,time_slot*file.shape[0],len(a)).tolist()
	#ax.set_yscale('log')
	output=open(input+'-spectD','w')
	for i in range(len(t)):
		output.write(str(t[i])+','+str(f[a[i]])+'\n')
			

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

def build_filter(Order,pass_band,stop_band,band,fs,filter,ripple,attenuation,plot=False):
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
		if plot==True:
			plt.plot((nyq/np.pi)*w,abs(h))
			plt.title(filter+' filter frequency response')
			plt.xlabel('Frequency')
			plt.ylabel('Amplitude')
			plt.grid(True)
			plt.legend(loc='best')
			plt.show()	
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

def plotData(dataFile,filtered,fs1,fs2,seg1,seg2,file=True):
	time_slot1=1/fs1
	time_slot2=1/fs2
	if file==True:
		data=pd.read_csv(dataFile,names=['val']).round(9)
		data=data['val'].iloc[:].values
	else:
		data=dataFile
	#t1=np.linspace(0,time_slot1*len(data),len(data))
	t1=range(1,1+dataFile.shape[0])
	plt.figure(1)
	plt.subplot(211)
	plt.title('time vs amplitude plot')
	label1=str(seg1)+'_'+str(seg2)
	plt.plot(t1,data,label=label1)
	plt.legend(loc='best')		
	plt.grid(True)
	t2=np.linspace(0,time_slot2*len(filtered),len(filtered))
	plt.subplot(212)
	plt.plot(t2,filtered,label='uncut')
	plt.legend(loc='best')		
	plt.grid(True)
	plt.show()

def checkSegmentation(inFile):
	data=pd.read_csv(inFile,names=['name','starttime','endtime','start','end','len','signal-qual','label'])
	for i in range(0,data.shape[0]):
		file1=pd.read_csv(data.iloc[i]['name'],names=['time','val'])
		file2=file1['val'].iloc[data.iloc[i]['start']:data.iloc[i]['end']]
		file1=file1['val']
		print(data.iloc[i]['name']+','+str(i))
		plotData(file1,file2,5000,5000,data.iloc[i]['start'],data.iloc[i]['end'],False)

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
	files=glob.glob('*_*')
	#mp.set_start_method('fork')
	b1,a1=build_filter(6,high,None,'lowpass',100000.0,'ellip',.01,40)
	b2,a2=build_filter(6,low,None,'highpass',5000.0,'ellip',.01,80)	
	for file in files:
		if '.py' not in file and 'Hz' not in file and  os.path.isdir(file) == False:
			p=mp.Process(target=getFiltered,args=(file,b1,a1,b2,a2,low,high,1,))
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

def buildSpectrogramFiles(blob,npoint,window,fs,padding=None,time=None,inputDir=None,saveFile=False,title=None,plotShort=False,logScale=False,saveDir=None,noSep=False,Seg=False,chop=False):
	if inputDir is None:
		files=glob.glob(blob)
	else:
		files=glob.glob(inputDir+'/'+blob)
	b1,a1=build_filter(6,1000,None,'lowpass',8000,'ellip',.01,40)
	b2,a2=build_filter(6,10,None,'highpass',8000.0,'ellip',.01,80)
	processes=[]
	count=0
	file_count=0
	err=open('file-not-processed','a')
	#mp.set_start_method('fork')
	for file in files:
		if '.py' not in file and 'spec' not in file and os.path.isdir(file) == False:
#			print(file)
			if title is None:
				title=file.strip().split('/')[-1]
			if noSep:
				f=open(file,'r')
				f=f.readlines()
				if len(f) > 1:
					f=pd.read_csv(file,names=['val'])
				else:
					try:
						f=f[0]
					except:
						continue
					flist=[]
					for i in range(0,len(f),8):
						try:
							flist.append(float(''.join(f[i:i+8])))
						except:
							#print(''.join(f[i:i+8]))
							print('came here')
							continue
					f=pd.DataFrame(data=flist,columns=['val'])
			'''
			if f.shape[0] < 60000:
				err.write(file)			

			else:
				add_size=f.shape[0]-60000
				parts=int(add_size/3)
				start=int(parts*2.5)
				end=start+60000
				f=f.iloc[start:end]
			'''
			if f['val'].dtype != np.float16 and f['val'].dtype != np.float32 and f['val'].dtype != np.float64:
				try:
					f=pd.to_numeric(f['val'].values,errors='raise')
					f=pd.DataFrame(data=f,columns=['val'])
				except:
					print("failed:"+file+"::"+str(count))
					count+=1
					continue
			try:
				filt1=apply_filter(b1,a1,f['val'].values)
			except:
				print("failed:"+file+"::"+str(count))
				count+=1
				continue
			#filt1=np.concatenate((filt1,filt1[8000:12000],filt1[8000:12000]))
			'''
			plt.plot(filt1.tolist())
			plt.show()
			'''
			filt2=apply_filter(b2,a2,filt1)
			'''
			plt.plot(filt2.tolist())
			plt.show()
			'''
			file_count+=1	
		#	while os.path.isfile('../'+saveDir+'/'+title+str(file_count)+'.jpeg'):
		#		file_count+=1
			
			#filt2=apply_filter(b2,a2,f['val'].values)
			if chop:
				for i in range(0,filt2.shape[0],8000):
					if (i+8000 > filt2.shape[0]):
						break
					indata=filt2[i:i+8000]		
					p=mp.Process(target=build_spectrogram,args=(indata,npoint,window,0,None,fs,padding,False,saveFile,title+str(file_count)+'-'+str(i),plotShort,logScale,saveDir,Seg,chop,))
					if count > 4:
						for k in processes:
							k.join()
						count=0
						processes=[]
					processes.append(p)
					count+=1
					p.start()
			else:
				indata=filt2		
				p=mp.Process(target=build_spectrogram,args=(indata,npoint,window,0,None,fs,padding,False,saveFile,title+str(file_count),plotShort,logScale,saveDir,Seg,chop,))
				if count > 4:
					for k in processes:
						k.join()
					count=0
					processes=[]
				processes.append(p)
				count+=1
				p.start()
	return

def createAudioFile(inFile):
	f=pd.read_csv(inFile,names=['val'])
	f_val=f['val'].values
	b1,a1=build_filter(6,1000,None,'lowpass',8000,'ellip',.01,40)
	b2,a2=build_filter(6,25,None,'highpass',8000.0,'ellip',.01,80)
	filt1=apply_filter(b1,a1,f['val'].values)
	filt2=apply_filter(b2,a2,filt1)
	scaled=np.int16(filt2/np.max(np.abs(filt2))*32767)
	write(inFile+'.wav',44100,scaled)
	
def buildSpectrogramDir(blob,npoint,window,fs,padding=None,time=None,inputDir=None,saveFile=False,title=None):
	if inputDir is None:
		files=glob.glob(blob)
	else:
		files=glob.glob(inputDir+'/'+blob)
	files=sorted(files,key=lambda x: (int(x.split('/')[-1].split('-')[0]),int(x.split('/')[-1].split('-')[1])))
	b1,a1=build_filter(6,1000,None,'lowpass',8000,'ellip',.01,40)
	b2,a2=build_filter(6,25,None,'highpass',8000.0,'ellip',.01,80)
	appended_value=None
	print(files)
	for file in files:
		f=pd.read_csv(file,names=['val'])
		if appended_value is None:
			appended_value=f['val'].values
		else:
			appended_value=np.concatenate((appended_value,f['val'].values))
#	print(appended_value.shape)
	filt1=apply_filter(b1,a1,appended_value)
	filt2=apply_filter(b2,a2,filt1)
	build_spectrogram(filt2,npoint,window,0,None,fs,padding,False,saveFile,title)

def buildSpectrogramDirMultpl(blob,npoint,window,fs,padding=None,time=None,inputDir=None,saveFile=False,title=None,plotShort=False,logScale=False,saveDir=None,Eng=False,compArea=False,shuffle=None):
	b1,a1=build_filter(6,1000,None,'lowpass',8000,'ellip',.01,40)
	b2,a2=build_filter(6,10,None,'highpass',8000.0,'ellip',.01,80)
	if inputDir is None:
		files=glob.glob(blob)
	else:
		files=glob.glob(inputDir+'/'+blob)
	file_samples={}
	for file in files:
		file_name=file.split('/')[-1]
		if len(file_name.split('-')) ==5:
			file_key=49
		else:
			file_key=int(file_name.split('-')[2])
		try:
			file_samples[file_key].append(file_name)
		except:
			file_samples[file_key]=[]
			file_samples[file_key].append(file_name)
	processes=[]
	count=0
	for key in file_samples.keys():
		if key!=16:
			continue
		file_names=file_samples[key]
		file_names=sorted(file_names,key=lambda x: (int(x.split('-')[0]),int(x.split('-')[1])))
		appended_value=[]
#		print(file_names)
		for file_name in file_names:
			flist=[]
			f=open(inputDir+'/'+file_name,'r')
			f=f.readlines()
			if len(f) > 1:
				f=pd.read_csv(inputDir+'/'+file_name,names=['val'])
			else:
				try:
					f=f[0]
				except:
					continue
				flist=[]
				for i in range(0,len(f),8):
					try:
						flist.append(float(''.join(f[i:i+8])))
					except:
						print(''.join(f[i:i+8]))
						continue
				f=pd.DataFrame(data=flist,columns=['val'])
			appended_value.append(f['val'].values)
		if shuffle:
			append_new=[]
			for i in shuffle:
				append_new.append(appended_value[i])
		else:
			append_new=appended_value
		p=mp.Process(target=build_spectrogram_seg,args=(append_new,npoint,window,0,None,fs,a1,b1,a2,b2,padding,False,saveFile,title+str(key),plotShort,logScale,saveDir,Eng,compArea))
		if count > 4:
			for k in processes:
				k.join()
			count=0
			processes=[]
		processes.append(p)
		count+=1
		p.start()
		break
	return

def computeEnergy(blob,npoint=None,window=None,fs=None,padding=None,time=None,inputDir=None,saveFile=False,title=None,plotShort=False,logScale=False,saveDir=None):
	b1,a1=build_filter(6,1000,None,'lowpass',8000,'ellip',.01,40)
	b2,a2=build_filter(6,10,None,'highpass',8000.0,'ellip',.01,80)
	if inputDir is None:
		files=glob.glob(blob)
	else:
		files=glob.glob(inputDir+'/'+blob)
	file_samples={}
	for file in files:
		file_name=file.split('/')[-1]
		if len(file_name.split('-')) ==5:
			file_key=49
		else:
			file_key=int(file_name.split('-')[2])
		try:
			file_samples[file_key].append(file_name)
		except:
			file_samples[file_key]=[]
			file_samples[file_key].append(file_name)
	processes=[]
	count=0
	output=open(title,'w')
	for key in file_samples.keys():
		file_names=file_samples[key]
		file_names=sorted(file_names,key=lambda x: (int(x.split('-')[0]),int(x.split('-')[1])))
		appended_value=None
#		print(file_names)
		for file_name in file_names:
			f=pd.read_csv(inputDir+'/'+file_name,names=['val'])
			if appended_value is None:
				appended_value=f['val'].values
			else:
				appended_value=np.concatenate((appended_value,f['val'].values))
#	print(appended_value.shape)
		filt1=apply_filter(b1,a1,appended_value)
		filt2=apply_filter(b2,a2,filt1)
		output.write(str(key)+','+str(np.sum(filt2**2))+'\n')
		

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
	files=glob.glob('*A_*')
	#mp.set_start_method('fork')
	b1,a1=build_filter(6,high,None,'lowpass',100000.0,'ellip',.01,40)
	b2,a2=build_filter(6,low,None,'highpass',5000.0,'ellip',.01,80)
	for file in files:
		if '.py' not in file and 'Hz' not in file and 'A' in file  and os.path.isdir(file) == False:
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

def getFeatures(inData,dataBase=None,specData=None,inFile=None,inLabel=None,labels=None):
	featureVector=[]
	'''
	for label in labels:
		p=mp.Process(target=getMeanDist,args=(database,inData,inFile,label,))
		p.start()
		p.join()
	for label in labels:
		p=mp.Process(target=getMeanSpectrogramDist,args=(database,specData,inFile,label,))
		p.start()
		p.join()
	'''
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
	return ','.join(featureVector)

def writeDistance(dataBase,data,data1,file1,start,end,):
	outFile=open('distances/seg5'+'_'+str(start)+'_'+str(end),'w')
	outputs=[]
	for i in range(start,end):
		data2=dataBase[data.iloc[i]['name']]
		data2=data2.tolist()
		label=data.iloc[i]['label']
		file2=data.iloc[i]['name']
		outputs.append(file1+','+file2+','+str(gf.getDTWDist(data1,data2))+','+label+'\n')
	for line in outputs:
		outFile.write(line)

def computeDistances(dataFile):
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
	for i in range(0,2):
		input1=dataBase[data.iloc[i]['name']]
		for j in range(i+1,data.shape[0],100):
			if j+100>data.shape[0]:
				end=data.shape[0]
			else:
				end=j+100
			p=mp.Process(target=writeDistance,args=(dataBase,data,input1.tolist(),data.iloc[i]['name'],j,end))
			if count>7:
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

def buildFeatures(dataFile,timeDistFile,timeSpecFile,central,timeVarFile=None,specVarFile=None,labels=None):
	data=pd.read_csv(dataFile,names=['name','starttime','endtime','start','end','len','signal-qual','label'])
#	mp.set_start_method('fork')
	#for i in range(data.shape[0]):
	f=open(timeDistFile,'r')
	lines=f.readlines()
	dtw_time={}
	for line in lines:
		tokens=line.strip().split(',')
		dtw_time[tokens[0]]=','.join(tokens[1:])
	f.close()
	f=open(timeSpecFile,'r')
	lines=f.readlines()
	dtw_freq={}
	for line in lines:
		tokens=line.strip().split(',')
		dtw_freq[tokens[0]]=','.join(tokens[1:])
	f.close()
	if timeVarFile!=None:
		f=open(timeVarFile,'r')
		var_time={}
		for line in lines:
			tokens=line.strip().split(',')
			var_time[tokens[0]]=','.join(tokens[1:])
		f.close()
	if specVarFile!=None:
		f=open(specVarFile,'r')
		var_spec={}
		for line in lines:
			tokens=line.strip().split(',')
			var_spec[tokens[0]]=','.join(tokens[1:])
		f.close()
	outFile=open(dataFile+'-features-'+central,'w')
	for i in range(data.shape[0]):
		label=data.iloc[i]['name'].split('_')[1]
		if label=='wake':
			label='wakeup'
		inData=pd.read_csv(data.iloc[i]['name'],names=('time','val'))
		inData=inData['val'].iloc[data.iloc[i]['start']:data.iloc[i]['end']]
		try:
			features=getFeatures(inData)
		except:
			print(data.iloc[i]['name'])
		if specVarFile != None and timeVarFile !=None:
			outFile.write(data.iloc[i]['name']+','+dtw_time[data.iloc[i]['name']]+','+dtw_freq[data.iloc[i]['name']]+','+var_time[data.iloc[i]['name']]+','+var_spec[data.iloc[i]['name']]+','+features+','+label+'\n')
		else:
			outFile.write(data.iloc[i]['name']+','+dtw_time[data.iloc[i]['name']]+','+dtw_freq[data.iloc[i]['name']]+','+features+','+label+'\n')
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

def makePredictions(dataDir,trainFile=None,testFile=None,time=None,train=None,cv=None,test=None,labels=None,depths=None,onlyTest=False):
	if onlyTest:
		bestdtc=pickle.load(open('dtc-nomlt-model','rb'))			
		testData=pd.read_csv(testFile,names=['file','p-1','p-2','p-3','p-4','p-5','p-6','label'])
		y_pred=bestdtc.predict(testData.iloc[:,1:7])
		print(str(confusion_matrix(testData.iloc[:,7:8],y_pred,labels=[0,1])))
		print(str(accuracy_score(testData.iloc[:,7:8],y_pred)))
		return
	dataFiles=glob.glob(dataDir+'*')
	trainData=None
	cvData=None
	testData=None
	'''
	for dataFile in dataFiles:
		data=pd.read_csv(dataFile,names=['file','p-1','p-2','p-3','p-4','p-5','p-6'])
		if 'random' in dataFile or 'none' in dataFile:
			data=data.assign(label=np.zeros(data.shape[0]))
		else:
			data=data.assign(label=np.zeros(data.shape[0])+1)
		if trainData is None:
			trainData=data[:35]
			cvData=data[35:40]
			testData=data[40:]
		else:
			trainData=pd.concat([trainData,data[:35]])
			cvData=pd.concat([cvData,data[35:40]])
			testData=pd.concat([testData,data[40:]])
	parameters={'max_depth':range(5,9)}
	clf=GridSearchCV(DecisionTreeClassifier(),parameters, n_jobs=4,cv=10,verbose=2)
	trainData=pd.read_csv(trainFile,names=['file','p-1','p-2','p-3','p-4','p-5','p-6','label'])
	cvData=pd.read_csv(testFile,names=['file','p-1','p-2','p-3','p-4','p-5','p-6','label'])
	clf.fit(trainData.iloc[:,1:7].values,trainData.iloc[:,7:8].values)
	bestdtc=clf.best_estimator_
	pickle.dump(bestdtc,open('dtc-nomlt-model','wb'))			
	return	
	best=0
	bestdtc=None
	for depth in depths:
		for k in range(1000):
			#dtc=buildTree(trainData.iloc[:,1:41],trainData.iloc[:,41:42],depth)
			dtc=buildTree(trainData.iloc[:,1:7],trainData.iloc[:,7:8],depth)
			correct_predictions=0
			for j in range(cvData.shape[0]):
				#predicted=dtc.predict(testData.iloc[j,1:41].values.reshape(1,-1))
				predicted=dtc.predict(cvData.iloc[j,1:7].values.reshape(1,-1))
				#actualed=testData.iloc[j,41:42].values[0]
				actualed=cvData.iloc[j,7:8].values[0]
				if predicted[0]==actualed:
					correct_predictions+=1
			if correct_predictions>best:
				best=correct_predictions
				bestdtc=dtc
	pickle.dump(bestdtc,open('dtc-nomlt-model','wb'))			
	'''
	bestdtc=pickle.load(open('dtc-nomlt-model','rb'))			
	testData=pd.read_csv(testFile,names=['file','p-1','p-2','p-3','p-4','p-5','p-6','label'])
	#dtc=buildTree(trainData.iloc[:,[m for m in range(1,14)]+[n for n in range (27,41)]],trainData.iloc[:,41:42],depth)
	#printTree(dtc,['h1','h2','h3','5','m','mean','area','abs_mean','abs_area','entropy','skew','kur','iqr','fft1','fft2','fft3','fft4','fft5','energy'])
	correct={'30l':0,'30r':0,'60l':0,'strt':0,'rand1':0,'rand3':0,'randw':0,'none':0}
	wrong={'30l':0,'30r':0,'60l':0,'strt':0,'rand1':0,'rand3':0,'randw':0,'none':0}
	predict={0:0,1:0}
	correct_predicts=0
	wrong_predicts=0
	outFile=open('output','a')
	for i in range(testData.shape[0]):
		predicted=bestdtc.predict(testData.iloc[i,1:7].values.reshape(1,-1))
		predict[predicted[0]]+=1
		actualed=testData.iloc[i,7:8].values[0]
		if predicted[0]==actualed:
			correct_predicts+=1
		else:
			wrong_predicts+=1
	print('correct predictions ='+str(correct_predicts)+', Wrong='+str(wrong_predicts)+'\n')
	'''
	outFile.write('\n')
	outFile.write('For time::'+str(time)+'\n')
	outFile.write('correct predictions ='+str(correct_predicts)+', Wrong='+str(wrong_predicts)+'\n')
	outFile.write(str(correct))
	outFile.write('\n')
	outFile.write(str(wrong))
	for key in actual.keys():
		print(key+','+str(predict[key])+','+str(actual[key])+','+str(correct_gesture[key]))
	for key in correct.keys():
		print(key+','+str(correct[key]))		
	'''
def makePredictionsRandom(dataFile,train=None,test=None,labels=None,trainFile=None,testFile=None,estimators=None,features=None,onlyTest=False):
	if onlyTest:
		bestdtc=pickle.load(open('rfrst-model','rb'))			
		testData=pd.read_csv(testFile,names=['file','p-1','p-2','p-3','p-4','p-5','p-6','label'])
		y_pred=bestdtc.predict(testData.iloc[:,1:7])
		print(str(confusion_matrix(testData.iloc[:,7:8],y_pred,labels=[0,1])))
		print(str(accuracy_score(testData.iloc[:,7:8],y_pred)))
		return
	'''
	data=pd.read_csv(dataFile,header=None)
	trainData=data.iloc[:train]
	cvData=data.iloc[cv:test]
	testData=data.iloc[test:]
	best=0
	bestclf=None
	#print(data.shape,trainData.shape,testData.shape)
	for estimator in estimators:
		for feature in features:
			for k in range(5):
				clf=RandomForestClassifier(n_estimators=estimator,max_depth=None,min_samples_split=2,n_jobs=4,max_features=feature)
				#clf.fit(trainData.iloc[:,1:41],trainData.iloc[:,41:42].values.ravel())
				clf.fit(trainData.iloc[:,1:37],trainData.iloc[:,37:38].values.ravel())
				correct_predictions=0
				for j in range(testData.shape[0]):
					#predicted=clf.predict(testData.iloc[j,1:41].values.reshape(1,-1))
					predicted=clf.predict(testData.iloc[j,1:37].values.reshape(1,-1))
					#actualed=testData.iloc[j,41:42].values[0]
					actualed=testData.iloc[j,37:38].values[0]
					if predicted[0]==actualed:
						correct_predictions+=1
				if correct_predictions>best:
					best=correct_predictions
					bestclf=clf
	param_grid={'max_depth':[6,7,8,9,10],'max_features':[1,2],'min_samples_leaf':[2,3,4],'min_samples_split':[8,10,12],'n_estimators':[20,30,40]}
	clf=GridSearchCV(RandomForestClassifier(),param_grid, n_jobs=4,cv=10,verbose=2)
	trainData=pd.read_csv(trainFile,names=['file','p-1','p-2','p-3','p-4','p-5','p-6','label'])
	clf.fit(trainData.iloc[:,1:7].values,trainData.iloc[:,7:8].values)
	bestdtc=clf.best_estimator_
	pickle.dump(bestdtc,open('rfrst-model','wb'))		
	return	
	'''
	bestdtc=pickle.load(open('rfrst-model','rb'))			
	testData=pd.read_csv(testFile,names=['file','p-1','p-2','p-3','p-4','p-5','p-6','label'])
	predict={0:0,1:0}
	correct_predicts=0
	wrong_predicts=0
	outFile=open('output','a')
	for i in range(testData.shape[0]):
		predicted=bestdtc.predict(testData.iloc[i,1:7].values.reshape(1,-1))
		predict[predicted[0]]+=1
		actualed=testData.iloc[i,7:8].values[0]
		if predicted[0]==actualed:
			correct_predicts+=1
		else:
			wrong_predicts+=1
	print('correct predictions ='+str(correct_predicts)+', Wrong='+str(wrong_predicts)+'\n')
	'''
	for i in range(testData.shape[0]):
		name=testData.iloc[i,0].split('_')[0]
		#predicted=bestclf.predict(testData.iloc[i,1:41].values.reshape(1,-1))
		predicted=bestclf.predict(testData.iloc[i,1:37].values.reshape(1,-1))
		#predicted=clf.predict(testData.iloc[i,[m for m in range(1,14)]+[n for n in range(27,41)]].values.reshape(1,-1))
		predict[predicted[0]]+=1
		#actualed=testData.iloc[i,41:42].values[0]
		actualed=testData.iloc[i,37:38].values[0]
		#actual[testData.iloc[i,41:42].values[0]]+=1
		actual[testData.iloc[i,37:38].values[0]]+=1
		if predicted[0]==actualed:
			correct_predicts+=1
			correct_gesture[predicted[0]]+=1
			correct[name]+=1
		conf_matx[actualed][predicted[0]]+=1
	outFile.write("correct ="+str(correct_predicts)+','+user+','+'rand-'+str(time.time())+'\n')
	outFile.write(str(conf_matx))
	for key in conf_matx.keys():
		to_write=[]
		to_write.append(key)
		item=conf_matx[key]
		for label in labels:
			to_write.append(str(item[label]))
		outFile.write(','.join(to_write)+'\n')
	for key in actual.keys():
		print(key+','+str(predict[key])+','+str(actual[key])+','+str(correct_gesture[key]))
	for key in correct.keys():
		print(key+','+str(correct[key]))		
	'''

def makePredictionsSVM(dataFile,train,test,onlyTest=False,labels=None):
	if onlyTest:
		clf=pickle.load(open('svm-model','rb'))			
		trainData=pd.read_csv(train,names=['file','p-1','p-2','p-3','p-4','p-5','p-6','label'])
		testData=pd.read_csv(test,names=['file','p-1','p-2','p-3','p-4','p-5','p-6','label'])
		scaler=StandardScaler()
		scaler.fit(trainData.iloc[:,1:7])
		train=scaler.transform(trainData.iloc[:,1:7])
		test=scaler.transform(testData.iloc[:,1:7])
		y_pred=clf.predict(test)
		print(str(confusion_matrix(testData.iloc[:,7:8],y_pred,labels=[0,1])))
		print(str(accuracy_score(testData.iloc[:,7:8],y_pred)))
		return
	#output=open('output','a')
	svm = SVC()
	parameters = {'kernel':('linear', 'rbf'), 'C':(1,0.25,0.5,0.75),'gamma': (1,2,3,'auto'),'decision_function_shape':('ovo','ovr'),'shrinking':(True,False)}
#	data=pd.read_csv(dataFile,header=None)
	trainData=pd.read_csv(train,names=['file','p-1','p-2','p-3','p-4','p-5','p-6','label'])
	testData=pd.read_csv(test,names=['file','p-1','p-2','p-3','p-4','p-5','p-6','label'])
	scaler=StandardScaler()
	scaler.fit(trainData.iloc[:,1:7])
	train=scaler.transform(trainData.iloc[:,1:7])
	test=scaler.transform(testData.iloc[:,1:7])
	clf=GridSearchCV(svm, parameters,cv=10)
	clf=clf.fit(train,trainData.iloc[:,7:8].values.ravel())
	pickle.dump(clf,open('svm-model','wb'))			
	return
	clf=pickle.load(open('svm-model','rb'))			
	y_pred=clf.predict(test)
	#output=open('output','a')
	#output.write(dataFile+'-svm-Predictions:\n')
	print(str(confusion_matrix(testData.iloc[:,7:8],y_pred,labels=[0,1])))	

def makePredictionsPCA(modelType,trainDir,testDir,modelFile=None,pcaFile=None,onlyTest=False):
	if onlyTest:
		model=pickle.load(open(modelFile,'rb'))
		pca=pickle.load(open(pcaFile,'rb'))
		testData,testLabels,pca=computePCAFeatures(testDir,pca)
		if modelType=='DT':
			print(model.score(testData,testLabels))
		if modelType=='RDF':
			print(model.score(testData,testLabels))
		if modelType=='SVM':
			print(model.score(testData,testLabels))
		return		
	if pcaFile is None:	
		trainData,trainLabels,pca=computePCAFeatures(trainDir)
		pickle.dump(pca,open('pca-param','wb'))
	else:
		pca=pickle.load(open(pcaFile,'rb'))
		trainData,trainLabels,pca=computePCAFeatures(trainDir,pca)
	testData,testLabels,pca=computePCAFeatures(testDir,pca)
	if modelType=='DT':
		parameters={'max_depth':range(2,30)}
		clf=GridSearchCV(DecisionTreeClassifier(),parameters, n_jobs=4,cv=10,verbose=2)
		clf.fit(trainData,trainLabels)
		bestdtc=clf.best_estimator_
		pickle.dump(bestdtc,open('dtc-pca-model','wb'))
		pickle.dump(pca,open('pca-param','wb'))
	if modelType=='RDF':
		param_grid={'max_depth':range(2,10),'max_features':[1,2,3,4],'min_samples_leaf':[2,3,4],'min_samples_split':[8,10,12],'n_estimators':range(10,100,10)}
		clf=GridSearchCV(RandomForestClassifier(),param_grid, n_jobs=4,cv=10,verbose=2)
		clf.fit(trainData,trainLabels)
		bestdtc=clf.best_estimator_
		pickle.dump(bestdtc,open('RDF-pca-model','wb'))
	if modelType=='SVM':
		svm = SVC()
		parameters = {'kernel':('linear', 'rbf'), 'C':(1,0.25,0.5,0.75),'gamma': (1,2,3,'auto'),'decision_function_shape':('ovo','ovr'),'shrinking':(True,False)}
		clf=GridSearchCV(svm, parameters,cv=10)
		clf=clf.fit(trainData,trainLabels)
		pickle.dump(clf,open('SVM-pca-model','wb'))			
	return

def createLabels(dataFile):
	f=open(dataFile,'r')
	outfile=open(dataFile+'_labeled','w')
	lines=f.readlines()
	for line in lines:
		tokens=line.strip().split(',')
		label=tokens[0].split('_')[1]
		outfile.write(line.strip()+','+label+'\n')

def computeLabelAverageDistances(labels):
        files=glob.glob('*-*')
        average_distances={}
        for file in files:
                if 'merged' in file:
                        f=open(file,'r')
                        lines=f.readlines()
                        for line in lines:
                                file1,file2,distance,label=line.strip().split(',')
                                distance=float(distance)
                                label1=file1.split('_')[1]
                                label2=file2.split('_')[1]
                                if label1=='wake':
                                        label1='wakeup'
                                if label2=='wake':
                                        label2='wakeup'
                                try:
                                        average_distances[file1][label2]+=distance
                                        average_distances[file1][label2+'_count']+=1
                                except:
                                        average_distances[file1]={}
                                        for label in labels:
                                                average_distances[file1][label]=0
                                                average_distances[file1][label+'_count']=0
                                        average_distances[file1][label2]+=distance
                                        average_distances[file1][label2+'_count']+=1
                                try:
                                        average_distances[file2][label1]+=distance
                                        average_distances[file2][label1+'_count']+=1
                                except:
                                        average_distances[file2]={}
                                        for label in labels:
                                                average_distances[file2][label]=0
                                                average_distances[file2][label+'_count']=0
                                        average_distances[file2][label1]+=distance
                                        average_distances[file2][label1+'_count']+=1
        outfile=open('dtw_time_prof_seg-2','w')
        for key in average_distances.keys():
                towrite=[]
                towrite.append(key)
                for label in labels:
                        towrite.append(str(average_distances[key][label]/average_distances[key][label+'_count']))
                outfile.write(','.join(towrite)+'\n')
        return

def computeLabelMedianDistances(labels):
        files=glob.glob('*-*')
        average_distances={}
        for file in files:
                if 'merged' in file:
                        f=open(file,'r')
                        lines=f.readlines()
                        for line in lines:
                                file1,file2,distance,label=line.strip().split(',')
                                distance=float(distance)
                                label1=file1.split('_')[1]
                                label2=file2.split('_')[1]
                                if label1=='wake':
                                        label1='wakeup'
                                if label2=='wake':
                                        label2='wakeup'
                                try:
                                        average_distances[file1][label2].append(distance)
                                except:
                                        average_distances[file1]={}
                                        for label in labels:
                                                average_distances[file1][label]=[]
                                        average_distances[file1][label2].append(distance)
                                try:
                                        average_distances[file2][label1].append(distance)
                                except:
                                        average_distances[file2]={}
                                        for label in labels:
                                                average_distances[file2][label]=[]
                                        average_distances[file2][label1].append(distance)
        outfile=open('dtw_spec_prof_seg-2-med','w')
        for key in average_distances.keys():
                towrite=[]
                towrite.append(key)
                for label in labels:
                        towrite.append(str(np.median(average_distances[key][label])))
                outfile.write(','.join(towrite)+'\n')
        return

def computeLabelModeDistances(labels):
        files=glob.glob('*-*')
        average_distances={}
        for file in files:
                if 'merged' in file:
                        f=open(file,'r')
                        lines=f.readlines()
                        for line in lines:
                                file1,file2,distance,label=line.strip().split(',')
                                distance=round(float(distance))
                                label1=file1.split('_')[1]
                                label2=file2.split('_')[1]
                                if label1=='wake':
                                        label1='wakeup'
                                if label2=='wake':
                                        label2='wakeup'
                                try:
                                        average_distances[file1][label2].append(distance)
                                except:
                                        average_distances[file1]={}
                                        for label in labels:
                                                average_distances[file1][label]=[]
                                        average_distances[file1][label2].append(distance)
                                try:
                                        average_distances[file2][label1].append(distance)
                                except:
                                        average_distances[file2]={}
                                        for label in labels:
                                                average_distances[file2][label]=[]
                                        average_distances[file2][label1].append(distance)
        outfile=open('dtw_spec_prof_seg-2-mod','w')
        for key in average_distances.keys():
                towrite=[]
                towrite.append(key)
                for label in labels:
                        towrite.append(stats.mode(average_distances[key][label])[0][0]/stats.mode(average_distances[key][label])[1][0])
                towrite=[str(x) for x in towrite]
                outfile.write(','.join(towrite)+'\n')
        return

def computePCAFeatures(inputDir,pca=None):
	inFiles=glob.glob(inputDir+'*')
	inData=[]
	inLabel=[]
	for inFile in inFiles:
		im=Image.open(inFile)
		im=im.convert("L")
		im=np.asarray(im,dtype=np.float32)
		im=im.reshape(-1)
		if 'random' in inFile or 'none' in inFile or 'rand-mot' in inFile or 'rand-wlk' in inFile or 'prof-rand' in inFile or 'ding-rand' in inFile:
			inData.append(im)
			inLabel.append('0')
		else:
			inData.append(im)
			inLabel.append('1')
	inData_array=np.array(inData)
	if pca is 'test':
		pca=PCA()
		pca.fit(inData_array)
	elif pca is None:
		pca=PCA(n_components=75)
		pca.fit(inData_array)
	elif isinstance(pca,int):
		pca=PCA(n_components=pca)
		pca.fit(inData_array)
	inData_reduced=pca.transform(inData_array)	
	return pca,inData_reduced,inLabel


def buildSpectrogramAllClass(inDir,user,outDir):
	inFiles=glob.glob(inDir+'*')
	for inFile in inFiles:
		class_name=inFile.strip().split('/')[-1]
		if class_name!='heat':
			continue
		print(class_name)
		buildSpectrogramFiles(inputDir=inFile,blob='*',npoint=6400,window=8,fs=8000,padding=5600,saveFile=True,title=user+'-'+class_name,saveDir=outDir+'/'+class_name,plotShort=True,logScale=True,noSep=True,chop=True)

'''
Low pass at 200 Hz order 6 rp=.01, rs=40
High pass at 10 Hz 6 rp=.01 and rs =80
'''

buildSpectrogramFiles(inputDir='/home/gmuadmin/Desktop/Research Experimets/Raw-data/ding/snow',blob='gest-108996-text',npoint=6400,window=8,fs=8000,padding=5600,saveFile=False,title='12-straight',saveDir='./temp',plotShort=True,logScale=True,noSep=True,Seg=True,chop=False)
#buildSpectrogramFiles(inputDir='./',blob='temp3-mavr',npoint=6400,window=16,fs=8000,padding=5600,saveFile=False,title='12-straight',saveDir='./temp',plotShort=True,logScale=True,noSep=True,Seg=False,chop=False)
#buildSpectrogramAllClass('/home/gmuadmin/Desktop/Research Experimets/Raw-data/prof/','prf','/home/gmuadmin/Desktop/Research Experimets/Split-Spectrograms/prof/')

#createAudioFile('/home/gmuadmin/Desktop/Research Experimets/8-9-text/rain-2-1551-text-maver')
#buildSpectrogramDirMultpl(inputDir='/home/gmuadmin/Desktop/Research Experimets/wake-word-data/10-4-text/angle/30l/sample1',blob='*',npoint=6400,window=8,fs=8000,padding=5600,saveFile=True,title='energy-int',saveDir='./',logScale=True,plotShort=True,Eng=True,compArea=False,shuffle=None)
#buildSpectrogramDirMultpl(inputDir='/home/gmuadmin/Desktop/Research Experimets/wake-word-data/10-4-text/walking-behind/sample-2',blob='*',npoint=6400,window=8,fs=8000,padding=5600,saveFile=False,title='energy-30ls217-',saveDir='./',logScale=True,plotShort=True,Eng=True,compArea=False,shuffle=None)
#computeEnergy(inputDir='/home/gmuadmin/Desktop/Research Experimets/9-22-exp/4/random-1',blob='*',title='random-1')
#f=pd.read_csv('/home/gmuadmin/Desktop/Research Experimets/8-15-panneer/rain-5467-text',names=['val'])
#f['val']=f['val'].str.replace('i','j')
#f['val']=f['val'].str.replace('+','-')
#f['val']=f['val'].str.replace(' ','').apply(lambda x: np.complex(x))
#plt.plot(np.real(f['val'].values).tolist())
#plt.show()
#build_spectrogram_complex(np.real(f['val'].values),npoint=6400,start=0,end=None,window=8,fs=8000,padding=5600,saveFile=False,inFile=False)
#makePredictions('3area/',trainFile='trainData-label',testFile='evalProfData-label',time='2-seconds-rain',depths=[5,6,7,8],onlyTest=True)
#makePredictionsRandom('3area/',trainFile='trainData-label',testFile='evalProfData-label',onlyTest=True)
#makePredictionsSVM('2area-rain/',train='./Features-Non-CNN/trainData-label',test='./Features-Non-CNN/evalProfData-label',onlyTest=False)
#build_spectrogram(f['val'].values,6400000,800000,0,None,16000000,False,'obj-speed1-no-smooth')
#pca,inp,label=computePCAFeatures(inputDir='/home/gmuadmin/Desktop/Research Experimets/training-data/train-new/',pca=2000)
#makePredictionsPCA(modelType='DT',trainDir='train_low/trainImages-low/',testDir='train_low/evalImages-panneer-low/',modelFile='dtc-pca-model',pcaFile='pca-param',onlyTest=True)

