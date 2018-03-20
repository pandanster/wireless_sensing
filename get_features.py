import numpy as np
import scipy as sp
import mlpy
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

def getMean(data):
	result = np.mean(data)
	return result

def getTotalMean(datax, datay, dataz):
	data = datax + datay + dataz
	result = np.mean(data)
	return result

def getArea(data):
	result = np.sum(data)
	return result 

def getPostureDist(datax, datay, dataz):
	diffxy = datax - datay
	diffxz = datax - dataz
	diffyz = datay - dataz
	return [diffxy, diffxz, diffyz]

def getAbsMean(data):
	result = np.mean(np.abs(data))
	return result

def getAbsArea(data):
	result = np.sum(np.abs(data))
	return result

def getTotalAbsArea(datax, datay, dataz):
	data = np.abs(datax) + np.abs(datay) + np.abs(dataz)
	result = np.sum(data)
	return result

def getTotalSVM(datax, datay, dataz):
	result = np.mean(np.sqrt(datax**2 + datay**2 + dataz**2))
	return result

#calculate the entropy of fft values
def getEntropy(data):
	data_fft = sp.fft(data)
	data_fft_abs = np.abs(data_fft)
	data_fft_abs_sum = np.sum(data_fft_abs)
	data_fft_abs_norm = data_fft_abs/data_fft_abs_sum
	data_fft_abs_norm_log2 = np.log2(data_fft_abs_norm)
	result = - np.sum(data_fft_abs_norm * data_fft_abs_norm_log2)
	result = result/len(data_fft)
	return result

def getSkew(data):
	result = sp.stats.skew(data)
	return result

def getKur(data):
	result = sp.stats.kurtosis(data)
	return result

def getQuartiles(data):
	q1 = np.percentile(np.abs(data), 25)
	q2 = np.percentile(np.abs(data), 50)
	q3 = np.percentile(np.abs(data), 75)
	return [q1, q2, q3]

#calculate the variance of data in each window
def getVar(data):
	result = np.var(data)
	return result

def getAbsCV(data):
	std = np.std(np.abs(data))
	mean = np.mean(np.abs(data))
	result = std / mean * 100.0
	return result

def getIQR(Q3, Q1):
	result = Q3 - Q1
	return result

def getRange(data):
	big = max(data)
	small = min(data)
	result = big - small
	return result

def getFFTCoeff(data):
	data_fft = abs(sp.fft(data))
	return data_fft[1:]

# get the first 5 largest fft coefficient
def getFFTPeaks(data):
	data_fft = sp.fft(data)
	result = np.sort(abs(data_fft))
	result = result[::-1]
	return result[1:6] 

def getEnergy(data):
	data_fft = sp.fft(data)
	data_fft_half = data_fft[1:int(len(data_fft)/2+1)]
	data_fft_half_abs = np.abs(data_fft_half)
	result = np.sum(data_fft_half_abs**2)
	result = result/len(data_fft_half)
	return result
	
# calculate the second peak of autocorrelation of fft values
def getPitch(data):
	data_fft = sp.fft(data)
	result = np.correlate(data_fft, data_fft, 'full')
	result = np.sort(np.abs(result))
	return result[len(result)-2]

def getDomFreqRatio(data):
	data_fft = sp.fft(data)
	data_fft_sort = list(np.sort(abs(data_fft[:(len(data_fft)/2+1)])))
	large = data_fft_sort.pop()
	ratio = large / np.sum(data_fft_sort)
	return ratio

def getMCR(data):
	mean = np.mean(data)
	k = 0
	for i in range(len(data)-1):
		if (data[i] - mean) * (data[i+1] - mean) < 0:
			k += 1
	result = k * 1.0 / (len(data) - 1)
	return result

# Pearson's correlation coefficients
def getCorr(datax, datay, dataz):
	result0 = np.corrcoef(datax, datay)
	result1 = np.corrcoef(datax, dataz)
	result2 = np.corrcoef(datay, dataz)
	return [result0[0][1], result1[0][1], result2[0][1]]


def getDTWDist(data1,data2):
	distance,path= fastdtw(data1,data2,dist=euclidean)
	return distance
