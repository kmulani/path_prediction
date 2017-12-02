from pykalman import KalmanFilter
import numpy as np
import matplotlib.pyplot as plt
import time
import copy

k = [] #first bin
overlap = [] #overlapping bin
i = 1
measurements = np.array([])

def kf_predict(meassurements):

	global i
	newMeasurement = np.ma.asarray(-1)

	initial_state_mean = [measurements[0, 0],0,measurements[0, 1],0]

	transition_matrix = [[1, 0, 0.4, 0],[1, 0, 0.1, 0],[0, 0.2, 1, 1],[0.2, 0.5, 0.3, 1]] #TWEAK THIS BASED ON EXPERIMENTATION

	observation_matrix = [[1, 0, 0, 0],[0, 0, 1, 0]]
	
	kf1 = KalmanFilter(transition_matrices = transition_matrix,observation_matrices = observation_matrix,initial_state_mean = initial_state_mean)

	kf1 = kf1.em(measurements, n_iter=5)
	(smoothed_state_means, smoothed_state_covariances) = kf1.smooth(measurements)

	#PREDICTION
	x_now = smoothed_state_means[-1, :]
	P_now = smoothed_state_covariances[-1, :]
	
	newMeasurement = np.ma.asarray(measurements[measurements.shape[0]-1])

	#print "++++++++"
	#print x_now
	#print P_now
	#print "+++++++++"
	(x_now, P_now) = kf1.filter_update(filtered_state_mean = x_now,filtered_state_covariance = P_now, observation = newMeasurement) 

	#PLOTS - Comment out later
	plt.figure(i)
	i = i+1
	plt.hold(True)
	times = range(measurements.shape[0])
	plt.plot(times, measurements[:, 0], 'bo',times, measurements[:, 1], 'ro',times, smoothed_state_means[:, 0], 'b--',times, smoothed_state_means[:, 2], 'r--')
	print x_now[0]
	print x_now[1]
	#print np.array(times).shape[0] + 1
	plt.plot([np.array(times).shape[0] +1], x_now[0],'xb', [np.array(times).shape[0] +1], x_now[1],'xr')
	return (x_now)


def path_pred(data): #Give input as data from topic

	global k
	global overlap
	global measurements
	flag = 0

	#Shift in window 
	over_ele  = 10 		#TWEAK THIS BASED ON EXPERIMENTATION
	#Buffer Size
	buffer_size = 20 	#TWEAK THIS BASED ON EXPERIMENTATION

	k.append(data)
	measurements = np.array(k)
	#print measurements.size/2

	if( (measurements.size/2) > over_ele ): # Overlap by ( buffer_size - over_ele )
		overlap.append(data)	
	#print "xxxxxx"
	#print overlap

	if ( (measurements.size/2) == buffer_size ): #Input buffer size is given above
		x_now = kf_predict(measurements)
		flag = 1
		#kf_blind(measurements)
		
		print "-----"
		print measurements
		print "-----"
		k = list(overlap)
		print k	
		if(over_ele > 0):
			for i in range (0,(over_ele)) :
				overlap.pop(0)	
		print "------"
		print overlap
	
	if (flag == 1):
		return (x_now)
	else:
		return (data)


#measurements = np.asarray([(399,293),(403,299),(409,308),(416,315),(418,318),(420,323),(429,326),(423,328),(429,334),(431,337),(433,342),(434,352),(434,349),(433,350),(431,350),(430,349),(428,347),(427,345),(425,341),(429,338),(431,328),(410,313),(406,306),(402,299),(397,291),(391,294),(376,270),(372,272),(351,248),(336,244),(327,236),(307,220)])
data = [399,293]
data = path_pred(data)
data = [403,299]
data = path_pred(data)
data = [409,308]
data = path_pred(data)
data = [416,315]
data = path_pred(data)
data = [418,318]
data = path_pred(data)
data = [420,323]
data = path_pred(data)
data = [429,326]
data = path_pred(data)
data = [423,328]
data = path_pred(data)
data = [429,334]
data = path_pred(data)
data = [431,337]
data = path_pred(data)
data = [433,342]
data = path_pred(data)
data = [434,352]
data = path_pred(data)
data = [434,349]
data = path_pred(data)
data = [433,350]
data = path_pred(data)
data = [431,350]
data = path_pred(data)
data = [430,349]
data = path_pred(data)
data = [428,347]
data = path_pred(data)
data = [427,345]
data = path_pred(data)
data = [425,341]
data = path_pred(data)
data = [429,338]
data = path_pred(data)
data = [431,328]
data = path_pred(data)
data = [410,313]
data = path_pred(data)
data = [406,306]
data = path_pred(data)
data = [402,299]
data = path_pred(data)
data = [397,291]
data = path_pred(data)
data = [391,294]
data = path_pred(data)
data = [376,270]
data = path_pred(data)
data = [372,272]
data = path_pred(data)
data = [351,248]
data = path_pred(data)
data = [336,244]
data = path_pred(data)
data = [327,236]
data = path_pred(data)
data = [307,220]
data = path_pred(data) 
plt.show() #Comment out later
