#!/usr/bin/python3
import pandas as pd
import numpy as np
import random
import argparse
import math
import util
import copy
from sting.data import parse_c45
#import matplotlib.pyplot as plt
#from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder


def all_index(data, num):
	# return all index of num in data(1D)
	length = len(data)
	index = []
	for i in range(length):
		if data[i] == num:
			index.append(i)
	return index


def P(r, x):
	res = x
	if x <-r:
		res = -r
	if x>r:
		res = r
	return res

# load data
def load_iris():
	df = pd.read_csv('440data/iris/iris.csv', header=None)
	X = df.values[:, :-1].astype(float)
	Y = df.values[:, -1]
	
	n_class = len(np.unique(Y))
	encoder = LabelEncoder()
	Y = encoder.fit_transform(Y)
	#
	return n_class, X, Y


def myKmeans(n_cluster, data, init=20):
	# n_cluster: number of clusters
	# init: number of iteration of initialization
	
	n_sample = data.shape[0]
	n_feature = data.shape[1]
	
	# initialization: choose randomly and pick the initial clustering that corresponds to the lowest tightness.
	Q = 0 # overall tightness
	centroid = []
	for num in range(init):
		partition_current = np.random.randint(n_cluster, size=n_sample)
		centroid_current = np.zeros((n_cluster, n_feature))
		q1 = np.zeros((n_cluster)) # corresponding within-cluster tightness

		# compute centroids
		for cluster in range(n_cluster):
			#centroid_current[cluster,:] = np.mean(data[partition_current==cluster,:],0)
			centroid_current[cluster, :] = np.mean(data[all_index(partition_current,cluster), :], 0)

		# compute within-cluster tightness
		for j in range(n_sample):
			cluster = partition_current[j]
			q1[cluster] += np.linalg.norm(data[j,:]-centroid_current[cluster,:])

		# compute currently overall tightness Q and choose smaller one
		Q_current = np.sum(q1)
		if Q_current<Q or num==0:
			Q = Q_current
			centroid = centroid_current
			partition = partition_current
	# # initialization finished

	# begin iteration
	tau = 1e-2 # tolerance
	diff = 2*tau # make sure diff>tau at first
	while(diff>tau):
		# find closest cluster for x(j), update partition
		for j in range(n_sample):
			temp = np.linalg.norm(data[j,:]-centroid[0,:]) # compare with cluster 0
			partition[j] = 0
			for cluster in range(1,n_cluster):
				temp_cluster = np.linalg.norm(data[j,:]-centroid[cluster,:])
				if temp_cluster<temp:
					temp = temp_cluster
					partition[j] = cluster # reassign data(j,:)
					
		# update centroid for each cluster and compute new q1 (within-cluster tightness)
		for cluster in range(n_cluster):
			#centroid[cluster,:] = np.mean(data[partition==cluster,:],0)
			centroid[cluster, :] = np.mean(data[all_index(partition, cluster), :], 0)
			for i in range(n_feature):
				if math.isnan(centroid[cluster, i])==True:
					centroid[cluster, i] = 0
		
		q1 = np.zeros((n_cluster))
		for j in range(n_sample):
			cluster = partition[j]
			q1[cluster] += np.linalg.norm(data[j, :] - centroid[cluster, :])
			
		# get new Q
		Q_new = np.sum(q1)
		diff = np.abs(Q_new - Q)
		Q = Q_new
	
	return partition


def myKmeans_DP(n_cluster, data, init=20, eps=1.0):
	# n_cluster: number of clusters
	# init: number of iteration of initialization
	
	n_sample = data.shape[0]
	n_feature = data.shape[1]
	rangeData = (np.max(data) - np.min(data))/2
	eps0 = eps/ ((n_feature*rangeData+1)*1)

	
	# initialization: choose randomly and pick the initial clustering that corresponds to the lowest tightness.
	Q = 0  # overall tightness
	centroid = []
	for num in range(init):
		partition_current = np.random.randint(n_cluster, size=n_sample)
		centroid_current = np.zeros((n_cluster, n_feature))
		q1 = np.zeros((n_cluster))  # corresponding within-cluster tightness
		
		# compute centroids
		for cluster in range(n_cluster):
			#centroid_current[cluster, :] = np.mean(data[partition_current == cluster, :], 0)
			centroid_current[cluster, :] = np.mean(data[all_index(partition_current, cluster), :], 0)
		
		# compute within-cluster tightness
		for j in range(n_sample):
			cluster = partition_current[j]
			q1[cluster] += np.linalg.norm(data[j, :] - centroid_current[cluster, :])
		
		# compute currently overall tightness Q and choose smaller one
		Q_current = np.sum(q1)
		if Q_current < Q or num == 1:
			Q = Q_current
			centroid = centroid_current
			partition = partition_current
	# initialization finished
	
	# begin iteration
	#for iter in range(n_iter):
	tau = 1e-2  # tolerance
	diff = 2 * tau  # make sure diff>tau at first
	maxiter = 10
	numiter = 0
	while (diff>tau and numiter<maxiter):
		# find closest cluster for x(j), update partition
		for j in range(n_sample):
			temp = np.linalg.norm(data[j, :] - centroid[0, :])  # start with cluster 0 and compare
			partition[j] = 0
			for cluster in range(1, n_cluster):
				temp_cluster = np.linalg.norm(data[j, :] - centroid[cluster, :])
				if temp_cluster < temp:
					temp = temp_cluster
					partition[j] = cluster  # reassign data(j,:)
				
		# update centroid for each cluster and compute new q1 (within-cluster tightness)
		for cluster in range(n_cluster):
			#centroid[cluster, :] = np.mean(data[partition == cluster, :], 0)
			centroid[cluster, :] = np.mean(data[all_index(partition, cluster), :], 0)
			# add noise to count
			count = partition.tolist().count(cluster) + np.random.laplace(0, 1 / eps0)
			for i in range(n_feature):
				# add noise to centroids
				sumi = np.sum(data[partition == cluster, i]) + np.random.laplace(0, 1 / eps0)
				centroid[cluster, i] = P(rangeData, sumi/count)
				if math.isnan(centroid[cluster, i]) == True:
					centroid[cluster, i] = 0
		
		q1 = np.zeros((n_cluster))
		for j in range(n_sample):
			cluster = partition[j]
			q1[cluster] += np.linalg.norm(data[j, :] - centroid[cluster, :])
		
		# get new Q
		Q_new = np.sum(q1)
		diff = np.abs(Q_new - Q)
		Q = Q_new
		numiter += 1
	
	return partition
			
	
		
if __name__ == '__main__':
	np.set_printoptions(threshold=np.inf)
	
	parser = argparse.ArgumentParser(description='A simple fully connected network.')
	parser.add_argument('-dp', dest='dp', action='store_true', default=False, help='Use privacy preserving learning.')
	parser.add_argument('-eps', dest='eps', default=10, type=float, help='Epsilon differential privacy.')
	args = parser.parse_args()

	n_class, X, Y = load_iris()

	accuracy = []
	for i in range(20):
		if args.dp:
			print('Use differential privacy')
			eps = float(args.eps)
			y_pred = myKmeans_DP(n_class, X, init=20, eps=eps)
		else:
			print('Not use differential privacy')
			y_pred = myKmeans(n_class, X, init=20)


		label_list = np.unique(Y).tolist()
		acc = util.accuracy(Y, y_pred)
		Y_0 = copy.deepcopy(Y)
		for j in range(30):
			#random.shuffle(label_list)
			label_list = np.random.permutation(n_class)
			for i in range(n_class):
				Y_0[all_index(Y,i)] = label_list[i]
			temp=util.accuracy(Y_0, y_pred)
			if temp>acc:
				acc = temp

		print(acc)
		accuracy.append(acc)
		
	# print(Y.tolist())
	# print(y_pred.tolist())
	print('Mean accuracy=', np.mean(accuracy))
	print('When epsilon is really small, the result will be highly random.')
	
	
	
	
	
	
