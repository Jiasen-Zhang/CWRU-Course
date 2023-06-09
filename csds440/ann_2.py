#!/usr/bin/python3
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import argparse
import copy
import util
from sklearn.preprocessing import LabelEncoder
from sting.data import parse_c45


def oneHot(label, n_class):
	return torch.eye(n_class)[label, :]


def load_volcanoes():
	# with parse_c45
	schema, X, Y = parse_c45('volcanoes', '440data')
	Y = Y.reshape(Y.shape[0], 1)
	n_class = len(np.unique(Y))
	n_feature = X.shape[1]

	# transform the labels
	encoder = LabelEncoder()
	Y_encoder = encoder.fit_transform(Y.ravel())
	
	return X, Y_encoder, n_class, n_feature
	
	
# load data
def load_iris():
	# with pandas
	df = pd.read_csv('440data/iris/iris.csv')
	X = df.values[:, :-1].astype(float)
	Y = df.values[:, -1]
	n_class = len(np.unique(Y))
	n_feature = X.shape[1]

	# transform the labels
	encoder = LabelEncoder()
	Y_encoder = encoder.fit_transform(Y.ravel())
	
	return X, Y_encoder, n_class, n_feature


class MyNetwork(nn.Module):
	def __init__(self, n_input, n_output):
		super(MyNetwork, self).__init__()
		#self.flat = nn.Flatten()
		self.lin1 = nn.Linear(n_input, 150, bias=True)
		self.lin2 = nn.Linear(150, 150, bias=True)
		self.lin3 = nn.Linear(150, n_output, bias=True)
		self.sigmoid = nn.Sigmoid()
		self.softmax = nn.Softmax(dim=1)
		
	def forward(self, x):
		#x = self.flat(x)
		x = self.lin1(x)
		x = self.sigmoid(x)
		x = self.lin2(x)
		x = self.sigmoid(x)
		x = self.lin3(x)
		x = self.softmax(x)
		return x


class MyNetwork_DP(nn.Module):
	def __init__(self, n_input, n_output, eps):
		super(MyNetwork_DP, self).__init__()
		# self.flat = nn.Flatten()
		self.lin1 = nn.Linear(n_input, 150, bias=True)
		self.lin2 = nn.Linear(150, 150, bias=True)
		self.lin3 = nn.Linear(150, n_output, bias=True)
		self.sigmoid = nn.Sigmoid()
		self.softmax = nn.Softmax(dim=1)
		self.v = 1/eps
	
	def forward(self, x):
		# x = self.flat(x)
		x = self.lin1(x)
		x = self.sigmoid(x)
		x = self.lin2(x)
		x = self.sigmoid(x)
		x = self.lin3(x)
		x = self.softmax(x)
		x = torch.add(x, torch.tensor(np.random.laplace(0, self.v, size=x.shape)).to(device=device))
		#x = self.softmax(x)
		return x


def train(network, data, label, epochs=5000, lr=0.01):
	network.train()
	lossF = nn.CrossEntropyLoss()
	optimizer = torch.optim.SGD(network.parameters(), lr=lr, momentum=0)
	
	for epoch in range(epochs):
		output = network(data)
		loss = lossF(output, label)
		
		# backpropagation
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		
		if (epoch % 1000) == 999:
			print('epoch=', epoch+1, 'loss=', loss.item())


		
if __name__ == '__main__':
	# iris: lr = 0.1
	# volcanoes: lr = 0.01
	np.set_printoptions(threshold=np.inf)
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	

	parser = argparse.ArgumentParser(description='A simple fully connected network.')
	parser.add_argument('dataset', type=str, help='Name of dataset, iris or volcanoes.')
	parser.add_argument('-dp', dest='dp', action='store_true', default=False, help='Use privacy preserving learning.')
	parser.add_argument('-eps', dest='eps', default=10, type=float, help='Epsilon differential privacy.')
	args = parser.parse_args()
	dataset = args.dataset


	# save the accuracy of each cross validation
	accuracy = []


	if dataset == 'iris':
		print('Dataset = iris')
		lr = 0.1
		X, Y, n_class, n_feature = load_iris()
	elif dataset == 'volcanoes':
		print('Dataset = volcanoes')
		lr = 0.01
		X, Y, n_class, n_feature = load_volcanoes()
	else:
		raise argparse.ArgumentTypeError('Only support iris and volcanoes!')


	# number of cross validation
	k=5
	# save the accuracy of each cross validation
	accuracy = []
	X_list, Y_list = util.cv_split(X, Y, k)
	# train and predict with cross validation
	for h in range(k):
		tmp_X_list = copy.deepcopy(X_list)
		tmp_Y_list = copy.deepcopy(Y_list)
		testing_data = tmp_X_list.pop(h)
		testing_label = tmp_Y_list.pop(h)

		# get training data and label
		training_data = tmp_X_list[0]
		training_label = tmp_Y_list[0]
		for i in range(k - 2):
			training_data = np.vstack((training_data, tmp_X_list[i + 1]))
			training_label = np.hstack((training_label, tmp_Y_list[i + 1]))

		# transform the label to onehot
		testing_label = oneHot(testing_label, n_class)
		training_label = oneHot(training_label, n_class)

		# convert to tensor
		training_data = torch.tensor(training_data, dtype=torch.float32)
		testing_data = torch.tensor(testing_data, dtype=torch.float32)
		training_label = training_label.float()

		training_data = training_data.to(device=device)
		training_label = training_label.to(device=device)
		testing_data = testing_data.to(device=device)
		testing_label = testing_label.tolist()

		# train
		if args.dp:
			print('Use differential privacy')
			eps = args.eps
			network = MyNetwork_DP(n_input=n_feature, n_output=n_class, eps=eps)
		else:
			print('Not use differential privacy')
			network = MyNetwork(n_input=n_feature, n_output=n_class)
		network.to(device=device)

		train(network, training_data, training_label, epochs=8000, lr=lr)
		print('Training finished')


		# predict
		network.eval()
		with torch.no_grad():
			prediction = network(testing_data)
		prediction = torch.Tensor.tolist(prediction)

		# evaluate the result
		pred_label = []
		true_label = []
		for i in range(np.size(prediction,0)):
			temp = prediction[i]
			pred_label.append(temp.index(max(temp))+1)
			temp = testing_label[i]
			true_label.append(temp.index(max(temp))+1)

		# print('Predicted label and true label:')
		# print(pred_label)
		# print(true_label)
		acc = util.accuracy(np.array(true_label), np.array(pred_label))

		print('Cross validation',h+1, ', acc=',acc)
		accuracy.append(acc)

	# print the mean accuracy
	print('Mean accuracy=', np.mean(accuracy))
	

