#!/usr/bin/python3
# Fully connected network with DP-SGD
import pandas as pd
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
from torch.optim.optimizer import required
from torch.optim.optimizer import Optimizer
from typing import List, Optional
import torch.nn.functional as F
import argparse
import copy
import util
from math import log, sqrt
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


def my_dp_sgd(params: List[Tensor],
        d_p_list: List[Tensor],
        momentum_buffer_list: List[Optional[Tensor]],
        *,
        weight_decay: float,
        momentum: float,
        lr: float,
        dampening: float,
        nesterov: bool,
        sigma):
    r"""Functional API that performs SGD algorithm computation.

    See :class:`~torch.optim.SGD` for details.
    """
    threshold = 1
    for i, param in enumerate(params):
        d_p = d_p_list[i]

        # clip the gradient
        d_p /= max(1, torch.norm(d_p, p=2) / threshold)

        # add Gaussian noise
        mu = threshold * sigma
        d_p += torch.normal(0, mu*mu, size=d_p.shape).to(device=device)

        if weight_decay != 0:
            d_p = d_p.add(param, alpha=weight_decay)

        if momentum != 0:
            buf = momentum_buffer_list[i]

            if buf is None:
                buf = torch.clone(d_p).detach()
                momentum_buffer_list[i] = buf
            else:
                buf.mul_(momentum).add_(d_p, alpha=1 - dampening)

            if nesterov:
                d_p = d_p.add(buf, alpha=momentum)
            else:
                d_p = buf

        param.add_(d_p, alpha=-lr)


class my_DPSGD(Optimizer):
    """Implements stochastic gradient descent (optionally with momentum).

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)
    """

    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, eps=10):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(my_DPSGD, self).__init__(params, defaults)
        self.eps = eps

    def __setstate__(self, state):
        super(my_DPSGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            d_p_list = []
            momentum_buffer_list = []
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            lr = group['lr']

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    d_p_list.append(p.grad)

                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        momentum_buffer_list.append(None)
                    else:
                        momentum_buffer_list.append(state['momentum_buffer'])

            # one step of SGD
            # parameters for DPSGD
            delt = 1e-5
            sigma = sqrt(2 * log(1.25/delt)) / self.eps
            my_dp_sgd(params_with_grad,
                  d_p_list,
                  momentum_buffer_list,
                  weight_decay=weight_decay,
                  momentum=momentum,
                  lr=lr,
                  dampening=dampening,
                  nesterov=nesterov,
                  sigma = sigma)

            # update momentum_buffers in state
            for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
                state = self.state[p]
                state['momentum_buffer'] = momentum_buffer

        return loss


class MyNetwork(nn.Module):
    def __init__(self, n_input, n_output, hidden):
        super(MyNetwork, self).__init__()
        #self.flat = nn.Flatten()
        self.lin1 = nn.Linear(n_input, hidden, bias=True)
        self.lin2 = nn.Linear(hidden, hidden, bias=True)
        self.lin3 = nn.Linear(hidden, n_output, bias=True)
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


def train_DP(network, data, label, epochs, lr, eps):
    # T = number of training samples
    network.train()
    lossF = nn.CrossEntropyLoss()
    optimizer = my_DPSGD(network.parameters(), lr=lr, eps=eps)

    for epoch in range(epochs):
        output = network(data)
        loss = lossF(output, label)

        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch % 1000) == 999:
            print('epoch=', epoch+1, 'loss=', loss.item())


def train(network, data, label, epochs=2000, lr=0.01):
    network.train()
    lossF = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(network.parameters(), lr=lr)

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
    np.set_printoptions(threshold=np.inf)
    # use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    parser = argparse.ArgumentParser(description='A simple fully connected network.')
    parser.add_argument('dataset', type=str, help='Name of dataset, iris or volcanoes.')
    parser.add_argument('-dp', dest='dp', action='store_true', default=False, help='Use privacy preserving learning.')
    parser.add_argument('-eps', dest='eps', default=10, type=float, help='Epsilon differential privacy.')
    args = parser.parse_args()
    dataset = args.dataset

    if dataset == 'iris':
        print('Dataset = iris')
        lr = 0.1
        hidden = 80
        nepoch = 10000
        X, Y, n_class, n_feature = load_iris()
    elif dataset == 'volcanoes':
        print('Dataset = volcanoes')
        lr = 0.01
        hidden = 150
        nepoch = 10000
        X, Y, n_class, n_feature = load_volcanoes()
    else:
        raise argparse.ArgumentTypeError('Only support iris and volcanoes!')


    # number of cross validation
    k = 5
    # save the accuracy of each cross validation
    n_train = int(len(Y)*(k-1)/k)
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
        network = MyNetwork(n_input=n_feature, n_output=n_class, hidden=hidden)
        network.to(device=device)

        if args.dp:
            print('Use differential privacy')
            eps = args.eps
            train_DP(network, training_data, training_label, epochs=nepoch, lr=lr, eps=eps)
        else:
            print('Not use differential privacy')
            train(network, training_data, training_label, epochs=nepoch, lr=lr)

        print('Training finished')


        # predict
        network.eval()
        with torch.no_grad():
            prediction = network(testing_data)
        prediction = torch.Tensor.tolist(prediction)


        # evaluate
        pred_label = []
        true_label = []
        for i in range(np.size(prediction,0)):
            temp = prediction[i]
            pred_label.append(temp.index(max(temp))+1)
            temp = testing_label[i]
            true_label.append(temp.index(max(temp))+1)

        acc = util.accuracy(np.array(true_label), np.array(pred_label))

        print('Cross validation', h + 1, ', acc=', acc)
        accuracy.append(acc)
    # cross validation end

    # print('Predicted label and true label:')
    # print(pred_label)
    # print(true_label)
    print('Mean Accuracy=', np.mean(accuracy))
    print('When epsilon is really small (<5), it\'s hard to converge and highly random.')


