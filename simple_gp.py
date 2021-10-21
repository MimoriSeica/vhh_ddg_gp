import json
import random
import numpy as np
import pandas as pd
import math
import torch
import gpytorch
from sklearn.model_selection import KFold


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
def train(training_times, model, likelihood, optimizer, mll, train_x, train_y):
    cou = []
    losses = []
    for i in range(training_times):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        if (i + 1) % 10 == 0:
            print('Iter %d/%d - Loss: %.5f' % (i + 1,
                                               training_times,
                                               loss.item()))
        cou.append(i)
        losses.append(loss.item())
        optimizer.step()

def predict(train_x, train_y, test_x, training_times):
    train_x = torch.Tensor(train_x)
    train_y = torch.Tensor(train_y)
    
    # print(train_x.device)
    # print(train_y.device)
    
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(train_x, train_y, likelihood)

    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam([
        {'params': model.parameters()},  # Includes GaussianLikelihood parameters
    ], lr=0.1)

    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    train(training_times, model, likelihood, optimizer, mll, train_x, train_y)

    model.eval()
    likelihood.eval()
    
    pred = likelihood(model(torch.Tensor(test_x)))
    
    # print(pred.mean.device)
    
    means = pred.mean.detach().numpy()
    lower, upper = pred.confidence_region()
    lower = lower.detach().numpy()
    upper = upper.detach().numpy()
    return means, lower, upper


def solve(data_x, data_y, n_splits, training_times, shuffle=True, random_state=1234):
    kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    ndatas = data_x.shape[0]
    pred_y = np.zeros(ndatas)
    lower_y = np.zeros(ndatas)
    upper_y = np.zeros(ndatas)
    for index, (train_indices, valid_indices) in enumerate(kf.split(range(ndatas))):
        # print(index)
        # print(train_indices)
        # print(valid_indices)
        train_x, test_x = data_x[train_indices], data_x[valid_indices] 
        train_y = data_y[train_indices]
        means, lower, upper = predict(train_x, train_y, test_x, training_times)
        pred_y[valid_indices] = means
        lower_y[valid_indices] = lower
        upper_y[valid_indices] = upper

    return pred_y, lower_y, upper_y