from __future__ import print_function, division
import argparse
import os
import time
from threading import Lock
import torch
import torch.distributed.autograd as dist_autograd
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.distributed.optim import DistributedOptimizer
import torch.distributed as dist
from torchvision import datasets, transforms
import ipdb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import copy
import os
from PIL import ImageFile
import random
ImageFile.LOAD_TRUNCATED_IMAGES = True
import argparse
import pickle
import sys
import ray
sys.path.append(os.getcwd() + "/res")

from res.loader.loader import get_loader


##### Details for different data loaders created ######
CODE_ROOT = './'


DATASET_NAMES = ['mnist_rotation_one_by_nine', 'mnist_rotation_three_by_nine',
                 'mnist_rotation_six_by_nine']
OOD_DATASET_NAME = 'mnist_rotation_nine_by_nine'
NUM_EPOCHS = 10
BATCH_SIZE = 4
ARCH = 'LATE_BRANCHING_COMBINED'

image_transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])


GPU = 1

NUM_CLASSES = (10,10,10,10)
loader_new = get_loader('multi_attribute_loader_file_list_mnist_rotation')

file_list_root = '%s/dataset_lists/mnist_rotation_lists/'%CODE_ROOT
att_path = '%s/dataset_lists/combined_attributes.p'%CODE_ROOT

shuffles = {'train':True,'val':True,'test':False}

data_dir = '%s/data/'%CODE_ROOT

#### Function to build different data loaders ####
def build_loaders_for_dataset(DATASET_NAME):
    file_lists = {}
    dsets = {}
    dset_loaders = {}
    dset_sizes = {}
    for phase in ['train','test']:
        file_lists[phase] = "%s/%s_list_%s.txt"%(file_list_root,phase,DATASET_NAME)
        dsets[phase] = loader_new(file_lists[phase],att_path, image_transform, data_dir)
        dset_loaders[phase] = torch.utils.data.DataLoader(dsets[phase], batch_size=BATCH_SIZE, shuffle = shuffles[phase], num_workers=2,drop_last=True)
        dset_sizes[phase] = len(dsets[phase])
    return dsets, dset_loaders, dset_sizes



# --------- MNIST Network to train, from pytorch/examples -----
'''
Definition for Neural Networks. We could replace the architecture with our network design
#TODO: Update the network architecture
'''
class ConvNet(nn.Module):
    """Small ConvNet for MNIST."""

    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 28, 3, 1)
        self.conv2 = nn.Conv2d(28, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)

        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

    def get_weights(self):
        return {k: v.cpu() for k, v in self.state_dict().items()}

    def set_weights(self, weights):
        self.load_state_dict(weights)

    def get_gradients(self):
        grads = []
        for p in self.parameters():
            grad = None if p.grad is None else p.grad.data.cpu().numpy()
            grads.append(grad)
        return grads

    def set_gradients(self, gradients):
        for g, p in zip(gradients, self.parameters()):
            if g is not None:
                p.grad = torch.from_numpy(g)


@ray.remote
class ParameterServer(object):
    def __init__(self, lr):
        self.model = ConvNet()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)

    def apply_gradients(self, *gradients):
        summed_gradients = [
            np.stack(gradient_zip).sum(axis=0) for gradient_zip in zip(*gradients)
        ]
        self.optimizer.zero_grad()
        self.model.set_gradients(summed_gradients)
        self.optimizer.step()
        return self.model.get_weights()

    def get_weights(self):
        return self.model.get_weights()


@ray.remote
class DataWorker(object):
    def __init__(self, train_data_loader, test_data_loader):
        self.model = ConvNet()
        self.train_data_loader = train_data_loader
        self.test_data_loader = test_data_loader
        self.data_iterator = iter(train_data_loader)

    def compute_gradients(self, weights):
        self.model.set_weights(weights)
        try:
            [data, target, _] = next(self.data_iterator)
            target = target[3]
        except StopIteration:  # When the epoch ends, start a new epoch.
            self.data_iterator = iter(self.train_data_loader)
            [data, target, _] = next(self.data_iterator)
            target = target[3]
        self.model.zero_grad()
        output = self.model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        return self.model.get_gradients()

def evaluate(model, test_loader):
    """Evaluates the accuracy of the model on a validation dataset."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (data, target, paths) in enumerate(test_loader):
            # This is only set to finish evaluation faster.
            target = target[3]
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    return 100.0 * correct / total

if __name__ == "__main__":
    num_workers = 3
    iterations = 10000
    print("Running Asynchronous Parameter Server Training.")
    ray.init(ignore_reinit_error=True)
    ps = ParameterServer.remote(1e-2)
    ood_rank_dset, ood_rank_loaders, ood_rank_dset_sizes = build_loaders_for_dataset(OOD_DATASET_NAME)
    train_loader = ood_rank_loaders['train']
    test_loader = ood_rank_loaders['test']
    workers = [DataWorker.remote(train_loader, test_loader) for i in range(num_workers)]
    current_weights = ps.get_weights.remote()
    model = ConvNet()
    gradients = {}
    
    for worker in workers:
        gradients[worker.compute_gradients.remote(current_weights)] = worker

    for i in range(iterations * num_workers):
        ready_gradient_list, _ = ray.wait(list(gradients))
        ready_gradient_id = ready_gradient_list[0]
        worker = gradients.pop(ready_gradient_id)

        # Compute and apply gradients.
        current_weights = ps.apply_gradients.remote(*[ready_gradient_id])
        gradients[worker.compute_gradients.remote(current_weights)] = worker

        if i % 10 == 0:
            # Evaluate the current model after every 10 updates.
            model.set_weights(ray.get(current_weights))
            accuracy = evaluate(model, test_loader)
            print("Iter {}: \taccuracy is {:.1f}".format(i, accuracy))

    print("Final accuracy is {:.1f}.".format(accuracy))