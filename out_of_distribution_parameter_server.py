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
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
sys.path.append('./res/')
# sys.path.append('./res/models/')
# sys.path.append('./res/loader/')
from models.models import get_model
from loader.loader import get_loader

##### Details for different data loaders created ######
CODE_ROOT = './'


# DATASET_NAMES = ['mnist_rotation_one_by_nine', 'mnist_rotation_one_by_nine',
                 # 'mnist_rotation_one_by_nine','mnist_rotation_one_by_nine']
OOD_DATASET_NAME = 'mnist_rotation_nine_by_nine'
NUM_EPOCHS = 10
BATCH_SIZE = 100
ARCH = 'LATE_BRANCHING_COMBINED'

image_transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])


GPU = 1

TERMINATE_AT_ITER = None  # for early stopping when debugging
NUM_TRAINERS_WAITED = 0
num_trainers_waited_lock = Lock()

def trainer_arrived():
    with num_trainers_waited_lock:
        global NUM_TRAINERS_WAITED
        NUM_TRAINERS_WAITED += 1

def wait_all_trainers(rank, world_size):
    global NUM_TRAINERS_WAITED
    # Send RPC to all other trainers (non-zero ranks)
    for i in range(world_size):
        if i != rank and i != 0:
            rpc.rpc_sync(f"trainer_{i}", trainer_arrived, args=())
    # Wait for all trainers to arrive
    with num_trainers_waited_lock:
        cur_num_trainers = NUM_TRAINERS_WAITED

    if cur_num_trainers != world_size - 1:
        time.sleep(0.01)
        with num_trainers_waited_lock:
            cur_num_trainers = NUM_TRAINERS_WAITED



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
    for phase in ['train','val','test']:
        file_lists[phase] = "%s/%s_list_%s.txt"%(file_list_root,phase,DATASET_NAME)
        dsets[phase] = loader_new(file_lists[phase],att_path, image_transform, data_dir)
        dset_loaders[phase] = torch.utils.data.DataLoader(dsets[phase], batch_size=BATCH_SIZE, shuffle = shuffles[phase], num_workers=1,drop_last=True)
        dset_sizes[phase] = len(dsets[phase])
    return dsets, dset_loaders, dset_sizes



# --------- MNIST Network to train, from pytorch/examples -----
'''
Definition for Neural Networks. We could replace the architecture with our network design
#TODO: Update the network architecture
'''
class Net(nn.Module):
    def __init__(self, num_gpus=0):
        super(Net, self).__init__()
        print(f"Using {num_gpus} GPUs to train", flush=True)
        self.num_gpus = num_gpus
        device = torch.device(
            "cuda:0" if torch.cuda.is_available() and self.num_gpus > 0 else "cpu")
        print(f"Putting first 2 convs on {str(device)}", flush=True)
        # Put conv layers on the first cuda device, or CPU if no cuda device
        self.conv1 = nn.Conv2d(3, 28, 3, 1).to(device)
        self.conv2 = nn.Conv2d(28, 64, 3, 1).to(device)
        # Put rest of the network on the 2nd cuda device, if there is one
        if "cuda" in str(device) and num_gpus > 1:
            device = torch.device("cuda:1")
        '''
        The setup here only supports 0-2 gpus, can be extended if more available
        '''

        print(f"Putting rest of layers on {str(device)}", flush=True)
        self.dropout1 = nn.Dropout2d(0.25).to(device)
        self.dropout2 = nn.Dropout2d(0.5).to(device)
        self.fc1 = nn.Linear(9216, 128).to(device)
        self.fc2 = nn.Linear(128, 10).to(device)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)

        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        # Move tensor to next device if necessary
        '''
        This is necessary because we need to make sure the tensor we operate need to be on the same device
        '''
        next_device = next(self.fc1.parameters()).device
        x = x.to(next_device)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output





# --------- Helper Methods --------------------

# On the local node, call a method with first arg as the value held by the
# RRef. Other args are passed in as arguments to the function called.
# Useful for calling instance methods. method could be any matching function, including
# class methods.
def call_method(method, rref, *args, **kwargs):
    return method(rref.local_value(), *args, **kwargs)

# Given an RRef, return the result of calling the passed in method on the value
# held by the RRef. This call is done on the remote node that owns
# the RRef and passes along the given argument.
# Example: If the value held by the RRef is of type Foo, then
# remote_method(Foo.bar, rref, arg1, arg2) is equivalent to calling
# <foo_instance>.bar(arg1, arg2) on the remote node and getting the result
# back.

def remote_method(method, rref, *args, **kwargs):
    args = [method, rref] + list(args)
    '''
    rpc.rpc_sync blocks the program until it gets the result from the remote machine
    '''
    return rpc.rpc_sync(rref.owner(), call_method, args=args, kwargs=kwargs)

# --------- Parameter Server --------------------
class ParameterServer(nn.Module):
    def __init__(self, num_gpus=0):
        torch.autograd.set_detect_anomaly(True)
        super().__init__()
        # model = Net(num_gpus=num_gpus)
        model = torchvision.models.resnet18(pretrained = False)
        num_final_in = model.fc.in_features
        NUM_CLASSES = 10
        model.fc = nn.Linear(num_final_in, NUM_CLASSES)

        self.model = model
        self.input_device = torch.device(
            "cuda:0" if torch.cuda.is_available() and num_gpus > 0 else "cpu")
    def forward(self, inp):
        inp = inp.to(self.input_device)
        out = self.model(inp)
        # This output is forwarded over RPC, which as of 1.5.0 only accepts CPU tensors.
        # Tensors must be moved in and out of GPU memory due to this.
        out = out.to("cpu")
        return out
    # def save_model(self,save_path):
    #     with open(save_path,'wb') as F:
    #         torch.save(self.model, F)

    # Use dist autograd to retrieve gradients accumulated for this model.
    # Primarily used for verification.
    def get_dist_gradients(self, cid):
        '''
        remote method called by workers
        '''

        grads = dist_autograd.get_gradients(cid)
        # This output is forwarded over RPC, which as of 1.5.0 only accepts CPU tensors.
        # Tensors must be moved in and out of GPU memory due to this.
        cpu_grads = {}
        for k, v in grads.items():
            k_cpu, v_cpu = k.to("cpu"), v.to("cpu")
            cpu_grads[k_cpu] = v_cpu
        return cpu_grads

    # Wrap local parameters in a RRef. Needed for building the
    # DistributedOptimizer which optimizes paramters remotely.
    def get_param_rrefs(self):
        '''
        return a reference of all the parameters in this model
        so the distributed optimizer could use
        '''
        param_rrefs = [rpc.RRef(param) for param in self.model.parameters()]
        return param_rrefs



# The global parameter server instance.
param_server = None
# A lock to ensure we only have one parameter server.
global_lock = Lock()


def get_parameter_server(num_gpus=0):
    """
    Returns a singleton parameter server to all trainer processes
    """
    global param_server
    # Ensure that we get only one handle to the ParameterServer.
    with global_lock:
        if not param_server:
            # construct it once
            param_server = ParameterServer(num_gpus=num_gpus)
        return param_server

def run_parameter_server(rank, world_size):
    # The parameter server just acts as a host for the model and responds to
    # requests from trainers.
    # rpc.shutdown() will wait for all workers to complete by default, which
    # in this case means that the parameter server will wait for all trainers
    # to complete, and then exit.
    print("PS master initializing RPC", flush=True)
    rpc.init_rpc(name="parameter_server", rank=rank, world_size=world_size)
    print("RPC initialized! Running parameter server...", flush=True)
    rpc.shutdown()
    print("RPC shutdown on parameter server.", flush=True)

'''
TrainerNet is a class for trainers that consists of a singleton parameter server ref
'''

class TrainerNet(nn.Module):
    def __init__(self, num_gpus=0):
        super().__init__()
        self.num_gpus = num_gpus
        '''
        get a reference to the parameter instance
        '''
        self.param_server_rref = rpc.remote(
            "parameter_server", get_parameter_server, args=(num_gpus,))

    def get_global_param_rrefs(self):
        remote_params = remote_method(
            ParameterServer.get_param_rrefs,
            self.param_server_rref)
        return remote_params

    def forward(self, x):
        model_output = remote_method(
            ParameterServer.forward, self.param_server_rref, x)
        return model_output

    # def save_trainer(self, save_path):
        # param_server.save_model(ParameterServer,save_path=save_path)

from threading import Condition
trainer_cv = Condition()
def set_cv():
    global trainer_cv
    with trainer_cv:
        trainer_cv.notify()

def get_accuracy(test_loader, model):
    model.eval()
    correct_sum = 0
    # Use GPU to evaluate if possible
    device = torch.device("cuda:0" if model.num_gpus > 0
        and torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        for i, (data, target, paths) in enumerate(test_loader):
            target = target[:,1]
            out = model(data)
            pred = out.argmax(dim=1, keepdim=True)
            pred, target = pred.to(device), target.to(device)
            correct = pred.eq(target.view_as(pred)).sum().item()
            correct_sum += correct
    print(f"Accuracy {correct_sum / len(test_loader.dataset)}", flush=True)


def run_training_loop(net,rank, world_size, num_gpus, train_loader, test_loader, ood_test_loader, corruption_rate):
    # Runs the typical nueral network forward + backward + optimizer step, but
    # in a distributed fashion.
    # Build DistributedOptimizer.
    param_rrefs = net.get_global_param_rrefs()
    # param_rrefs = param_rrefs[:10]
    for i, (data, target, paths) in enumerate(train_loader):
        if TERMINATE_AT_ITER is not None and i == TERMINATE_AT_ITER:
            break
        '''
        generates a context cid for each worker for parameter to accumulate gradeients
        '''
        with dist_autograd.context() as cid:
            if rank == 1:
                with trainer_cv:
                    trainer_cv.wait()
            model_output = net(data)
            # print(target)
            # print(target[:,3])
            target = target[:,1]
            # print('targets:%s'%target)
            # print('paths:')
            # print(paths)
            target = target.to(model_output.device)
            CE_loss = torch.nn.CrossEntropyLoss()
            loss = CE_loss(model_output, target)

            if i % 5 == 0:
                print(f"Rank {rank} training batch {i} loss {loss.item()}", flush=True)
            # wait_all_trainers(rank, world_size)
            '''
            # Run the backward pass.
            dist_autograd.backward(context_id, [loss])
            # Retrieve the gradients from the context.
            dist_autograd.get_gradients(context_id)
            '''
            if random.random() < corruption_rate:
                # loss = loss/100
                # print('using prev cid %s'%prev_cid)
                param_rrefs = net.get_global_param_rrefs()
                param_rrefs = param_rrefs[:int(len(param_rrefs)/10)]
                opt = DistributedOptimizer(optim.SGD, param_rrefs, lr=0.001)
                dist_autograd.backward(cid, [loss])
                # print(len(param_rrefs))
            else:
                param_rrefs = net.get_global_param_rrefs()
                # print(len(param_rrefs))
                opt = DistributedOptimizer(optim.SGD, param_rrefs, lr=0.001)
                dist_autograd.backward(cid, [loss])
                prev_cid = cid
            opt.step(cid)

        if rank != 1 :
            rpc.rpc_sync("trainer_%s"%(int(rank)-1), set_cv, args=())
            with trainer_cv:
                trainer_cv.wait()

        if rank == 1:
            rpc.rpc_sync("trainer_%s"%(int(world_size) - 1), set_cv, args=())

    print("Training complete!", flush=True)
    print("Getting accuracy....", flush=True)
    print('In-D accuracy...', flush=True)
    get_accuracy(test_loader, net)
    print('OOD accuracy...', flush=True)
    get_accuracy(ood_test_loader ,net)

# Main loop for trainers.
def run_worker(rank, world_size, num_gpus, train_loader, test_loader, ood_test_loader, corruption_rate, num_epochs, model_save_name):
    print(f"Worker rank {rank} initializing RPC", flush=True)

    '''
    name (str) – a globally unique name of this node.
    rank (int) – a globally unique id/rank of this node.
    world_size (int) – The number of workers in the group.
    '''

    rpc.init_rpc(
        name=f"trainer_{rank}",
        rank=rank,
        world_size=world_size)

    print(f"Worker {rank} done initializing RPC", flush=True)
    net = TrainerNet(num_gpus=num_gpus)
    save_name = '/Users/spandanmadan/saved_models/%s_rank_%s.pt'%(model_save_name, rank)
    print('saving as %s'%save_name)
    # torch.save(net,save_name)
    # net.save_trainer(save_name)
    for epoch_num in range(num_epochs):
        print('Starting Epoch:%s'%epoch_num, flush=True)
        run_training_loop(net,rank, world_size, num_gpus, train_loader, test_loader, ood_test_loader, corruption_rate)
    print('saving model', flush=True)
    # net.save_trainer('/Users/spandanmadan/saved_models/%s_rank_%s.pt'%(model_save_name, rank))
    rpc.shutdown()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Parameter-Server RPC based training")
    parser.add_argument(
        "--world_size",
        type=int,
        default=4,
        help="""Total number of participating processes. Should be the sum of
        master node and all training nodes.""")
    parser.add_argument(
        "--rank",
        type=int,
        default=None,
        help="Global rank of this process. Pass in 0 for master.")
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=0,
        help="""Number of GPUs to use for training, Currently supports between 0
         and 2 GPUs. Note that this argument will be passed to the parameter servers.""")
    parser.add_argument(
        "--master_addr",
        type=str,
        default="localhost",
        help="""Address of master, will default to localhost if not provided.
        Master must be able to accept network traffic on the address + port.""")
    parser.add_argument(
        "--master_port",
        type=str,
        default="29500",
        help="""Port that master is listening on, will default to 29500 if not
        provided. Master must be able to accept network traffic on the host and port.""")

    parser.add_argument(
        "--corruption_rate",
        type=float,
        default= 0.0,
        help="""Corruption rate for all the workers. If corruption rate is 0.0, then there
        won't be any corruption. Otherwise, each worker will have the corruption rate chance
        to drop the gradient update.""")

    parser.add_argument('--dataset_names',
        nargs='+',
        help='list of datasets for models',
        required=True)

    parser.add_argument('--model_save_name',
        type=str,
        help='name stub for models',
        required=True)


    parser.add_argument(
        "--num_epochs",
        type=int,
        default= 1,
        help="""Number of epochs agents should be trained""")

    args = parser.parse_args()
    assert args.rank is not None, "must provide rank argument."
    assert args.num_gpus <= 3, f"Only 0-2 GPUs currently supported (got {args.num_gpus})."
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ["MASTER_PORT"] = args.master_port


    processes = []
    world_size = args.world_size

    mp.set_start_method("spawn")
    if args.rank == 0:
        '''
        rank 0 starts the parameter server
        '''
        p = mp.Process(target=run_parameter_server, args=(0, world_size))
        p.start()
        processes.append(p)
    else:
        '''
        Other starts the worker process
        '''
        # Get data to train on
        rank_dataset_name = args.dataset_names[args.rank-1]
        print('Building train + in-distribution test data loader from %s'%rank_dataset_name, flush=True)
        print('Building OOD test data loader from %s'%OOD_DATASET_NAME, flush=True)

        rank_dset, rank_loaders, rank_dset_sizes = build_loaders_for_dataset(rank_dataset_name)
        ood_rank_dset, ood_rank_loaders, ood_rank_dset_sizes = build_loaders_for_dataset(OOD_DATASET_NAME)
        train_loader, ind_test_loader = rank_loaders['train'], rank_loaders['val']
        ood_test_loader = ood_rank_loaders['test']

        print('loaders done, starting training...', flush=True)
        p = mp.Process(
            target=run_worker,
            args=(
                args.rank,
                world_size,
                args.num_gpus,
                train_loader,
                ind_test_loader,
                ood_test_loader,
                args.corruption_rate,
                args.num_epochs,
                args.model_save_name))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
