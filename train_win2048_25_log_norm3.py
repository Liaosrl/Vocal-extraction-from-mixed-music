#!/usr/bin/env python
# coding: utf-8

import numpy as np
import librosa
import os
import h5py
import time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

exp_name = '2048_25_norm1'
if not os.path.exists(exp_name):
    os.makedirs(exp_name)
parser = argparse.ArgumentParser(description=exp_name)

# add hyperparameters to the parser
parser.add_argument('--batch-size', type=int, default=16,
                    help='input batch size for training (default: 8)')
parser.add_argument('--epochs', type=int, default=300,
                    help='number of epochs to train (default: 100)')
parser.add_argument('--freq-dim', type=int, default=1025,
                    help='number of epochs to train (default: 100)')
parser.add_argument('--cuda', action='store_true', default=True,
                    help='enables CUDA training (default: True)')  # when you have a GPU
parser.add_argument('--lr', type=float, default=0.05e-6,
                    help='learning rate (default: 1e-5)')
parser.add_argument('--model-save', type=str,  default='C:/Users/socia/OneDrive/Documents/COLUMBIA/Speech and Audio/project/2048_25/win2048_25_log_norm3.pt',
                    help='path to save the best model')
parser.add_argument('--optimizer-save', type=str,  default='C:/Users/socia/OneDrive/Documents/COLUMBIA/Speech and Audio/project/2048_25/win2048_25_log_norm_op.pt',
                    help='path to save the opt')
parser.add_argument('--tr-data', type=str,  
                    default='D:/speech_data/2048_25/tr_set_log_norm3.hdf5',
                    help='path to training dataset')
parser.add_argument('--val-data', type=str,  
                    default='D:/speech_data/2048_25/va_set_log_norm3.hdf5',
                    help='path to validation dataset')
parser.add_argument('--test-data', type=str,  default='test_set_log_norm3.hdf5',
                    help='pa    th to training dataset')
parser.add_argument('--load-model', type=bool,  default=True,
                    help='continue last process (default: True)')
parser.add_argument('--load-loss', type=bool,  default=True,
                    help='continue last process (default: True)')

args, _ = parser.parse_known_args()
args.cuda = args.cuda and torch.cuda.is_available()

if args.cuda:
    kwargs = {'num_workers': 0, 'pin_memory': True} 
else:
    kwargs = {}

#####################################
##  CNN
#####################################
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class CNN2D(nn.Module):
    def __init__(self):
        super(CNN2D, self).__init__()
        self.channel1 = 64
        self.channel2 = 128
        self.channel3 = 32
        self.kernel1 = 3
        self.kernel2 = 3
        pad1=1
        pad2=1
        self.convIn = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=self.channel1,
                      kernel_size=self.kernel2,
                      padding=pad2),
            nn.BatchNorm2d(self.channel1),          
            nn.ReLU()
        )
        
        self.convB = nn.Sequential(
            nn.Conv2d(in_channels=self.channel1,
                      out_channels=self.channel1,
                      kernel_size=self.kernel1,
                      padding=pad1),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.channel1,
                      out_channels=self.channel2,
                      kernel_size=self.kernel2,
                      padding=pad2),
            nn.BatchNorm2d(self.channel2),
            nn.ReLU()           
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=self.channel2,
                      out_channels=self.channel2,
                      kernel_size=self.kernel1,
                      padding=pad1),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.channel2,
                      out_channels=self.channel2,
                      kernel_size=self.kernel1,
                      padding=pad1),
            nn.BatchNorm2d(self.channel2),
            nn.ReLU()           
            #nn.MaxPool2d(2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=self.channel2*2,
                      out_channels=self.channel2,
                      kernel_size=self.kernel1,
                      padding=pad1),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.channel2,
                      out_channels=self.channel1,
                      kernel_size=self.kernel1,
                      padding=pad1),
            nn.BatchNorm2d(self.channel1),
            nn.ReLU()
            #nn.MaxPool2d(2)
        )

        self.convOut = nn.Sequential(
            nn.Conv2d(in_channels=self.channel1*2,
                      out_channels=self.channel1,
                      kernel_size=self.kernel1,
                      padding=pad1),
            
            nn.BatchNorm2d(self.channel1),
            nn.ReLU(),
            #nn.MaxUnpool2d(2),            
            nn.Conv2d(in_channels=self.channel1,
                      out_channels=self.channel3,
                      kernel_size=self.kernel1,
                      padding=pad1),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.channel3,
                      out_channels=1,
                      kernel_size=self.kernel1,
                      padding=pad1),
            nn.ReLU()
        )

    def forward(self, input):
        
        input = input.unsqueeze(1)  # (batch, 1, freq, time)
        
        outputA = self.convIn(input)
        outputB = self.convB(outputA)
        output = self.conv1(outputB)
        output = self.conv2(torch.cat((outputB,output),1))
        output = self.convOut(torch.cat((outputA,output),1))

        return output[:,0,:,:]


model_CNN2D = CNN2D()
#print(model_CNN2D)

# define the optimizer
optimizer = optim.Adam(model_CNN2D.parameters(), lr=args.lr)
scheduler  = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.5)
scheduler.step()

if args.cuda:
    model_CNN2D = model_CNN2D.cuda()

if args.load_model:
    try:
        model_CNN2D.load_state_dict(torch.load(args.model_save))
        optimizer.load_state_dict(torch.load(args.optimizer_save))
        print('previous model loaded')
    except: 
        print('cannot find previous model')    

if args.load_loss:    
    try:
        training_loss=np.load('C:/Users/socia/OneDrive/Documents/COLUMBIA/Speech and Audio/project/2048_25/training_loss_log_norm3.npy').tolist()
        validation_loss=np.load('C:/Users/socia/OneDrive/Documents/COLUMBIA/Speech and Audio/project/2048_25/validation_loss_log_norm3.npy').tolist()
        print('previous loss loaded')
    except: 
        training_loss = []
        validation_loss = []
        print('cannot find previous loss')
else:
    training_loss = []
    validation_loss = []
        


# In[48]:


def MSE(input, output):
    # input shape: (batch, freq, time)
    # output shape: (batch, freq, time)
    
    batch_size = input.size(0)
    input = input.view(batch_size, -1)
    output = output.view(batch_size, -1)
    loss = (input - output).pow(2).sum(1)  # (batch_size, 1)
    return loss.mean()


from torch.utils.data import Dataset, DataLoader

class dataset_pipeline(Dataset):
    def __init__(self, path):
        super(dataset_pipeline, self).__init__()

        self.h5pyLoader = h5py.File(path, 'r')
        
        self.spec = self.h5pyLoader['spec_song']

        self.vocal = self.h5pyLoader['spec_vocal']
        
        self._len = self.spec.shape[0]  # number of utterances
    
    def __getitem__(self, index):
        spec_item = torch.from_numpy(self.spec[index].astype(np.float32))
        vocal_item = torch.from_numpy(self.vocal[index].astype(np.float32))    
        return [spec_item,vocal_item]
    
    def __len__(self):
        return self._len
    
# define data loaders
train_loader = DataLoader(dataset_pipeline(args.tr_data), 
                          batch_size=args.batch_size, 
                          shuffle=True, 
                          num_workers=0)

validation_loader = DataLoader(dataset_pipeline(args.val_data), 
                               batch_size=args.batch_size, 
                               shuffle=False, 
                               num_workers=0)

args.dataset_len = len(train_loader)
args.log_step = args.dataset_len // 4

#print(train_loader)


# In[50]:


def train(model, epoch, versatile=True, writer=None):
    start_time = time.time()
    model = model.train()  # set the model to training mode
    train_loss = 0.
    
    # load batch data
    for batch_idx, data in enumerate(train_loader):
        batch_spec = data[0]
        batch_vocal = data[1]

        if args.cuda:
            batch_spec = batch_spec.cuda()
            batch_vocal = batch_vocal.cuda()

        optimizer.zero_grad()
        
        spec_output = model(batch_spec)
        
        # MSE as objective
        loss = MSE(batch_vocal, spec_output)

        loss.backward()
        optimizer.step()
        
        train_loss += loss.data.item()
        
        if versatile:
            if (batch_idx+1) % args.log_step == 0:
                elapsed = time.time() - start_time
                print('| epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.2f} | MSE {:5.4f} |'.format(
                    epoch, batch_idx+1, len(train_loader),
                    elapsed * 1000 / (batch_idx+1), 
                    train_loss / (batch_idx+1)
                    ))
    
    train_loss /= (batch_idx+1)
    print('-' * 99)
    print('    | end of training epoch {:3d} | time: {:5.2f}s | MSE {:5.4f} |'.format(
            epoch, (time.time() - start_time), train_loss))
    return train_loss


def validate(model, epoch, writer=None):
    start_time = time.time()
    model = model.eval()
    validation_loss = 0.
    # load batch data
    for batch_idx, data in enumerate(validation_loader):
        batch_spec = data[0]
        batch_vocal = data[1]

        if args.cuda:
            batch_spec = batch_spec.cuda()
            batch_vocal = batch_vocal.cuda()
        
        with torch.no_grad():
        
            spec_output = model(batch_spec)
        
            # MSE as objective
            loss = MSE(batch_vocal, spec_output)
        
            validation_loss += loss.data.item()
    
    validation_loss /= (batch_idx+1)
    print('    | end of validation epoch {:3d} | time: {:5.2f}s | MSE {:5.4f} |'.format(
            epoch, (time.time() - start_time), validation_loss))
    print('-' * 99)
    
    return validation_loss




decay_cnt = 0
for epoch in range(1, args.epochs + 1):
    training_loss.append(train(model_CNN2D, epoch))
    validation_loss.append(validate(model_CNN2D, epoch))

    if training_loss[-1] == np.min(training_loss):
        with open(args.optimizer_save, 'wb') as g:
            torch.save(optimizer.state_dict(), g)
        with open(args.model_save, 'wb') as f:
            torch.save(model_CNN2D.state_dict(), f)
            np.save('C:/Users/socia/OneDrive/Documents/COLUMBIA/Speech and Audio/project/2048_25/training_loss_log_norm3',training_loss)
        print('      Best training model found and saved.')     
        if validation_loss[-1] == np.min(validation_loss):
            np.save('C:/Users/socia/OneDrive/Documents/COLUMBIA/Speech and Audio/project/2048_25/validation_loss_log_norm3',validation_loss)
            print('      Best validation model found.')

    print('-' * 99)
    decay_cnt += 1
    # lr decay
    # decay when no best training model is found for 3 consecutive epochs
    if np.min(training_loss) not in training_loss[-3:] and decay_cnt >= 3:
        scheduler.step()
        decay_cnt = 0
        print('      Learning rate decreased.')
        print('-' * 99)






