

import numpy as np
import torch
from torch import autograd
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.optim import lr_scheduler
from os import path
from os import system
from os import listdir
import copy
#from importlib import reload

import pickle
import scipy.io as sio
import time
from matplotlib import use
use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

# my modules
#import myfcn
import myfcn.annutils as fs
import myfcn.anntransforms as ts
#from myfcn.mylosses import batchMSELoss
# import default parameters
from visual_arguments import get_nondef, get_args

#-----------------------------
# define some auxiliary and highly specialised functions
def get_net_nb(savedir):
    outf = listdir(savedir)
    outf = [yy for yy in outf if '_net' in yy]
    if len(outf)==0:
        print('Surprise!')
        return 1
    else:
        net_nb = 0
        for yy in outf:
            netidx = yy.find('net')
            netidx += 3
            try:
                #print('File: ',ii)
                jj = int(yy[netidx:netidx+5])
                if jj>net_nb:
                    net_nb = jj
            except:
                1
        #print('net nb. ', net_nb+1)
        return net_nb + 1#-------------------------------------

def generate_correlation_map(x, y):
    """Correlate each n with each m.

    Parameters
    ----------
    x : np.array
      Shape N X T.

    y : np.array
      Shape M X T.

    Returns
    -------
    np.array
      N X M array in which each element is a correlation coefficient.

    """
    mu_x = x.mean(1)
    mu_y = y.mean(1)
    n = x.shape[1]
    if n != y.shape[1]:
        raise ValueError('x and y must ' +
                         'have the same number of timepoints.')
    s_x = x.std(1, ddof=n - 1)
    s_y = y.std(1, ddof=n - 1)
    cov = np.dot(x,
                 y.T) - n * np.dot(mu_x[:, np.newaxis],
                                  mu_y[np.newaxis, :])
    return cov / np.dot(s_x[:, np.newaxis], s_y[np.newaxis, :])

'''
Very first thing: parse arguments
'''
divider = '-'*70
print(divider)
print('Parsing arguments and setting parameters')
args,parser = get_args()
# store the nondefault ones in a dictionary
nondef = get_nondef(args,parser)


'''
Class and functions definition
'''

class iDataset(Dataset):
    def __init__(self, datatype = 'train_poisson', options = 'default', fname = None, realfname = None, transform=None):
    
        """
        Args:
            
            transform (callable, optional): Optional transform to be applied
                on a sample.
            datatype : whether to load the training or the test data; the real or the poisson 
                simulated data. Options are: 'train_poisson','test_poisson','train_real','test_real'
            options: number of neurons to consider, which repetition and which RF sampling strategy. 
                It is only necessary for real data 
        """
        
        self.transform = transform
        
        # check you received a file name to load the data from
        assert(fname is not None)
        assert(realfname is not None)

        if options == 'default':
            options = {'neurons': 200, 'BIG_FOV': False, 'irep': 0}
        if options['BIG_FOV']:
            extraline = '_expanding_'
        else:
            extraline = '_'

        # get the data as well as the mean and standard deviation over the training dataset
        # first load the stimulus (it's the same for poisson and real)
        mat = sio.loadmat(realfname)
        if 'train' in datatype:
            self.Y = np.array(mat['training_inputs']/2.55e-4, dtype = np.float)
        else:
            self.Y = np.array(mat['val_inputs']/2.55e-4, dtype = np.float)

        # now load the responses, real or simulated
        if 'real' in datatype:
            mat = sio.loadmat(fname)
            # load the traces
            trainX = np.array(mat['training_set'], dtype = np.float)
            # compute statistics on training dataset
            self.mu = np.mean(trainX, axis =0,keepdims = 1)
            self.sigma = np.std(trainX, axis =0,keepdims = 1)
            if 'train' in datatype:
                self.X = copy.deepcopy(trainX)
            else:
                self.X = np.array(mat['val_set'], dtype = np.float)
            # normalise
            self.X = (self.X - self.mu)/self.sigma
            self.meanOLE = 0.272 #linear decoder performance
        else:
            # now load the simulated responses
            mat = sio.loadmat(fname.format(extraline,options['neurons'],options['irep']))
            trainX = mat['R0'].T
            # remove non responsive
            mat['non_responsive'] = np.logical_not(np.squeeze(mat['non_responsive'].astype(bool)))
            #print(mat['non_responsive'][0:10])
            trainX = trainX[:, mat['non_responsive'][0:-1]]
            # compute statistics on training dataset
            self.mu = np.mean(trainX, axis =0,keepdims = 1)
            self.sigma = np.std(trainX, axis =0,keepdims = 1)
            if 'train' in datatype:
                self.X = trainX
            else:
                self.X = mat['Rtest0'].T
                self.X = self.X[:, mat['non_responsive'][0:-1]]
            # normalise responses
            self.X = (self.X - self.mu)/self.sigma
            self.meanOLE = np.mean(mat['Ctest0'][0][0])

        self.nsamples = self.Y.shape[0]
        self.nneurons = self.X.shape[1]
        print('In dataset init. N samples, n neurons: ', self.nsamples, self.nneurons)

    def __len__(self):
        return self.nsamples

    def __getitem__(self, idx):
        sample = {'Input': [], 'Target': []}
        sample['Input'] = self.X[idx] #copy.deepcopy(self.X[idx])
        sample['Target'] = self.Y[idx] #copy.deepcopy(self.Y[idx])
        # the sample is already normalised
        # transform the sample to tensor
        if self.transform:
            sample = self.transform(sample)
        return sample
    
'''
Define the ANN model
'''
def expand_args(x, base, pr_name = 'parameter to set'):
    # not sure I covered all cases
    if type(x) == type(base):
        if type(x) == list:
            if len(x) == len(base):
                return x
            elif len(x)==1:
                return [x[0]]*len(base)
            else:
                raise ValueError('Dimension of requested {} is ambiguous and not coherent with other parameters')
        elif type(x) == np.ndarray:
            if x.size == base.size:
                return x
            elif x.size == 1:
                return x + np.zeros_like(base)
            else:
                raise ValueError('Dimension of requested {} is ambiguous and not coherent with other parameters')
    elif type(x) == int or type(x)==float:
        if type(base) == list:
            return [x]*len(base)
        elif type(base) == np.ndarray:
            return x + np.zeros_like(base)
    elif type(x) == np.ndarray:
        if x.size == len(base):
            return x
        elif x.size == 1:
            return [x[0]]*len(base)

#add custom activation function, such as sigmoid-alpha. plus more functions if needed
class sigmoidalpha(nn.Module):
    def __init__(self,alpha):
        super(sigmoidalpha, self).__init__()
        self.alpha = alpha
        
    def forward(self,x):
        x = torch.sigmoid(self.alpha * x)
        return x

class tanhbeta(nn.Module):
    def __init__(self,beta):
        super(tanhbeta, self).__init__()
        self.beta = beta
    def forward(self,x):
        x = torch.tanh(self.beta*x)
        return x      

class identity(nn.Module):
    def __init__(self):
        super(identity, self).__init__()
    def forward(self,x):
        return x

activation_dict = {'relu': nn.ReLU(True), 'leakyrelu': nn.LeakyReLU(inplace=True), 'sigmoid': nn.Sigmoid(), 
                'hardtanh': nn.Hardtanh(inplace= True), 'tanh': nn.Tanh(), 'softsign': nn.Softsign(), 
                'selu': nn.SELU(True), 'prelu': nn.PReLU(), 'sigmoidalpha': sigmoidalpha(args.alpha), 
                'tanhbeta': tanhbeta(args.beta), 'identity': identity()}
                
class rec_fcnn1(nn.Module):
    '''
    Fully connected neural network.
    '''
    def __init__(self, pr, use_gpu = True):
        super(rec_fcnn1, self).__init__()

        self.use_gpu = use_gpu

        self.nlayers= len(pr['nchans'])+1
        in_chan = [pr['input_size']] + pr['nchans']
        out_chan = pr['nchans'] + [pr['output_size']]

        # expand dropout probabilities
        pr['p_dropout'] = expand_args(pr['p_dropout'], pr['nchans'], 'p dropout')

        # possibly: rewrite as list of layers and use activation function in the forward pass
        self.decoder = nn.Sequential()
        for ilayer in range(self.nlayers):
            # apply dropout is layer is not the first one and if required
            if ilayer>0 and pr['p_dropout'][ilayer-1]>0:
                name = 'dropout{}'.format(ilayer+1)
                self.decoder.add_module(name,nn.Dropout(p = pr['p_dropout'][ilayer-1]))
            name = 'FC{}'.format(ilayer+1)
            self.decoder.add_module(name, nn.Linear(in_chan[ilayer],out_chan[ilayer]))
            if pr['use_bn'] and ilayer<self.nlayers-1:
                name = 'BN{}'.format(ilayer+1)
                self.decoder.add_module(name, nn.BatchNorm1d(out_chan[ilayer]))
            name = 'activation{}'.format(ilayer+1)
            if ilayer<self.nlayers-1:
                self.decoder.add_module(name, activation_dict[pr['hid_activation']])
            else:
                self.decoder.add_module(name, activation_dict[pr['out_activation']])
            
    def forward(self,sample):
        z = self.decoder(sample)
        return z


class rec_dcnn1(nn.Module):
    '''
    In this template we have a deconvolutional neural network.
    '''
    def __init__(self, pr, use_gpu = True):
        super(rec_dcnn1, self).__init__()
        
        assert(len(pr['nchans']) == len(pr['ksize'])-1)
        # expand the dropout probability to the proper length
        pr['stride'] = expand_args(pr['stride'], pr['ksize'], 'stride')
        pr['p_dropout'] = expand_args(pr['p_dropout'], pr['ksize'], 'p dropout')
        self.use_gpu = use_gpu

        self.nlayers_fc = len([x for x in pr['ksize'] if x==0])
        self.nlayers = len(pr['ksize'])
        self.nlayers_conv = self.nlayers - self.nlayers_fc
        print(pr['p_dropout'],self.nlayers, self.nlayers_fc)
        in_chan = [pr['input_size']] + pr['nchans']
        out_chan = pr['nchans'] + [1] #the output is 1 channel - pixels are from HxW [pr['output_size']]

        self.H = int(np.sqrt(pr['output_size'])) # output image resolution

        #fully connected part
        self.decoder = nn.Sequential()
        for ilayer in range(self.nlayers_fc):
            # apply dropout is layer is not the first one and if required
            if ilayer>0 and pr['p_dropout'][ilayer-1]>0:
                name = 'dropout{}'.format(ilayer+1)
                self.decoder.add_module(name,nn.Dropout(p = pr['p_dropout'][ilayer-1]))
            name = 'FC{}'.format(ilayer+1)
            self.decoder.add_module(name, nn.Linear(in_chan[ilayer],out_chan[ilayer]))
            if pr['use_bn']:
                name = 'BN{}'.format(ilayer+1)
                self.decoder.add_module(name, nn.BatchNorm1d(out_chan[ilayer]))
            name = 'activation{}'.format(ilayer+1)
            if ilayer<self.nlayers-1:
                self.decoder.add_module(name, activation_dict[pr['hid_activation']])
            else:
                self.decoder.add_module(name, activation_dict[pr['out_activation']])

        # Deconvolutional part
        self.decoder_conv = nn.Sequential()
        for ilayer in range(self.nlayers_fc, self.nlayers):
            # apply dropout layer if requested
            if pr['p_dropout'][ilayer-1]>0:
                name = 'dropout{}'.format(ilayer+1)
                self.decoder_conv.add_module(name,nn.Dropout2d(p = pr['p_dropout'][ilayer-1]))
            # transposed convolutional layer
            name = 'Deconv{}'.format(ilayer+1)
            if ilayer<self.nlayers-1:
                if pr['stride'][ilayer]>1:
                    # if requested stride is more than 1
                    self.decoder_conv.add_module(name, nn.ConvTranspose2d(in_chan[ilayer],out_chan[ilayer], pr['ksize'][ilayer], padding = 0, stride = pr['stride'][ilayer]))
                else:
                    # if stride is 1, then upsample using bilinear interpolation
                    self.decoder_conv.add_module(name, nn.ConvTranspose2d(in_chan[ilayer],out_chan[ilayer], pr['ksize'][ilayer], padding = 0, stride = 1))
                    name = 'Upsample{}'.format(ilayer+1)
                    # hardcode a bilinear upsampling with scale factor 2
                    self.decoder_conv.add_module(name, nn.Upsample(scale_factor=2, mode= 'bilinear'))
                # batch norm layer if needed
                if pr['use_bn']:
                    name = 'BN{}'.format(ilayer+1)
                    self.decoder_conv.add_module(name, nn.BatchNorm2d(out_chan[ilayer]))
                name = 'activation{}'.format(ilayer+1)
                self.decoder_conv.add_module(name, activation_dict[pr['hid_activation']])
            else:
                # for the output layer always upsample using bilinear interpolation to get the desired output size
                # Also, do not use batch norm
                self.decoder_conv.add_module(name, nn.ConvTranspose2d(in_chan[ilayer],out_chan[ilayer], pr['ksize'][ilayer], padding = 0, stride = 1))
                name = 'Upsample{}'.format(ilayer+1)
                # hardcode a bilinear upsampling with scale factor 2
                self.decoder_conv.add_module(name, nn.Upsample(size = (self.H, self.H), mode = 'bilinear'))
                name = 'activation{}'.format(ilayer+1)
                self.decoder_conv.add_module(name, activation_dict[pr['out_activation']])
                    
    def forward(self, sample):
        # fully connected decoder
        z = self.decoder(sample)
        # reshape
        b_,n_ = z.size()
        z = z.view(b_,n_,1,1)
       
        # convolutional decoder
        z = self.decoder_conv(z)
        b_,n_,h_,w_ = z.size()
        assert(n_==1)
        z = z.view(b_,h_*w_)
        
        return z


class rec_dcnn2(nn.Module):
    '''
    In this template we have a deconvolutional neural network with more flexibility in the numbers of layers.
    '''
    def __init__(self, pr, use_gpu = True):
        super(rec_dcnn2, self).__init__()
        
        assert(len(pr['nchans']) == len(pr['ksize'])-1)
        # expand the dropout probability to the proper length
        pr['p_dropout'] = expand_args(pr['p_dropout'], pr['ksize'], 'p dropout')
        self.use_gpu = use_gpu

        self.nlayers_fc = len([x for x in pr['ksize'] if x==0])
        self.nlayers = len(pr['ksize'])
        self.nlayers_conv = self.nlayers - self.nlayers_fc
        in_chan = [pr['input_size']] + pr['nchans']
        out_chan = pr['nchans'] + [1] #the output is 1 channel - pixels are from HxW [pr['output_size']]
        assert(len(pr['upsample']) == self.nlayers_conv - 1)
        if isinstance(pr['stride'],int):
            pr['stride'] = expand_args(pr['stride'], pr['ksize'], 'stride')
        else:
            assert(len(pr['stride']) == self.nlayers_conv - 1)

        self.H = int(np.sqrt(pr['output_size'])) # output image resolution

        #fully connected part
        self.decoder = nn.Sequential()
        for ilayer in range(self.nlayers_fc):
            # apply dropout is layer is not the first one and if required
            if ilayer>0 and pr['p_dropout'][ilayer-1]>0:
                name = 'dropout{}'.format(ilayer+1)
                self.decoder.add_module(name,nn.Dropout(p = pr['p_dropout'][ilayer-1]))
            name = 'FC{}'.format(ilayer+1)
            self.decoder.add_module(name, nn.Linear(in_chan[ilayer],out_chan[ilayer]))
            if pr['use_bn']:
                name = 'BN{}'.format(ilayer+1)
                self.decoder.add_module(name, nn.BatchNorm1d(out_chan[ilayer]))
            name = 'activation{}'.format(ilayer+1)
            if ilayer<self.nlayers-1:
                self.decoder.add_module(name, activation_dict[pr['hid_activation']])
            else:
                self.decoder.add_module(name, activation_dict[pr['out_activation']])

        # Deconvolutional part
        self.decoder_conv = nn.Sequential()
        for ilayer in range(self.nlayers_fc, self.nlayers):
            ilayer_conv = ilayer - self.nlayers_fc
            # apply dropout layer if requested
            if pr['p_dropout'][ilayer-1]>0:
                name = 'dropout{}'.format(ilayer+1)
                self.decoder_conv.add_module(name,nn.Dropout2d(p = pr['p_dropout'][ilayer-1]))
            # transposed convolutional layer
            name = 'Deconv{}'.format(ilayer+1)
            if ilayer<self.nlayers-1:
                if pr['stride'][ilayer_conv]>1:
                    # if requested stride is more than 1
                    self.decoder_conv.add_module(name, nn.ConvTranspose2d(in_chan[ilayer],out_chan[ilayer], pr['ksize'][ilayer], padding = 0, stride = pr['stride'][ilayer]))
                else:
                    # if stride is 1, then upsample using bilinear interpolation
                    self.decoder_conv.add_module(name, nn.ConvTranspose2d(in_chan[ilayer],out_chan[ilayer], pr['ksize'][ilayer], padding = 0, stride = 1))
                    if pr['upsample'][ilayer_conv]:
                        name = 'Upsample{}'.format(ilayer+1)
                        # hardcode a bilinear upsampling with scale factor 2
                        self.decoder_conv.add_module(name, nn.Upsample(scale_factor=2, mode= 'bilinear'))
                # batch norm layer if needed
                if pr['use_bn']:
                    name = 'BN{}'.format(ilayer+1)
                    self.decoder_conv.add_module(name, nn.BatchNorm2d(out_chan[ilayer]))
                name = 'activation{}'.format(ilayer+1)
                self.decoder_conv.add_module(name, activation_dict[pr['hid_activation']])
            else:
                # for the output layer always upsample using bilinear interpolation to get the desired output size
                # Also, do not use batch norm
                self.decoder_conv.add_module(name, nn.ConvTranspose2d(in_chan[ilayer],out_chan[ilayer], pr['ksize'][ilayer], padding = 0, stride = 1))
                name = 'Upsample{}'.format(ilayer+1)
                # hardcode a bilinear upsampling with scale factor 2
                self.decoder_conv.add_module(name, nn.Upsample(size = (self.H, self.H), mode = 'bilinear'))
                name = 'activation{}'.format(ilayer+1)
                self.decoder_conv.add_module(name, activation_dict[pr['out_activation']])
                    
    def forward(self, sample):
        # fully connected decoder
        z = self.decoder(sample)
        # reshape
        b_,n_ = z.size()
        z = z.view(b_,n_,1,1)
       
        # convolutional decoder
        z = self.decoder_conv(z)
        b_,n_,h_,w_ = z.size()
        assert(n_==1)
        z = z.view(b_,h_*w_)
        
        return z



'''
TRAINING FUNCTION
'''
def train_model0(model, criterion, optimizer, scheduler, dsets, dataloaders, start_loss = None, 
                 use_gpu = True, dset_names = ['train','test'], num_epochs=1000, verbose = False):
    since = time.time()

    if not start_loss:
        start_loss = {}
        for phase in dset_names:
            start_loss['chance_'+phase] = 0.
            
    epochT = 100 # Evaluate the full test dataset every epochT epochs 
    best_loss = 10000000000.0 # starting point
    best_flag = False
    epoch_acc= 0.0
    # initialise epoch history
    epoch_loss_hist = {}
    for i in dset_names:
        epoch_loss_hist[i] = []
        epoch_loss_hist['square_'+i] = [] #{'train': [], 'test': [], 'square_train': [], 'square_test': []}
    
    since2 = time.time()

    # provide timing output during training
    if verbose:
        cstep = 25
    else:
        cstep = 1000000
    # printed accuracy output during training
    pstep = 20 if num_epochs>60 else 5

    # cycle through the epochs
    for epoch in range(num_epochs):
        if epoch%pstep == 0:
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

        # Each epoch has a training, validation and test phase
        for phase in dset_names:
            since2 = time.time()
            ii = 0
            torch.set_grad_enabled(phase=='train')
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
                #vol_flag = False               
            else:
                model.train(False)  # Set model to evaluate mode
                model.eval()
                #vol_flag = True
                
                '''
                If I want to evaluate the model only ever tot epochs
                '''
                flagskip = (phase == 'full_test') and not (epoch%epochT == 0)
                flagskip = flagskip and not epoch==num_epochs-1
                if flagskip:
                    break

            running_loss = 0.0 # average loss
            running_square = 0.0 # average square loss to compute the standard deviation
            divider = 0

            # Iterate over data.
            t00 = time.time()
            for data in dataloaders[phase]:
                
                # check that the batch sample is a dictionary
                assert isinstance(data,dict)
                ii += 1
                # get the inputs and targets
                inputs = data['Input']
                labels = data['Target']
                if phase == 'train':
                    inputs.requires_grad_()


                if use_gpu:
                    inputs = inputs.cuda()
                    labels = labels.cuda()

                # zero the parameter gradients
                optimizer.zero_grad()
                #model.zero_grad() # do I need this?

                # run the forward pass (it's ok also if data is none)
                preds = model(inputs) #This are all the predictions
                
                if args.loss == 'kldiv':
                    loss = criterion(torch.log(preds), labels)
                else:
                    loss = criterion(preds, labels) 
                
                # keep relevant statistics             
                if len(loss.shape)>0:
                    # If I haven't averaged over the batch already
                    running_loss += loss.sum().item()
                    running_square += torch.pow(loss,2).sum().item()
                    divider += loss.shape[0]
                    loss = loss.mean()
                else:
                    # the output is a single number
                    running_loss += loss.item()
                    running_square += torch.pow(loss,2).item()
                    divider += 1
                
                if ii%cstep == 0:
                    print('make predictions ', time.time() - t00, ii, ii%cstep)

                # backward + optimize only if in training phase
                if phase == 'train':
                    if ii%cstep == 0:
                        t00 = time.time()
                    loss.backward()
                    optimizer.step()
                    if ii%cstep == 0:
                        print('backward step', time.time() - t00)


            epoch_loss = running_loss / divider
            epoch_loss_hist[phase].append(epoch_loss)
            epoch_loss_hist['square_'+phase].append(running_square / divider)

            if epoch%pstep == 0:
                print('{} Loss: {:.4f} Chance: {:.4f}; Elapsed time (seconds): {:.4f}'.format(
                    phase, epoch_loss, start_loss['chance_'+phase], (time.time() - since2) ))
                #if verbose:
                #    os.system('nvidia-smi --query-gpu=memory.free,memory.used --format=csv')

            # deep copy the model
            if phase == 'val':
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_idx = len(epoch_loss_hist['val'])
                    best_model_wts = model.state_dict()
                    best_flag = True
                else:
                    best_flag = False
            elif phase == 'test' and best_flag:
                best_test_loss = epoch_loss
                best_flag = False
        
        # update time, you're about to go into the next epoch
        if epoch%pstep == 0:
            print()

    # END OF TRAINING
    time_elapsed = time.time() - since
    print('Training completed in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    try:
        print('Best val Loss: {:.6f}. Corresponding test Loss: {:.6f}'.format(best_loss,best_test_loss))
    except:
        print('wrong best val loss')

    # load best model weights
    if 'val' in dset_names:
        print('copying the best model')
        model.load_state_dict(best_model_wts)
        del best_model_wts
        epoch_loss_hist['best_val_loss'] =best_loss
        epoch_loss_hist['best_idx'] = best_idx
        epoch_loss_hist['best_test_loss'] = best_test_loss
        epoch_loss_hist['best_train_loss'] = epoch_loss_hist['train'][best_idx]
    else:
        epoch_loss_hist['best_idx'] = num_epochs
        epoch_loss_hist['best_test_loss'] = epoch_loss_hist['test'][-1]
        epoch_loss_hist['best_train_loss'] = epoch_loss_hist['train'][-1]
    
    return model, epoch_loss_hist, #, full_loss


'''
Main script starts
'''

'''
Some default parameters
'''
# store default parameters to run the Adv-Diffusions experiments
run_id = 0
datadir = '/media/sg6513/DATADRIVE2/PhD-stuff/'
realdata = 'Antolik_data/Data/region1/'
poissondata = 'Poisson_data/Data/region1/LinDec_simulation_res_store/'
expid = args.dtype
if expid == 'real':
    fname = datadir + realdata + 'full_data.mat'
    realfname = fname
else:
    fname = datadir + poissondata + 'all_sim_option21{}n{}_pt1_rep{:02d}.mat'
    realfname = datadir + realdata + 'full_data.mat'
saveroot = '/media/sg6513/DATADRIVE2/PhD-stuff/Res_ANN_pytorch/'
studyid = 'Trial2'
savedir = saveroot + expid + '/' + studyid + '/'
divider = '-'*70
inet = get_net_nb(savedir)

# dataset parameters
usememory = True

# model to use
models_avail = {'1': rec_fcnn1,'2': rec_dcnn1, '3': rec_dcnn2 }

# define transforms
train_tr = [ts.ToTensor()]
train_tr_save=['ts.ToTensor()']
test_tr = [ts.ToTensor()]
test_tr_save=['ts.ToTensor()']
data_transforms = {
    'train': transforms.Compose(train_tr),
    'val': transforms.Compose(train_tr),
    'test': transforms.Compose(test_tr)
}
print('Building the dataset')
# define options as from command line
options = {'neurons': args.neurons, 'BIG_FOV': args.bigfov, 'irep': args.prep}
# DEFINE AND TEST THE DATASET
t0 = time.time()
ds = iDataset(datatype = 'train_'+args.dtype, options = options, transform = data_transforms['train'], fname = fname, realfname = realfname)
print('Dataset creation time and input type. ',time.time()-t0, type(ds[0]['Input']))

print('Initialising the model')
# build the parameters dictionary from the command line arguments
if args.central:
    nPixels = 16*16
else:
    nPixels = 961
if args.dtype == 'real':
    neurons = 103
    args.neurons = 103
else:
    neurons = ds.nneurons #args.neurons
pr = {'nchans': args.nchans, 'ksize': args.ksize, 'input_size': neurons, 'output_size': nPixels, 'p_dropout': args.p_dropout, 'use_bn': args.use_bn, 'hid_activation': args.hid_act, 'out_activation': args.out_act, 'stride': args.stride, 'upsample': args.upsample}


# Create the two datasets for training and testing
datatypes = { x: x + '_' + args.dtype for x in ['train','test']}
#indices = {'train': range(15000), 'test': range(15000,20518)}
ds = {x : iDataset(datatype = datatypes[x], options = options, fname = fname, realfname = realfname, transform = data_transforms[x]) for x in ['train','test']}
stop

# Create the two dataloaders
dataloaders = {x: torch.utils.data.DataLoader(ds[x], batch_size=args.batch, shuffle=True) 
               for x in ['train','test']}


print(divider)
print('Rebuilding the model')
# define model again
if args.use_gpu:
    model = models_avail[args.model](pr,use_gpu = args.use_gpu).cuda()
else:
    model = models_avail[args.model](pr, use_gpu = False)
if args.verbose:
    print(model)
    print('-'*20)
    system('nvidia-smi --query-gpu=memory.free,memory.used --format=csv')

print(divider)
print('Setting up optimiser and loss function')
# decide optimiser and cost function
if args.optim == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=args.lr, eps=1e-08, weight_decay=args.wdecay)
else:
    print('Other optimisers are not implemented, using Adam for now')
    optimizer = optim.Adam(model.parameters(), lr=args.lr, eps=1e-08, weight_decay=args.wdecay)
# Decay LR by a factor of 0.1 every Tot epochs
if args.lr_decay:
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lrds, gamma=args.lrdg)
# decide loss function
losses_dict = {'mse': nn.MSELoss(), 'binaryce': nn.BCELoss(), 'l1': nn.L1Loss(), 'kldiv': nn.KLDivLoss()}
loss_function = losses_dict[args.loss]

print(divider)
print(loss_function)
print('Starting the training')

# train the model
out = train_model0(model, loss_function, optimizer, exp_lr_scheduler, ds, dataloaders, 
                   start_loss = None, use_gpu = args.use_gpu, dset_names = ['train','test'], 
                   num_epochs=args.nepochs, verbose = False)
model = out[0]
loss_hist = out[1]

print(divider)
print('dtype {} and nb of neurons {}'.format(args.dtype,neurons))
print('Computing some stats for the results')
# get all the predictions
dataloaders = torch.utils.data.DataLoader(ds['test'], batch_size=50, shuffle=False)
data = next(iter(dataloaders))
if args.use_gpu:
    for k in data.keys():
        data[k] = data[k].cuda()
preds = model(data['Input']).cpu().detach().numpy()
labels = data['Target'].cpu().detach().numpy()
# compute MSE
MSE = np.mean((preds - labels)**2, axis = 1)
CORR = np.diag(generate_correlation_map(preds, labels))

print(divider)
print('Saving some example predictions')
f = plt.figure(figsize = (10,10))
for ipred in range(4):
    f.add_subplot(2,4,2*ipred+1)
    plt.imshow(np.reshape(preds[ipred,:],(31,31)), vmin = 0, vmax = 1, interpolation='none')
    f.add_subplot(2,4,2*ipred+2)
    plt.imshow(np.reshape(labels[ipred,:],(31,31)), vmin = 0, vmax = 1, interpolation = 'none')
    plt.axis('off')
plt.savefig(savedir + 'expreds_nbnet{:05d}.svg'.format(inet))
plt.close()

print(divider)
print('Saving the results')

sio.savemat(savedir + 'somedata_net{:05d}.mat'.format(inet),{'preds':preds, 'MSE':MSE, 'CORR':CORR, 'labels': labels})

nparams = sum(p.numel() for p in model.parameters() if p.requires_grad)

summary = {'net': str(model), 'train loss': loss_hist['best_train_loss'], 'test loss': loss_hist['best_test_loss'], 'val_loss': ['na'], 'train chance loss': ['na'], 'test chance loss': ['na'], 'val chance loss': ['nan'], 'train transforms': [train_tr_save], 'test transforms': [test_tr_save], 'normalise': '0 mean, 1 std - precomputed', 'net_name': [inet], 'neurons': [args.neurons], 'n pixels': [nPixels], 'avg MSE': [MSE.mean()], 'avg rho': [CORR.mean()], 'n params': [nparams], 'mean OLE': [ds['train'].meanOLE] }
for k in vars(args).keys():
    summary[k] = [vars(args)[k]]

fs.savemodel2pd(savedir + 'nets_full_summary.csv', summary)

# things needed to initialise the model:
# save trained model and things you need to create it again
with open(savedir + 'models/net{:05d}'.format(inet) + '_loadwith.pickle','wb') as f:
    pickle.dump({'models_avail':models_avail,'which':args.model,'pr':pr, 'use_batch_first':True}, f)
torch.save(model.state_dict(), savedir + 'models/net{:05d}_model.pt'.format(inet))


