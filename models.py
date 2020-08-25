import torch
import torch.nn as nn
from torch.backends import cudnn
from torch.autograd import Variable

from os import path, makedirs, listdir
from zoo.models import *

import numpy as np
np.random.seed(1)
import random
random.seed(1)

class XViewFirstPlaceLocModel(nn.Module):
    def __init__(self, model_size, models_folder='weights', devices=[0,0,0]):
        super(XViewFirstPlaceLocModel, self).__init__()
        self.models = []
        self.model_dict = {
            '34':Res34_Unet_Loc,
            '50':SeResNext50_Unet_Loc,
            '92':Dpn92_Unet_Loc,
            '154':SeNet154_Unet_Loc,
            
        }
        self.checkpoint_dict = {
            '34':'res34_loc_{}_1_best',
            '50':'res50_loc_{}_tuned_best',
            '92':'dpn92_loc_{}_tuned_best',
            '154':'se154_loc_{}_1_best',
            
        }
        self.pred_folder = f'pred{model_size}_loc'
        for ii, seed in enumerate([0, 1, 2]):
            snap_to_load = self.checkpoint_dict[model_size].replace('{}',str(seed))
            model = self.model_dict[model_size]()
            model.to(f'cuda:{devices[ii]}')
            print("=> loading checkpoint '{}'".format(snap_to_load))
            checkpoint = torch.load(path.join(models_folder, snap_to_load), map_location='cpu')
            loaded_dict = checkpoint['state_dict']
            sd = model.state_dict()
            for k in model.state_dict():
                if k in loaded_dict and sd[k].size() == loaded_dict[k].size():
                    sd[k] = loaded_dict[k]
            loaded_dict = sd
            model.load_state_dict(loaded_dict)
            print("loaded checkpoint '{}' (epoch {}, best_score {})"
                    .format(snap_to_load, checkpoint['epoch'], checkpoint['best_score']))
            model.eval()
            self.models.append(model)
            
    def execute_model(self, x, model):
        model_device = next(model.parameters()).device # Hack to get device
        inp = Variable(x).to(model_device)
        msk = model(inp)
        return msk
        
            
    def forward(self,x):
        msk_out = []
        x_shape = x.shape
        # Because this model actually executes something along the batch dimension, compress
        # the batch dimension, then uncompress at the end
        x = x.reshape([-1]+list(x.shape[-3:]))
        msk0 = self.execute_model(x, self.models[0]).cpu()
        msk1 = self.execute_model(x, self.models[1]).cpu()
        msk2 = self.execute_model(x, self.models[2]).cpu()
                     
        # Separating back into correct batch size for first dim
        new_shape = [x_shape[0],-1] + list(msk0.shape[1:])
        msk0 = msk0.reshape(new_shape).numpy()
        msk1 = msk1.reshape(new_shape).numpy()
        msk2 = msk2.reshape(new_shape).numpy()
                
        for i in range(msk0.shape[0]):
            pred = []
            for msk in [msk0, msk1, msk2]:                    
                pred.append(msk[i][0, ...])
                pred.append(msk[i][1, :, ::-1, :])
                pred.append(msk[i][2, :, :, ::-1])
                pred.append(msk[i][3, :, ::-1, ::-1])

            pred_full = np.asarray(pred).mean(axis=0) * 255
            msk_out.append(torch.tensor(pred_full.astype('uint8').transpose(1, 2, 0)).squeeze())
        msk_out = torch.stack(msk_out)
         
        return msk_out
            
            
            