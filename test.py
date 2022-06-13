# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 22:00:38 2022
@author: hbrhu
"""

import os, pickle, math  
import numpy as np 
import pandas as pd 
from itertools import chain 
import tensorflow as tf
from DatPiplelinesParams0 import DatPipleParams, TDatset, TVDatset, DatShu, MuSigma
from DatPreprogress import NormInpTar, MuSigmaC, InpSelection
from models import MagLoss, Evaluations, EvaParmSave, MSAM
from plot_func import PlotFigLab, PlotFigPdf, PltResRT, SpMag
from LRDecays import CustSchdle

def ObtDF(x):
    df_ = pd.DataFrame(x, index = [1])
    return df_
    
def Dict_(x):
    dic = {
        'event':x[0],
        'Qlat':x[1],
        'Qlon':x[2],
        'Depth':x[3],
        'Mag':x[4],
        'PreMlis':[np.array(x[5])],
        'tlis':[np.array(x[6])],
        'trigger_num':[np.array(x[7])],
        'Predicts':[x[-1]],
        }
    return dic 


#loss weights  
files_wei = './loss_weight.pkl'
with open(files_wei, 'rb') as f:
    wei_dic = pickle.load(f)
loss_wei = wei_dic['loss_weights']                                             #loss weights 

#Distance corrections mu. sigma 
InpTarName = './Info30kmWeiR2.pkl'                                             #mu, sigma, C, dirs 
mu, sigma, C = MuSigmaC(InpTarName)                                            #mu, sigma, C 

"""Model Initalizer"""
num_ly0 = 6                                                                    #Encoder layers numbers 
num_th = 66                                                                    #num_th chechkpoint
thre = 0.80                                                                    #threshold
InpType = 'T0'                                                                 #NetType 
warmup_steps = 1024                                                            #warm_up steps
init_lr = 0.01                                                                 #max learing rate 
lr_min = 1e-5                                                                  #learning rate 
lr = CustSchdle(init_lr, lr_min, warmup_steps)                                 #warmup learning rate 
kr_sz0, strides0 =[(3,1), (3,1)], [(3,1), (3,1)]                               #kernel-size, stride-size 
max2keep = 200                                                                 #max keep step 
training = False                                                               #Training State 
Shuffle = True                                                                 #Input Data Shuffle 
checkname = 'checkp.ckpt'                                                      #model name 
LossType = 'Type0'                                                             #Loss Parameters
optimizer = tf.keras.optimizers.Adam(lr, beta_1=0.9, beta_2=0.98, 
                                     clipnorm=1., clipvalue=0.5)               #Optimizer initializers      
model = MSAM(num_ly0, kr_sz0=kr_sz0, strides0=strides0)                        #model initializers 

checkpoint_path = './Mag_models_Att_lays_{}_Inp_{}'.format(num_ly0, InpType)   #checkpoint_path 
if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)
ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, checkpoint_name=checkname, max_to_keep=max2keep)


"""DataPiplelines Progressing"""
btc_sz_eve = 1                                                                 #batch size for event 
num0 = 32                                                                      #Input numbers
tri_num = 3                                                                    #trigger number
t_lis = list([ t for t in range(1, 31)])                                       #index P-arrival time list   

#files read  
dir_ttv = './Data/event'                                                       #Data
Datlis = list(chain(*[[os.sep.join([dir_ttv,d,x]) for x in os.listdir(os.sep.join([dir_ttv,d]))] for d in os.listdir(dir_ttv)]))


"""DataPiplelines"""
Dat0 = tf.data.Dataset.from_tensor_slices(Datlis).repeat(1).batch(btc_sz_eve)
Piplelines1 = DatPipleParams(num0, training = False)                           #UnTraining Datapiplelines 
InpTar = NormInpTar(mu, sigma, C, 'Type1')                                     #InpType 
InpSle = InpSelection(InpType)                                                 #Input Slection

check_name='./{}/{}-{}'.format(checkpoint_path, checkname, num_th)
ckpt.restore(check_name)
file0 = './Exms_lay_{}_Inp_{}/Examples_MSAM_lay_{}_Inp_{}_{}th_Vers0.pkl'.format(num_ly0, InpType,
                                                                         num_ly0, InpType, num_th)     #Training num_th 

df0 = pd.DataFrame()                                                       #Traning DataFrame

for f0 in Dat0:
    event = f0.numpy()[0].decode('utf-8').split('\\')[-1].split('_')[0]
    PreMl, tl, numl, Prelis = [], [], [], []
    for t0 in t_lis:
        IO = Piplelines1(f0.numpy(), t0)
    
        if len(IO[0]) > 0:
            _, _, _, _, _, Qlat, Qlon, Depth, event = IO
            TraInp, R_Mat, t_ind, S_info, Mag =[ np.array(IO[i]) for i in range(5) ]        
            TraInp, R_Mat, t_ind, S_info = [TraInp, R_Mat, t_ind, S_info]
            Inp,Tar = InpTar(TraInp, R_Mat, S_info, Mag, t_ind )
            Inp = InpSle(Inp)
            Pre, _ = model(Inp[0], Inp[1], Inp[2])
            Pre = Pre[0]
            
            t_sec = np.sum(t_ind, axis=-1)[0]*0.1
            tri_num = np.sum(np.max(t_ind, axis=-1))
            PreMag = np.sum(np.greater_equal(Pre, thre) + 0.)*0.1 + 2.
            
            PreMl.append(PreMag)
            tl.append(t_sec)
            numl.append(tri_num)
            Prelis.append(Pre.numpy())
        else:
            t_sec = np.zeros((32))
            tri_num = 0
            PreMag = -1. 
            Pre = np.zeros((64))
            
            PreMl.append(PreMag)
            tl.append(t_sec)
            numl.append(tri_num)
            Prelis.append(Pre)

    if np.max(PreMl) > -1:
        Qlat, Qlon, Depth, Mag = Qlat[0], Qlon[0], Depth[0], Mag[0]
        eve_inf = [event[0], Qlat, Qlon, Depth, Mag, PreMl, tl, numl, np.array(Prelis)]
        df0 = df0.append(ObtDF(Dict_(eve_inf)), ignore_index=True)   
        val = SpMag(Mag, PreMl)
        if np.max(Pre) >= thre:
            if val <= 0.5:
                Cls='Pos/level1'
            elif val <= 1.0 and val>0.5:
                Cls='Pos/level2'
            else:
                Cls='Pos/level3'
            dir_ = './Exms_lay_{}_Inp_{}/check_{}_th/Results/{}/'.format(num_ly0, InpType, num_th, Cls)
            if not os.path.exists(dir_):
                os.makedirs(dir_)
            PltResRT(PreMl, Mag, numl, t_lis, event, Depth, dir_)
        else:
            Cls = 'Neg'
            dir_ = './Exms_lay_{}_Inp_{}/check_{}_th/Results/{}/'.format(num_ly0, InpType, num_th, Cls)
            if not os.path.exists(dir_):
                os.makedirs(dir_)
            PltResRT(PreMl, Mag, numl, t_lis, event, Depth, dir_)

num = np.arange(len(df0))
df0['num'] = num 
df0.set_index('num', inplace=True)
df0.to_pickle(file0)

