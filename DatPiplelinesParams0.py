# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 23:44:28 2022

@author: hbrhu
"""

import os, time, pickle, math, random  
import numpy as np 
import tensorflow as tf 
from itertools import chain 
from tqdm import tqdm 


def DatShu(x):
    x = [ list(i) for i in x ] 
    x0, x1, x2, x3, x4 = x 
    combine = list(zip(x0, x1, x2, x3, x4))
    random.shuffle(combine)
    return [ np.array(x).astype('float32') for x in zip(*combine)]

def MuSigma(files):
    """
    Input:files 
    Output: [ Pa, Pv, Pd, IV2, CAV, tc ] 
    Mu,
    Sigma,
    """
    with open(files, 'rb') as f:
        dict_ = pickle.load(f)
        
    mu = dict_['Params_mu']
    sigma = dict_['Params_std']
    
    mu = np.transpose(mu, (0, 2, 1))
    sigma = np.transpose(sigma, (0, 2, 1))
    
    return mu[:, :, 5:], sigma[:, :, 5:] 

def TVDatset(ttv_dic, btc_sz = 1, dir0 = '../Params', TypeName='_Params_Type1'):
    """
    Generate Test and Validation Datapiplelines 
    Input:
    ttv_dic, Test/Validation dictionary 
    btc_sz, 1 
    dir0, Params
    TypeName, Params_Type1
    Output:
    TesDat, ValDat 
    """
    test_lis = []
    for i in range(7):
        test_sin = ttv_dic['Mag{}'.format(i)]['Test']
        test_lis += test_sin

    valid_lis = []
    for i in range(7):
        valid_sin = ttv_dic['Mag{}'.format(i)]['Valid']
        valid_lis += valid_sin
    
    random.shuffle(test_lis) 
    random.shuffle(valid_lis)
    
    test_flis =[ os.sep.join([dir0, x[0:4], x + TypeName]) for x in test_lis ]
    valid_flis = [ os.sep.join([dir0, x[0:4], x + TypeName]) for x in valid_lis ]
    
    TesDat = tf.data.Dataset.from_tensor_slices(test_flis).repeat(1).batch(btc_sz)
    ValDat = tf.data.Dataset.from_tensor_slices(valid_flis).repeat(1).batch(btc_sz)
    return TesDat, ValDat

def TDatset(ttv_dic, btc_sz = 1, training = False, dir0 = '../Params', TypeName='_Params_Type1'):
    """
    Generate Test and Validation Datapiplelines 
    Input:
    ttv_dic, Train dictionary 
    btc_sz, 1 
    dir0, Params
    TypeName, Params_Type1
    Output:
    TraDat 
    """
    train_lis = []
    for i in range(4):
        train_sin = ttv_dic['Mag{}'.format(i)]['Train']
        train_lis += train_sin 
    if training:
        train_lis += random.sample(ttv_dic['Mag{}'.format(4)]['Train'], 150)
        train_lis += random.sample(ttv_dic['Mag{}'.format(5)]['Train'], 400)
        train_lis += random.sample(ttv_dic['Mag{}'.format(6)]['Train'], 400)
    else: 
        train_lis += ttv_dic['Mag{}'.format(4)]['Train']
        train_lis += ttv_dic['Mag{}'.format(5)]['Train']
        train_lis += ttv_dic['Mag{}'.format(6)]['Train']
    # 200 200  
    random.shuffle(train_lis)
    
    train_flis =[ os.sep.join([dir0, x[0:4], x + TypeName]) for x in train_lis ]
    
    TraDat = tf.data.Dataset.from_tensor_slices(train_flis).repeat(1).batch(btc_sz)
    return TraDat

class DatPipleParams():
    
    def __init__(self, num0 = 32, 
                 indl = [ 128, 128, 64, 64, 48, 48, 48 ], 
                 Ratl = [ 16, 11, 10, 2, 1, 1, 1], 
                 tri_num =3, eps = 0., 
                 training = False ):
        
        super(DatPipleParams, self).__init__()
        self.num0 = num0 
        self.indl = indl 
        self.Ratl = Ratl 
        self.tri_num = tri_num 
        self.eps = eps 
        self.training = training
        self.MagSeg = np.array([0, 40, 45, 50, 55, 60, 65])
        
    def OpenFile(self, x):
        """
        Read Dictionary Files 
        Input:
        files dirs 
        Output:
        EventInfo, list -> [ Qname, Depth, Mag, Qlat, Qlon ]
        StaNames, list 
        StaInfo, array -> [ p-time, lat, lon, R ] -> (len_sta, 4)
        ParaInfo, array -> [ Pa, Pv, Pd, IV2, CAV, tc ] -> (len_sta, 6, component, len_t)
        """
        
        with open(x.decode('utf-8'), 'rb') as f:
            dic = pickle.load(f)
        
        Qname, Depth, Mag = dic['Qname'], dic['Depth'], dic['Mag']
        Qlat, Qlon, names = dic['Qlat'], dic['Qlon'], dic['Stations']
        lat, lon, R, pt = dic['lat'], dic['lon'], dic['R'], dic['pt']
        Pa, Pv, Pd = dic['Pa'], dic['Pv'], dic['Pd']
        IV2, CAV, tc = dic['IV2'], dic['CAV'], dic['tc']
        
        #location info [ lat, lon, R, pt ]  
        loc_info = np.array([pt, lat, lon, R])
        loc_info = np.transpose(loc_info, (1,0))
        
        #Params info 
        Params_info = np.array([Pa, Pv, Pd, IV2, CAV, tc])
        Params_info = np.transpose(Params_info, (1, 0, 3,2))
        
        #R_Mat
        R_Mat = dic['R_Mat'] 
        
        return [[Qname, Depth, Mag, Qlat, Qlon ], names, loc_info, Params_info, R_Mat ] 
    
    def MulParams(self, Q_info, names, S_info, Params, R_Mat, t0):
        """ 
        Input, 
        Out: 
        ParaInfo, array -> [ Pa, Pv, Pd, IV2, CAV, tc ] -> (len_sta, 6, component, len_t)
        R_Mat, array(len_sta,)
        t_index, shape
        S_info, 
        Mag, Qlat, Q_lon, Depth, Ori_Time
        """
        s0, s1, s2, s3 = Params.shape
        pt = S_info[:, 0][:, np.newaxis]                                       #Pwave loc based Ori_Time 
        t_mat = ((np.min(pt)) + t0 - pt) * 10                                  #Stations trigger t agter the 1st Trigger t sec 
        #Different Stations trigger stages after 1st Trigger t time 
        t_ind = np.less_equal(np.ones((s0, 1)) * np.arange(1, 301), t_mat) + 0.
        #Station info after 1 st trigger t sec 
        S_info0 = np.max(t_ind, axis = 1, keepdims= True) *  np.concatenate([S_info, t_mat/10], axis = 1) 
        #Params  Pa, Pv, Pd, IV2, CAV, tc
        Params = Params[:, :, :, 5:] * t_ind[:, np.newaxis, np.newaxis,:]
        names = [ names[i] for i in range(len(names)) if t_mat[i] > 0 ]
        
        if s0 < self.num0:
            s = self.num0 - s0 
            Params = np.pad(Params, ((0,s),(0,0),(0,0),(0,0)), constant_values = (0., 0.))
            t_ind = np.pad(t_ind, ((0, s),(0,0)), constant_values = (0., 0.))
            S_info0 = np.pad(S_info0, ((0, s),(0,0)), constant_values = (0., 0.))
            R_Mat = np.pad(R_Mat, ((0, s),(0, s)), constant_values =(0., 0.))
        
        return Params, R_Mat, t_ind, S_info0, Q_info[2], Q_info[3], Q_info[4], Q_info[1], Q_info[0]
    
    def ObtInd(self, x):
        x = int(x * 10)
        ind = 7 - np.sum(np.greater_equal(x, self.MagSeg) + 0 )
        return int(ind)
    
    def trigger_num(self, x):
        bools = False 
        if len(x) > 0:
            x = np.max(x, axis = 1 )
            x = np.sum(x)
            if x >= self.tri_num:
                bools = True 
        return bools 
    
    def zipfunc(self, x):
        """
        zip function 
        Input:
        names, Params, S_info, R_Mat
        Output:
        Zip information 
        """
        x = [ list(Info) for Info in x ]
        x0, x1, x2, x3 = x 
        x = list(zip(x0, x1, x2, x3))
        return x 
    
    def unzipfunc(self, x):
        """
        unzip function
        Input:
        zip infomation 
        output:
        names, Params, S_info, R_Mat   
        """
        x = sorted(x, key=lambda x:(float(x[2][0])))
        x = [ t for t in zip(*x)]
        x0 = list(x[0])
        x1, x2 = [ np.array(info).astype('float32') for info in x[1:3]]
        x3 = np.array(x[3])[:self.num0, :self.num0].astype('float32')
        return [ x0, x1, x2, x3 ]
    
    def __call__(self, files, t0 ):
        
        eve_lis = []
        for f in files:
            Q_info, names, S_info, Params, R_Mat = self.OpenFile(f)
            Mag = Q_info[2]
            if self.training:
                MagSeg = self.ObtInd(Mag) 
                ind = self.indl[MagSeg]
                names = names[:ind]
                S_info = S_info[:ind, :]
                Params = Params[:ind, :, :, :]
                R_Mat = R_Mat[:ind, :ind]
                zipinfo = self.zipfunc([names, Params, S_info, R_Mat])
                
                #先进行重采样
                for r in range(self.Ratl[MagSeg]):
                    # if self.Ratl[MagSeg] >1:
                        # print(self.Ratl[MagSeg])
                    if len(names) >= self.num0:
                        zipinfo0 = random.sample(zipinfo, self.num0)
                    else:
                        zipinfo0 = zipinfo
                
                    names0, Params0, S_info0, R_Mat0 = self.unzipfunc(zipinfo0)
                    R_Mat0 = R_Mat[:self.num0, :self.num0]
                    sin_eve = [ Q_info, names0, S_info0, Params0, R_Mat0 ]
                    eve_lis.append(sin_eve)
            else:
                sin_eve = [ Q_info, names[:self.num0], S_info[:self.num0, :], Params[:self.num0, :, :, :], R_Mat[:self.num0, :self.num0] ]
                eve_lis.append(sin_eve)
        #random event list 
        random.shuffle(eve_lis)
        # earth_inp = [ [], [], [], [], [], [], [], [], []]
        if len(eve_lis) > 0:
            #构建数据流
            earth_inp = [ [] for i in range(9)]
            earth_ = [ self.MulParams(x[0], x[1], x[2], x[3], x[4], t0) for x in eve_lis ]
            for x in earth_:
                if self.trigger_num(x[2]):
                    earth_inp = [ earth_inp[i] + [x[i]] for i in range(9) ]
                               
            if earth_inp == []:
                earth_inp = [ [] for i in range(9) ]
        else:
            
            earth_inp = [ [] for i in range(9)]
            
        """
        Params, [ Pa, Pv, Pd, IV2, CAV, tc ] 
        t_ind, 
        S_info, [ p-time, lat, lon, R, ptime ]
        Mag,
        Qlat, 
        Qlon,
        Depth,
        Qname 
        """
        return earth_inp 