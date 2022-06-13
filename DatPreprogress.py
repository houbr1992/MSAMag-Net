# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 14:00:43 2022

@author: hbrhu
"""
import pickle
import numpy as np
import tensorflow as tf 

# def MuSigmaC(fd):
    
#     with open(fd,'rb') as f:
#         info = pickle.load(f)
        
#     mu = info['Params10_mu'].astype('float32')
#     sigma = info['Params10_std'].astype('float32')
    
#     mu0 = info['Params_mu'].astype('float32')
#     sigma0 = info['Params_std'].astype('float32')
    
#     mu[5,:,:] = mu0[5,:,:]
#     sigma[5,:,:] =sigma0[5,:,:]
#     zeros = np.zeros((295,3)).astype('float32')
    
#     C = info['Params10C'].astype('float32')
#     C[5,:,:] = zeros 
    
#     mu = np.transpose(mu, (0, 2, 1))
#     sigma = np.transpose(sigma, (0, 2, 1))
#     C = np.transpose(C, (0, 2, 1))
#     return mu, sigma, C 


def MuSigmaC(fd):
    
    with open(fd,'rb') as f:
        info = pickle.load(f)
        
    mu = info['Params10_mu'].astype('float32')
    sigma = info['Params10_std'].astype('float32')
    C = info['Params10C'].astype('float32')
    mu = np.transpose(mu, (0, 2, 1))
    sigma = np.transpose(sigma, (0, 2, 1))
    C = np.transpose(C, (0, 2, 1))
    
    return mu, sigma, C 


class TargetEmbedding():
    """
    Type0:
    Multi-Label
    
    Type1:
    Multi-classes
    
    Type2:
    Gaussian-Label
    
    Inp: (..., num)
    """
    def __init__(self, Type='Type2', len_tra = 1024, sigma = 0.5, dm = 0.01):
        super(TargetEmbedding, self).__init__()
        self.Type = Type 
        self.len_tra = len_tra 
        self.sigma = sigma 
        self.dm = dm 
        
    def Mul_Lab(self,x):
        # x = np.max(x, axis=-1)
        x = np.floor(x*10).astype(np.int32) 
        # BMX = np.array([ x for x in range(29,81,1)])
        BMX = np.array([ x for x in range(21,85,1)])
        BMX_Exp = np.ones(x.shape)[:,np.newaxis] * BMX
        x = np.less_equal(BMX_Exp,x[:,np.newaxis]) + 0. 
        # x = np.pad(x, ((0, 0), (0, 12)), constant_values = (0., 0.))
        return x 
    
    def Mul_Cls(self,x):
        # x = np.max(x, axis=-1)
        x = np.floor(x*10).astype(np.int32) 
        BMX = np.array([ x for x in range(29,81,1)])
        BMX_Exp = np.ones(x.shape)[:,np.newaxis] * BMX
        x0 = np.less_equal(BMX_Exp,x[:,np.newaxis]) + 0
        x1 = np.greater_equal(BMX_Exp,x[:,np.newaxis]) + 0
        # y = np.pad(x0*x1, ((0, 0), (0, 12)), constant_values = (0., 0.))
        return x0*x1
    
    def Gau_pdf(self,x):
        # x = np.max(x, axis=-1)
        k0 = 1./(np.sqrt(2*np.pi)*self.sigma)
        k1 = -0.5/self.sigma**2
        t = np.arange(0,(self.len_tra)*self.dm,self.dm)
        t_exp = np.ones((x[:,np.newaxis].shape)) * t 
        x = k0 * np.exp(k1 *(t_exp - x[:,np.newaxis])**2)
        x = x / np.max(x)
        return x 
    
    def __call__(self,x):
        
        if self.Type == 'Type0':
            x = self.Mul_Lab(x)
            
        elif self.Type == 'Type1':
            x = self.Mul_Cls(x)    
            
        elif self.Type == 'Type2':
            x = self.Gau_pdf(x) 
            
        return x 
    
class NormInpTar():
    
    """
    Norm Input Target 
    mu -> mean of value in different tw, params, com 
    sigma -> sigma of values in different tw, params, com 
    """
    def __init__(self, mu, sigma, C, Type = 'Type0'):      
        
        self.mu = mu[ np.newaxis, np.newaxis, :5, :, 5: ]
        self.sigma  = sigma[ np.newaxis, np.newaxis, :5, :, 5: ]
        self.C = C[ np.newaxis, np.newaxis, :5, :, 5: ]
        self.Type = Type
        self.TarEmb = TargetEmbedding(Type = 'Type0')
        
    def __call__(self, TraInp, R_Mat, S_info, Mag, t_ind):
        
        #Correct Parameters 10km 
        R = S_info[ :, :, -2 ]                           #R Value 
        R = R[:, :, np.newaxis, np.newaxis, np.newaxis]  #R Exp 
        R = np.log10(R + 1e-6)                           #log10       
        TraInp = TraInp[:, :, :5, :, :]                  #TraInp
        TraInp = TraInp + self.C * (1 - R)               #TraInp 
        
        #Correct Z-score Values 
        s0, s1, s2, s3, s4 = TraInp.shape
        TraInp = (TraInp - self.mu) / self.sigma
        TraInp = TraInp * t_ind[:, :, np.newaxis, np.newaxis, :]
 
        if self.Type == 'Type0':
            Inp = np.reshape(TraInp, (s0, s1, -1, s4, 1))
        
        if self.Type == 'Type1':
            Inp0 = np.reshape(TraInp, (s0, s1, -1, s4, 1)).astype('float32')
            Inp1 = np.log10(R_Mat + 1.)[:, :, :, np.newaxis]/np.log10(400. + 1.)
            Inp1 = Inp1.astype('float32')
            t_ind = t_ind.astype('float32')
            Inp = tf.tuple([Inp0, Inp1, t_ind])
            
        Tar = self.TarEmb(Mag) 
        Tar = Tar.astype('float32')
        return Inp, Tar 

class InpSelection():
    """
    Select Input Parameters
    T0 -> Pa, Pv, Pd
    T1 -> Pa, Pv, Pd, CAV
    T2 -> Pa, Pv, Pd, IV2 
    T3 -> Pa, Pv, Pd, IV2, CAV  
    """
    def __init__(self, Type='T0'):
        self.Type = Type 
        
    def __call__(self,Inp):
        """
        T0 -> Pa, Pv, Pd
        T1 -> Pa, Pv, Pd, CAV
        T2 -> Pa, Pv, Pd, IV2 
        T3 -> Pa, Pv, Pd, IV2, CAV 
        """
        x0, x1, x2 = Inp 
        if self.Type =='T0':
            x0 = x0[:,:,:9,:,:]
            
        elif self.Type == 'T1':
            x0 = tf.concat([x0[:,:,:9,:,:], x0[:,:,12:,:,:]], axis=2)
            
        elif self.Type == 'T2':
            x0 = x0[:,:,:12,:,:]
            
        Inp = tf.tuple([x0,x1,x2])
        
        return Inp 
