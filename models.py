# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 12:39:56 2022
@author: hbrhu
"""

import pickle 
import numpy as np 
import tensorflow as tf 
from tensorflow.keras.layers import Conv1D, Conv2D, MaxPool2D, Conv2DTranspose, Dense, Flatten, GlobalMaxPooling1D
from tensorflow.keras.layers import ReLU, LeakyReLU, MultiHeadAttention,  LayerNormalization

def create_mask(x):
    
    mask = tf.reduce_max(x, axis = -1)
    mask0 = mask[ :, tf.newaxis, : ]
    mask1 = mask[ :, tf.newaxis, : ]
    
    return mask0, mask1 
    
class ParamsNet(tf.keras.layers.Layer):
    """
    filters0 -> [ 2, 4], 
    kr_sz0 -> [ (3,1), (6,1) ], 
    strides0 -> [ (3,1), (6,1) ],
    filter1 -> [5, 4, 4, 4 ],
    kr_sz1 -> [5, 5, 5, 5 ],
    strides1 -> [5, 4, 4, 4 ],
    inp_shp0 -> [ 1, 2, 3, 4, 5 ],
    inp_shp1 -> [ 1, 2, 3, 4 ],
    """
    def __init__(self, 
                 filters0 = [ 2, 4], 
                 kr_sz0 = [ (3,1), (3,1) ], 
                 strides0 = [ (3,1), (3,1) ],
                 filters1 = [ 8, 16, 32, 64],
                 kr_sz1 = [5, 5, 5, 5 ],
                 strides1 = [5, 4, 4, 4 ],
                 inp_shp0 = [ 1, 2, 3, 4, 5 ],
                 inp_shp1 = [ 1, 2, 3, 4 ]):
           
        super(ParamsNet, self).__init__()
        
        self.CNN_BK0 = [ Conv2D(filters0[i], kr_sz0[i], strides0[i], padding='same', 
                                kernel_initializer='glorot_uniform', bias_initializer= 'zeros',
                                input_shape = inp_shp0[2:]) 
                        for i in range(len(filters0))]
        self.CNN_BK1 = [ Conv1D(filters1[i], kr_sz1[i], strides1[i], padding='same', 
                                kernel_initializer='glorot_uniform', bias_initializer='zeros',
                                input_shape=inp_shp1[2:]) 
                        for i in range(len(filters1))]
        self.act0 = ReLU()        
        self.act1 = LeakyReLU( alpha = 0.05 )
        
    def __call__(self, x):
        
        for CNN in self.CNN_BK0:
            x = CNN(x)
            x = tf.keras.activations.gelu(x)
        x = tf.squeeze(x, axis=2)
        
        for CNN in self.CNN_BK1:
            x = CNN(x)
            x = tf.keras.activations.gelu(x)
        x = tf.squeeze(x, axis=2)    
        
        return x 

class point_wise_feed_forward_network(tf.keras.layers.Layer):
    """
    point wise feed forward network
    activations, gelu 
    """
    def __init__(self, dmodel, dff):
        super(point_wise_feed_forward_network, self).__init__()
        
        self.Dense0 = Dense(dff, kernel_initializer='glorot_uniform',
                                       bias_initializer='zeros')
        self.Dense1 = Dense(dmodel, kernel_initializer='glorot_uniform',
                                       bias_initializer='zeros')
    def __call__(self, x):
        
        x = self.Dense0(x)                                                     #(btc_sz,seq_len,dff)#(btc_sz,seq_len,dff)
        x = tf.keras.activations.gelu(x)
        x = self.Dense1(x)                                                     #(btc_sz,seq_len,dff)
        
        return x 

class Enlayer(tf.keras.layers.Layer):
    """
    dff -> 128
    dmodel -> 64 
    k_dims -> 64  此参数和缩放因子有关np.sqrt(dk)
    v_dims -> 64  此参数暂时不太清楚
    num_heads -> 1    
    """
    
    def __init__(self, dff=128, k_dims=64, v_dims=64, dmodel=64, num_heads=1):
        super(Enlayer, self).__init__()
        
        self.mha = MultiHeadAttention(num_heads=num_heads, key_dim=k_dims, value_dim=v_dims)
        self.ffn = point_wise_feed_forward_network(dmodel, dff)
        self.lay1 = LayerNormalization(epsilon=1e-6)
        self.lay2 = LayerNormalization(epsilon=1e-6)
    
    def __call__(self, x, mask=None):
        
        x1 = self.lay1(x)                                                      #(...,q_len,dmodel)
        att_out, wei = self.mha(x1, x1, x1, attention_mask=mask, 
                                return_attention_scores=True)                  #(...,q_len,dmodel)
        att_out = att_out + x                                                  #(...,q_len,dmodel)
        
        att_out1 = self.lay2(att_out)                                          #(...,q_len,dmodel)        
        ffn_out = self.ffn(att_out1)                                           #(...,q_len,dmodel)
        
        ffn_out = ffn_out + att_out                                            #(...,q_len,dmodel)
        
        return ffn_out, wei 

class Encoder(tf.keras.layers.Layer):
    """
    num_layers ->  layers number 
    dff -> 128
    dmodel -> 64 
    k_dims -> 64  此参数和缩放因子有关np.sqrt(dk)
    v_dims -> 64 
    num_heads -> 1    
    """
    def __init__(self, num_layers=1, dff=128, k_dims=64, v_dims=64, dmodel=64, num_heads=1):
        
        super(Encoder, self).__init__()
        
        self.EncoderBK = [ Enlayer(dff, k_dims, v_dims, dmodel, num_heads) for i in range(num_layers) ]
    
    def __call__(self, x, mask=None):
        
        att_wei = {}
        
        i = 0
        
        for Att in self.EncoderBK:
            
            x, wei = Att(x, mask)
            
            att_wei[f'encoder_layer_{i+1}_block'] = wei 
            
            i = i + 1
            
        return x, att_wei    

class MSAM(tf.keras.layers.Layer):
    """
    strides0 = [ (3,1), (6,1) ]
    kr_sz0 = [ (3,1), (6,1) ]
    filters0 = [ 2, 4]

    strides1 = [5, 4, 4, 4 ]
    kr_sz1 = [5, 5, 5, 5 ]
    filters1 = [ 8, 16, 32, 64]

    strides2 = [ 4, 4, 2 ]
    kr_sz2 = [ 5, 5, 3 ]
    filters2 = [ 2, 4, 8 ]
    
    inp_shp0 = [2, 32, 18, 290, 1]
    inp_shp1 = [2, 32, 290, 1 ]
    
    dmodel = 64 
    dff = 128
    k_dims0 = 64
    v_dims0 = 64 
    num_heads = 1
    """
    
    def __init__(self, num_ly0=1, num_ly1=1, dff=128, k_dims=64, v_dims=64, dmodel=64, num_heads=1,
                 filters0=[ 2, 4], kr_sz0 = [ (3,1), (3,1) ], strides0 = [ (3,1), (3,1) ],
                 filters1=[ 8, 16, 32, 64], kr_sz1 = [5, 5, 5, 5 ], strides1 = [5, 4, 4, 4 ],
                 filters2=[ 2, 4, 8 ], kr_sz2 = [ 5, 5, 3 ], strides2 = [ 4, 4, 2 ]):
        
        super(MSAM, self).__init__()
        
        # self.RNet = RNet(filters2, kr_sz2, strides2, dmodel) 
        self.ParamsNet = ParamsNet(filters0, kr_sz0, strides0, filters1, kr_sz1, strides1)
        self.Encoder = Encoder(num_ly0, dff, k_dims, v_dims, dmodel, num_heads)
        # self.Decoder = Decoder(num_ly1, dff, k_dims, v_dims, dmodel, num_heads)
        self.GMP1D = GlobalMaxPooling1D()
        
    def __call__(self, x, x1, t_ind):
    
        x = self.ParamsNet(x)
        
        # x1 = self.RNet(x1)
        # x1 = x1[:, tf.newaxis, : ]
        
        mask0, mask1 = create_mask(t_ind)
        x, enc_wei_bk = self.Encoder(x, mask0)
        x = self.GMP1D(x)
        # x, dec_wei_bk = self.Decoder(x1, x, mask1)
        x = tf.keras.activations.sigmoid(x)
        # x = x = tf.keras.activations.gelu(x)
        # x = tf.squeeze(x, axis = 1)
        return x, enc_wei_bk

class MagLoss(tf.keras.layers.Layer):
    """
    Type:
    Type0, Multi-label, Multi-class
    Type1, Multi-class, one-hot code 
    Type2, MeanSquaredError, Gaussian PDF 
    """
    def __init__(self, Type, weights):
        super(MagLoss,self).__init__()
        self.Type  = Type 
        self.MSE = tf.keras.losses.MeanSquaredError()                          #均方根误差损失函数 
        self.CCE = tf.keras.losses.CategoricalCrossentropy(from_logits=False)  #btc*label 的均值 multi-class one-hot 
        self.BCE = tf.keras.losses.BinaryCrossentropy(from_logits=False)       #btc*label 的均值 多标签 多分类 
        self.wei = weights                                                     #weights 
        #reduction=tf.keras.losses.Reduction.NONE
    
    def call(self,y_true,y_pred):
        #y_pred = tf.squeeze(y_pred,axis=-1)
        if self.Type == 'Type0':
            y_true = y_true[ :, :, tf.newaxis ]
            y_pred = y_pred[ :, :, tf.newaxis ]
            
            x = self.BCE(y_true, y_pred, sample_weight = self.wei)
            
        if self.Type == 'Type1':
            x = self.CCE(y_true,y_pred) * 52.
        
        if self.Type == 'Type2':
            x = self.MSE(y_true,y_pred) 
            
        return x 

class Evaluations():

    """
    Evaluatoins Parameters
    Functions:
    reset, reset PT, PF, Ng 
    Calculations, precision, accuracy, recall
    """
    
    def __init__(self):
        
        self.TP = 0. 
        self.FP = 0.
        self.Ng = 0. 
    
    def reset_states(self):
        
        self.TP = 0. 
        self.FP = 0.
        self.Ng = 0.
    
    def results(self):
        
        """pre, acc, rec"""
        pre = self.TP / ( self.TP + self.FP + 1.)
        acc = self.TP / ( self.TP + self.FP + self.Ng + 1.)
        rec = self.TP  / ( self.TP + self.Ng + 1.)
        # return pre[5,2], acc, rec 
        return [pre, acc, rec]
    
    def __call__(self, lab, pre):
        
        s0, _ = tf.shape(lab)
        
        #Positive Negative index bools
        Peak_lis = tf.range(0.5, 1.0, 0.05, dtype = tf.float32)
        Peak_Exp = tf.ones((s0, len(Peak_lis), 1)) * Peak_lis[tf.newaxis, :, tf.newaxis]
        PreBools = tf.cast( tf.greater_equal(pre[:, tf.newaxis,:], Peak_Exp), dtype = tf.float32)
        PreMag = tf.reduce_sum(PreBools, axis = -1 ) + 20. 
        PreMag = PreMag[:, :, tf.newaxis]
        Pos = tf.cast(tf.greater(PreMag, tf.cast(20., dtype = tf.float32)), dtype = tf.float32 )
        Neg = 1 - Pos 
        TarMag = tf.ones((s0, len(Peak_lis), 1), dtype = tf.float32) * tf.cast(tf.reduce_sum(lab, axis=-1, keepdims = True)[:, tf.newaxis, :], dtype = tf.float32) 
        TarMag = TarMag + tf.cast(20., dtype = tf.float32)

        #True False index bools 
        erros = tf.range(1., 11., 1.)
        Err_exp = tf.ones((s0, 1, 1)) * erros[tf.newaxis, tf.newaxis, :]
        Tru_ind = tf.cast(tf.less_equal( tf.abs(TarMag - PreMag), Err_exp) , dtype = tf.float32 )
        Fal_ind = 1 - Tru_ind 

        TP = Pos * Tru_ind 
        FP = Pos * Fal_ind
        Ng = Neg * (Tru_ind + Fal_ind)

        TP = tf.reduce_sum(TP, axis = 0)
        FP = tf.reduce_sum(FP, axis = 0)
        Ng = tf.reduce_sum(Ng, axis = 0)
        
    
        self.TP += TP
        self.FP += FP 
        self.Ng += Ng 
        
        return True
    
def EvaParmSave(files, Evalis):
    dict_ = {}
    dict_['Train'] = Evalis[0]
    dict_['Test'] = Evalis[1]
    dict_['Valid'] = Evalis[2]
    
    with open(files, 'wb') as f:
        pickle.dump(dict_, f)
    return 