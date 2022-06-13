# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 10:13:20 2022

@author: hbrhu
"""
import os
import numpy as np 
import matplotlib as mpb 
mpb.use('agg')
import matplotlib.pyplot as plt 
from matplotlib import gridspec 

def PlotFigPdf(Pre, Lab, t_ind, name, ind, dir_):
    """
    Plot Gaussian Probability Density Functions 
    """
    x_0 = np.arange(0., 1024, 1) * 0.01 
    t = np.arange(1, 291) * 0.1 
    t_sec = np.sum(t_ind, axis = -1) * 0.1 
    t0 = np.max(t_sec)
    s0, _ = t_ind.shape
    LM = round(np.argmax(Lab) * 0.01, 2) 
    PM = round(np.argmax(Pre) * 0.01, 2)  
    LMI = int(LM*10)
    file = os.sep.join([dir_, '{}_{}_{}_{}.png'.format(LMI, name, ind, t0)])
    
    fig = plt.figure()
    spec = gridspec.GridSpec(nrows = 2, ncols = 1, height_ratios = [1,4])
    ax0 = fig.add_subplot(spec[0])
    ax0.set_xlim(-0.05, 10.25)
    ax0.set_ylim(-0.1, 1.1)
    ax0.plot(x_0, Pre,'r')
    ax0.plot([LM, LM], [0., 1.0], 'black')
    ax0.text(2, 0.7, 'LMag={}'.format(LM))
    ax0.text(2, 0.3, 'PMag={}'.format(PM))
    plt.title(name)
    
    ax1 = fig.add_subplot(spec[1])
    ax1.set_xlim(0., 30.)
    ax1.set_ylim(-0.2, 10)
    yd = [ 1.2 * x for x in range(8)]
    for i in range(s0):
        y_ = t_ind[i,:] + yd[-i-1]
        ax1.plot(t, y_)
        ax1.text(20, yd[-i-1] + 0.5, 't={}s'.format(t_sec[i]) )
        
    plt.savefig(file, dpi = 300)
    plt.close()
    
    return True 

def PlotFigLab(Pre, Lab, thre, t_ind, name, ind, dir_):
    """
    Plot Fig Multiple Lable Class 
    """
    x_0 = np.arange(21, 85, 1) * 0.1
    t = np.arange(1, 301) * 0.1 
    t_sec = np.sum(t_ind, axis = -1) * 0.1 
    t0 = np.max(t_sec)
    s0, _ = t_ind.shape
    LM = round((np.sum(Lab) + 20) * 0.1, 2)
    PM = round((np.sum(np.greater_equal(Pre, thre) + 0 ) + 20) *0.1, 2)
    LMI = int(LM*10)
    file = os.sep.join([dir_, '{}_{}_{}_{}.png'.format(LMI, name, ind, t0)])
    
    fig = plt.figure(figsize=(20,10))
    spec = gridspec.GridSpec(nrows = 2, ncols = 1, height_ratios = [1,4])
    ax0 = fig.add_subplot(spec[0])
    ax0.set_xlim(-0.05, 10.25)
    ax0.set_ylim(-0.1, 1.1)
    ax0.plot(x_0, Pre,'r')
    ax0.plot([LM, LM], [0., 1.0], 'black')
    ax0.text(2, 0.7, 'LMag={}'.format(LM))
    ax0.text(2, 0.3, 'PMag={}'.format(PM))
    plt.title(name)
    
    ax1 = fig.add_subplot(spec[1])
    ax1.set_xlim(0., 30.)
    ax1.set_ylim(-0.2, 40)
    yd = [ 1.2 * x for x in range(32)]
    for i in range(s0):
        y_ = t_ind[i,:] + yd[-i-1]
        ax1.plot(t, y_)
        ax1.text(20, yd[-i-1] + 0.5, 't={}s'.format(t_sec[i]) )
        
    plt.savefig(file, dpi = 300)
    plt.close()
    
    return True 

def PltResRT(PreMl, Mag, trigger_num, t_lis, event, Depth, dir_):
    """
    plot Fig Multiple label Class Resutls
    """
    Mlis =[ Mag for x in t_lis ]
    name = '{}_M={}_{}km'.format(event, Mag, Depth)
    fig = plt.figure(figsize=(12,6))
    plt.title(name)
    plt.plot(t_lis, Mlis, linewidth=1.5, linestyle='dashed', color='black')
    plt.scatter(t_lis, PreMl, marker='x', s=30, c='red')
    plt.ylabel('Magnitude')
    plt.xlabel('Time(sec')
    plt.ylim(2.5, 7.5)
    for i in range(len(trigger_num)):
        plt.text(t_lis[i], Mlis[i], int(trigger_num[i]),
                 horizontalalignment='center', verticalalignment='bottom')
    file = os.sep.join([dir_, '{}_{}_{}km.jpg'.format(Mag, event, Depth)])
    plt.savefig(file, dpi=300)
    plt.close()
    return True 

def SpMag(Mag, x, thre=-1.):
    x0 = [ t for t in x if t > thre ]
    if x0 == []:
        x0 = [-1]
    x1 = np.array(x0) - Mag
    x1 = np.abs(x1)
    return np.max(x1)