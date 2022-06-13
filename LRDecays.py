# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 10:11:30 2022

@author: hbrhu
"""
import tensorflow as tf 

class CustSchdle(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, init_lr, lr_min=1e-5, warmup_steps=2048):
    super(CustSchdle, self).__init__()
    self.init_lr = tf.cast(init_lr, tf.float32) 
    self.warmup_steps = warmup_steps
    self.lr_min = tf.cast(lr_min, tf.float32)

  def __call__(self, step):
      if step < self.warmup_steps:
          warmup_percent_done = step / self.warmup_steps
          warmup_lr = self.init_lr * warmup_percent_done
          lr = warmup_lr
      else:
          num_init = step// self.warmup_steps
          lr = self.init_lr * 0.86**(1.2*num_init)
          lr = tf.math.maximum(lr, self.lr_min)
      return lr