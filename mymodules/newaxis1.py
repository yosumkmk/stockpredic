#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 11:59:45 2017

@author: yosu
"""

import numpy as np

def n011(na100):
    na011=na100[:,np.newaxis,np.newaxis]
    return na011
        
def n101(na010):
    na101=na010[np.newaxis,:,np.newaxis]
    return na101

def n110(na001):
    na110=na001[np.newaxis,np.newaxis,:]
    return na110

def n100(nb011):
    nb100=nb011[np.newaxis,:,:]
    return nb100

def n010(nb101):
    nb010=nb101[:,np.newaxis,:]
    return nb010

def n001(nb110):
    nb001=nb110[:,:,np.newaxis]
    return nb001

def n01(nb10):
    nb01=nb10[:,np.newaxis]
    return nb01

def n10(nb01):
    nb10=nb01[np.newaxis,:]
    return nb10