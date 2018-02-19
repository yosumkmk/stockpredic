#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 20:30:33 2017

@author: yosu
"""
from numba import jit
import numpy as np
import talib as ta

@jit
def RSI(stdatas,periods):
    stRSI=np.zeros((stdatas.shape[0],len(periods),stdatas.shape[1]))
    for code in range(stdatas.shape[1]):
        for prd in range(len(periods)):
            stRSI[:,prd,code]=ta.RSI(stdatas[:,code],periods[prd])
    stRSI[np.isnan(stRSI)]=0
    return stRSI
@jit
def EMA(stdatas,periods):
    stEMA=np.zeros((stdatas.shape[0],len(periods),stdatas.shape[1]))
    for code in range(stdatas.shape[1]):
        for prd in range(len(periods)):
            stEMA[:,prd,code]=ta.EMA(stdatas[:,code],periods[prd])
    stEMA[np.isnan(stEMA)]=0
    return stEMA
@jit
def Bupper(stdatas,periods,siguma):
    Bupper=np.zeros((stdatas.shape[0],len(periods),stdatas.shape[1]))
    Bmiddle=np.zeros((stdatas.shape[0],len(periods),stdatas.shape[1]))
    Blower=np.zeros((stdatas.shape[0],len(periods),stdatas.shape[1]))
    for code in range(stdatas.shape[1]):
        for prd in range(len(periods)):
            Bupper[:,prd,code],Bmiddle[:,prd,code],Blower[:,prd,code]=ta.BBANDS(stdatas[:,code], periods[prd], siguma, siguma)
    Bupper[np.isnan(Bupper)]=0
    return Bupper
@jit
def Bmiddle(stdatas,periods,siguma):
    Bupper=np.zeros((stdatas.shape[0],len(periods),stdatas.shape[1]))
    Bmiddle=np.zeros((stdatas.shape[0],len(periods),stdatas.shape[1]))
    Blower=np.zeros((stdatas.shape[0],len(periods),stdatas.shape[1]))
    for code in range(stdatas.shape[1]):
        for prd in range(len(periods)):
            Bupper[:,prd,code],Bmiddle[:,prd,code],Blower[:,prd,code]=ta.BBANDS(stdatas[:,code], periods[prd], siguma, siguma)
    Bmiddle[np.isnan(Bmiddle)]=0
    return Bmiddle
@jit
def Blower(stdatas,periods,siguma):
    Bupper=np.zeros((stdatas.shape[0],len(periods),stdatas.shape[1]))
    Bmiddle=np.zeros((stdatas.shape[0],len(periods),stdatas.shape[1]))
    Blower=np.zeros((stdatas.shape[0],len(periods),stdatas.shape[1]))
    for code in range(stdatas.shape[1]):
        for prd in range(len(periods)):
            Bupper[:,prd,code],Bmiddle[:,prd,code],Blower[:,prd,code]=ta.BBANDS(stdatas[:,code], periods[prd], siguma, siguma)
    Blower[np.isnan(Blower)]=0
    return Blower
@jit
def bbandy(stdatas,periods,siguma):
    Bupper=np.zeros((stdatas.shape[0],len(periods),stdatas.shape[1]))
    Bmiddle=np.zeros((stdatas.shape[0],len(periods),stdatas.shape[1]))
    Blower=np.zeros((stdatas.shape[0],len(periods),stdatas.shape[1]))
    for code in range(stdatas.shape[1]):
        for prd in range(len(periods)):
            Bupper[:,prd,code],Bmiddle[:,prd,code],Blower[:,prd,code]=ta.BBANDS(stdatas[:,code], periods[prd], siguma, siguma)
    Blower[np.isnan(Blower)]=0
    return Bupper,Bmiddle,Blower