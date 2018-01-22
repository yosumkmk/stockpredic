#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 20:58:30 2017

@author: yosu
"""

import yopath
import numpy as np
import math
import talib as ta
from copy import deepcopy
from ta_yo import RSI
from ta_yo import Bupper
from ta_yo import Bmiddle
from ta_yo import Blower
from newaxis1 import n100
from newaxis1 import n010
from newaxis1 import n001
from newaxis1 import n011
from newaxis1 import n101
from newaxis1 import n110
from newaxis1 import n01
from newaxis1 import n10

def buysk(stdatas,buysig):
    #株種類
    codes=stdatas.shape[2]
    #buysの大きさ
    buysk=np.zeros((stdatas.shape[0],4,stdatas.shape[2]))
    #翌日の価格
    stdatas1f=np.roll(stdatas,-1,axis=0)
    #sig作成用
    sigzero=np.zeros(stdatas.shape[0])
    sigone=np.ones(stdatas.shape[0])
    #sigzero_code,sigone_code,cはcodesの意味
    sigzc=np.tile(n011(sigzero),(1,1,codes))
    sigoc=np.tile(n011(sigone),(1,1,codes))

    #_buy=[買い実行SIG,執行価格,保有日数,売り価格]
    #成り行き価格置換え用
    sigbuy_nari=np.concatenate([sigoc,n010(stdatas1f[:,1,:]),sigzc,sigzc],axis=1)
    #成り行き
    buysk=np.where((np.tile(n010(buysig[:,0,:]),(1,4,1))==np.tile(n011(sigone),(1,4,codes))) 
                         & (np.tile(n010(buysig[:,2,:]),(1,4,1))==np.tile(n011(sigzero),(1,4,codes)))
                         &(np.tile(n010(stdatas1f[:,4,:]),(1,4,1))>0)
                         ,sigbuy_nari,buysk)
    #指値価格置換え1
    sigbuy_sa1=np.concatenate([sigoc,n010(buysig[:,3,:]),sigzc,sigzc],axis=1)        
    #指値＜始値　＆　指値＞安値
    buysk=np.where((np.tile(n010(buysig[:,0,:]),(1,4,1))==np.tile(n011(sigone),(1,4,codes))) 
               & (np.tile(n010(buysig[:,2,:]),(1,4,1))==np.tile(n011(sigone),(1,4,codes)))
               &(np.tile(n010(buysig[:,3,:]),(1,4,1)) < np.tile(n010(stdatas1f[:,1,:]),(1,4,1)))
                &(np.tile(n010(buysig[:,3,:]),(1,4,1)) > np.tile(n010(stdatas1f[:,3,:]),(1,4,1)))
                &(np.tile(n010(stdatas1f[:,4,:]),(1,4,1))>0)
               ,sigbuy_sa1,buysk)
    
    sigbuy_sa2=np.concatenate([sigoc,n010(stdatas1f[:,1,:]),sigzc,sigzc],axis=1)
    #指値＞始値
    #指値価格置換え2
    buysk=np.where((np.tile(n010(buysig[:,0,:]),(1,4,1))==np.tile(n011(sigone),(1,4,codes)))
               & (np.tile(n010(buysig[:,2,:]),(1,4,1))==np.tile(n011(sigone),(1,4,codes)))
               &(np.tile(n010(buysig[:,3,:]),(1,4,1)) > np.tile(n010(stdatas1f[:,1,:]),(1,4,1)))
               &(np.tile(n010(stdatas1f[:,4,:]),(1,4,1))>0)
               ,sigbuy_sa2,buysk)
    
    #逆指値<始値(成り行き)
    sigbuy_sa2=np.concatenate([sigoc,n010(stdatas1f[:,1,:]),sigzc,sigzc],axis=1)
    
    buysk=np.where((np.tile(n010(buysig[:,0,:]),(1,4,1))==np.tile(n011(sigone),(1,4,codes)))
                   &(np.tile(n010(buysig[:,2,:]),(1,4,1))==np.tile(n011(sigone)*2,(1,4,codes)))
                   &(np.tile(n010(buysig[:,3,:]),(1,4,1))==np.tile(n011(sigzero),(1,4,codes)))
                   &(np.tile(n010(buysig[:,4,:]),(1,4,1)) < np.tile(n010(stdatas1f[:,1,:]),(1,4,1)))
                   &(np.tile(n010(stdatas1f[:,4,:]),(1,4,1))>0)
               ,sigbuy_sa2,buysk)

    #逆指値>始値、逆指値＜高値(成り行き)
    sigbuy_sa2=np.concatenate([sigoc,n010(buysig[:,4,:]),sigzc,sigzc],axis=1)
    
    buysk=np.where((np.tile(n010(buysig[:,0,:]),(1,4,1))==np.tile(n011(sigone),(1,4,codes)))
                   &(np.tile(n010(buysig[:,2,:]),(1,4,1))==np.tile(n011(sigone)*2,(1,4,codes)))
                   &(np.tile(n010(buysig[:,3,:]),(1,4,1))==np.tile(n011(sigzero),(1,4,codes)))
                   &(np.tile(n010(buysig[:,4,:]),(1,4,1)) > np.tile(n010(stdatas1f[:,1,:]),(1,4,1)))    
                   &(np.tile(n010(buysig[:,4,:]),(1,4,1)) < np.tile(n010(stdatas1f[:,2,:]),(1,4,1)))
                   &(np.tile(n010(stdatas1f[:,4,:]),(1,4,1))>0)
               ,sigbuy_sa2,buysk)
    
    #逆指値&指値（逆指値＜高値、指値＞安値で包括可能）
    sigbuy_sa2=np.concatenate([sigoc,n010(buysig[:,3,:]),sigzc,sigzc],axis=1)
    
    buysk=np.where((np.tile(n010(buysig[:,0,:]),(1,4,1))==np.tile(n011(sigone),(1,4,codes)))
                   &(np.tile(n010(buysig[:,2,:]),(1,4,1))==np.tile(n011(sigone)*2,(1,4,codes)))
                   &(np.tile(n010(buysig[:,3,:]),(1,4,1))>np.tile(n011(sigzero),(1,4,codes)))
                   &(np.tile(n010(buysig[:,4,:]),(1,4,1)) < np.tile(n010(stdatas1f[:,2,:]),(1,4,1)))    
                   &(np.tile(n010(buysig[:,3,:]),(1,4,1)) > np.tile(n010(stdatas1f[:,3,:]),(1,4,1)))
                   &(np.tile(n010(stdatas1f[:,4,:]),(1,4,1))>0)
               ,sigbuy_sa2,buysk)
    #指値寄り付きのみ
    sigbuy_sa2=np.concatenate([sigoc,n010(stdatas1f[:,1,:]),sigzc,sigzc],axis=1)
    #指値＞始値
    #指値価格置換え2
    buysk=np.where((np.tile(n010(buysig[:,0,:]),(1,4,1))==np.tile(n011(sigone),(1,4,codes)))
               & (np.tile(n010(buysig[:,2,:]),(1,4,1))==np.tile(n011(sigone)*3,(1,4,codes)))
               &(np.tile(n010(buysig[:,3,:]),(1,4,1)) > np.tile(n010(stdatas1f[:,1,:]),(1,4,1)))
               &(np.tile(n010(stdatas1f[:,4,:]),(1,4,1))>0)
               ,sigbuy_sa2,buysk)


    return buysk
