#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 17:39:28 2018

@author: yosu
"""

import yopath
import psycopg2 as pg
import stockdata
import pandas.io.sql as psql
import sqlite3
import numpy as np
import xarray as xr
import math
import talib as ta
from copy import deepcopy
import yoim
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
from stmam import stmin
from stmam import stmax
from stmam import stmean
from stmam import stc
from stmam import stc2
from stmam import stc3
from stmam import stc4
from stmam import stc5
from yosellsk import sellsk

def bfkai41(stdatas):

    #sig作成用
    sigzero=np.zeros(stdatas.shape[0])
    sigone=np.ones(stdatas.shape[0])
    #sigzerocode,sigonecode
    sigzeroc=np.tile(n011(sigzero),(1,1,stdatas.shape[2]))
    sigonec=np.tile(n011(sigone),(1,1,stdatas.shape[2]))
    #出来高量確認
    volave=np.tile(n010(stmin(stdatas[:,8,:],[50])),(1,7,1))
    #移動平均剥離率7
    ave7h=((n010(stdatas[:,4,:])/n010(stmean(stdatas[:,4,:],[7]))-1)*100)

    #過去6日間に終値終値が移動平均２１の−１０％〜〜１６％の範囲内の日が１日以上
    ave21=stmean(stdatas[:,4,:],[21])
    stac21=np.tile(n010(stc4(stdatas[:,4,:],ave21*0.84,ave21*0.9,[6])),(1,7,1))
    #過去７０日間に移動平均50が移動平均50（2日前）より大きい日が50日以上
    ave50=stmean(stdatas[:,4,:],[50])
    ave502p=np.roll(ave50,2,axis=0)
    stac50=np.tile(n010(stc(ave50,ave502p,[70])),(1,7,1))
    #ボリンジャー１３
    stBlower=Blower(stdatas[:,4,:],[13],2)
    
    #欠損値のあるところはシグナル出ないようにする
    maxperiod=[50]
    #nosigiが0なら欠損値なし

    nosigp=stmean(stdatas[:,4,:],maxperiod)
    nosig=np.tile(n010(stc3(nosigp,maxperiod)),(1,7,1))+np.tile(n010(stc3(stdatas[:,4,:],[20])),(1,7,1))


    #[sig,買い売り、成り行き指値、指値価格、逆指値価格、必要額、優先順位]
    sig1=np.concatenate([sigonec,sigzeroc,sigonec*3,n010(stdatas[:,4,:]*0.9675),sigzeroc,n010(stdatas[:,4,:]*0.9675),ave7h],axis=1)
    sig=np.concatenate([sigzeroc,sigzeroc,sigzeroc,sigzeroc,sigzeroc,sigzeroc,sigzeroc],axis=1)
    #stopLはstdatas[:,5,:],stophはstdatas[:,6,:]
    #１→始値、２→高値、３→安値、４→終値
    sig=np.where(((np.tile(n010(stdatas[:,3,:]),(1,7,1))<np.tile(stBlower,(1,7,1)))&
                 (stac21>=1)&
                 (stac50>=50)&
                 (nosig==0)&
                 (volave>0))
                ,sig1,sig)
    sigz=np.concatenate([sigzeroc,sigzeroc,sigzeroc,sigzeroc,sigzeroc,sigzeroc,sigzeroc],axis=1)    
    sig=np.where(((np.tile(n011(stc5(sig[:,0,:])),(1,7,stdatas.shape[2]))>=10)&
                 (volave>50))
                ,sig,sigz)

    return sig


def bfkai41sell(stdatas,buysk):

    #sellsigは売りシグナル、この３、４列目に保有日数と執行価格を入れる
    #sig作成用
    codes=stdatas.shape[2]
    sigzero=np.zeros(stdatas.shape[0])
    sigone=np.ones(stdatas.shape[0])
    #sigzerocode,sigonecode
    sigzc=np.tile(n011(sigzero),(1,1,codes))
    sigoc=np.tile(n011(sigone),(1,1,codes))
    sigm1=deepcopy(buysk)
    #hoji[(日付、種類、コード）、（シグナル執行順＝合計が買い回数)）
    hoji=deepcopy(np.array(np.where(n010(buysk[:,0,:])==sigoc)))
    #残り日数
    alldate=stdatas.shape[0]
    zdate=alldate-hoji[0,:]

    #翌日,2日後の価格（買いシグナル基準）
    #未来の参照はマイナスを入れる
    stdatas1day=np.roll(stdatas,-1,axis=0)    
    #保有日数（買執行日から何日か、買い執行日は一日目）,最低保有日数はhdate+2
    for i in range(len(hoji[0,:])):
        
        #注文別、[sig,売買、成指区分、指値価格、逆指値価格]
        #保有日数
        zdate1=alldate-hoji[0,i]
        holddays=(np.arange(zdate1))
        #sig作成用
        sigzero1=np.zeros(zdate[i])
        sigone1=np.ones(zdate[i])
        #執行日から売りシグナル入れる


        #sellsigの大きさ
        sellsig=np.zeros((zdate[i],5))
    
        #保有日数4日以下→建値＋１４％
        sellsiga=np.array([sigone1,sigone1,sigone1,np.ones((zdate[i]))*(stdatas[hoji[0,i]-1,4,hoji[2,i]])*1.2,sigzero1]).T
        sellsig=np.where((np.tile(n01(holddays),(1,5)))<=1
                        ,sellsiga,sellsig)
    
        #保有日数4日より大きい
        sellsiga=np.array([sigone1,sigone1,sigone1*2,sigzero1,stdatas[hoji[0,i]::,4,hoji[2,i]]-stdatas[hoji[0,i]::,7,hoji[2,i]]]).T
        sellsig=np.where(np.tile(n01(holddays),(1,5))>1
                        ,sellsiga,sellsig)
        sigm1=sellsk(stdatas,sellsig,sigm1,hoji,i,stdatas1day)
    return sigm1
