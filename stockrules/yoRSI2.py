#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 22:38:04 2018

@author: yosu
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 31 13:07:19 2017

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
from yosellsk import sellsk

def RSI2(stdatas):

    #sig作成用
    sigzero=np.zeros(stdatas.shape[0])
    sigone=np.ones(stdatas.shape[0])
    #sigzerocode,sigonecode
    sigzeroc=np.tile(n011(sigzero),(1,1,stdatas.shape[2]))
    sigonec=np.tile(n011(sigone),(1,1,stdatas.shape[2]))
    #sigの大きさ決め
    sigRSIb=np.zeros((stdatas.shape[0],5,stdatas.shape[2]))
    #出来高量確認
    volave=np.tile(n010(stmin(stdatas[:,8,:],[50])),(1,7,1))
    #過去のを参照したいときはnp.roll(XXX,プラス,XXX)
    stdatas1p=np.roll(stdatas,1,axis=0)
    #欠損値のあるところはシグナル出ないようにする
    maxperiod=[100]
    minmp=stmin(stdatas[:,2,:],maxperiod)
    #nosigiが0なら欠損値なし
    nosig=np.tile(n010(stc3(minmp,maxperiod)),(1,7,1))+np.tile(n010(stc3(stdatas[:,4,:],[20])),(1,7,1))
    #移動平均剥離率7
    ave7h=((n010(stdatas[:,4,:])/n010(stmean(stdatas[:,4,:],[7]))-1)*100)
    stac=stc(stdatas[:,4,:],stmean(stdatas[:,4,:],[100]),[30])
    stRSI=RSI(stdatas[:,4,:],[14])
    stBlower=Blower(stdatas[:,4,:],[13],2)
    stBlower1p=np.roll(stBlower,1,axis=0)
    
    sigRb1=np.concatenate([sigonec,sigzeroc,sigzeroc,sigzeroc,sigzeroc,n010(stdatas[:,4,:]),ave7h],axis=1)
    sigRb2=np.concatenate([sigzeroc,sigzeroc,sigzeroc,sigzeroc,sigzeroc,sigzeroc,sigzeroc],axis=1)
    #stopLはstdatas[:,5,:],stophはstdatas[:,6,:]
    #１→始値、２→高値、３→安値、４→終値
    sigRSIb=np.where(((np.tile(n010(stac),(1,7,1))>=20)&
                     (np.tile(stRSI,(1,7,1))<25)&
                     (np.tile(stBlower1p,(1,7,1))>np.tile(n010(stdatas1p[:,3,:]),(1,7,1)))&      
                     (np.tile(n010(stdatas[:,4,:]),(1,7,1))!= np.tile(n010(stdatas1p[:,5,:]),(1,7,1)))&
                     (np.tile(n010(stdatas[:,4,:]),(1,7,1))<np.tile(n010(stdatas1p[:,3,:])*0.96,(1,7,1)))&
                     (nosig==0)&
                     (volave>50))
                      ,sigRb1,sigRb2)
    
    sigRa1=np.concatenate([sigonec,sigzeroc,sigzeroc,sigzeroc,sigzeroc,n010(stdatas[:,4,:]),ave7h],axis=1)
    #stopLはstdatas[:,5,:],stophはstdatas[:,6,:]
    #買いシグナル
    sigRSIb=np.where(((np.tile(n010(stac),(1,7,1))>=20)&
                     (np.tile(stRSI,(1,7,1))<25)&
                     (np.tile(stBlower1p,(1,7,1))<=np.tile(n010(stdatas1p[:,3,:]),(1,7,1)))&      
                     (np.tile(n010(stdatas[:,4,:]),(1,7,1))!= np.tile(n010(stdatas1p[:,5,:]),(1,7,1)))&
                     (np.tile(n010(stdatas[:,3,:]),(1,7,1))<np.tile(stBlower1p*0.96,(1,7,1)))&
                     (nosig==0)&
                     (volave>50))
                      ,sigRa1,sigRSIb)

    return sigRSIb
    

def RSI2sell(stdatas,buysk):

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
    RSI3=RSI(stdatas[:,4,:],[3])

    for i in range(len(hoji[0,:])):
        
        #注文別、[sig,売買、成指区分、指値価格、逆指値価格]
        #sellsigsは時間、上の５種類、株コード
        #sellsigの大きさは(zdate[i],5)
        #執行シグナルごとに売りシグナル出すから、株コードと執行シグナル別に
        #残り日数
        alldate=stdatas.shape[0]
        zdate=alldate-hoji[0,:]
        #買い執行手番の一日を追加
        sellsig=np.zeros((zdate[i],5))
        #sig作成用
        sigzero1=np.zeros(zdate[i])
        sigone1=np.ones(zdate[i])
        #執行日から売りシグナル入れる
        #売りシグナル計算→執行関数とは別
        sellsiga=np.array([sigone1,sigone1,sigzero1,sigzero1,sigzero1]).T
        sellsig=np.where(np.tile(n01(RSI3[hoji[0,i]::,0,hoji[2,i]]),(1,5))>70,
                         sellsiga,sellsig)
        sigm1=sellsk(stdatas,sellsig,sigm1,hoji,i,stdatas1day)
    return sigm1
