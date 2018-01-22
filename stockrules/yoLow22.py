#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 10:48:44 2018

@author: yosu
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 22:12:51 2018

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
from stmam import stc5
from yosellsk import sellsk

def Low22(stdatas):

    #sig作成用
    sigzero=np.zeros(stdatas.shape[0])
    sigone=np.ones(stdatas.shape[0])
    #sigzerocode,sigonecode
    sigzeroc=np.tile(n011(sigzero),(1,1,stdatas.shape[2]))
    sigonec=np.tile(n011(sigone),(1,1,stdatas.shape[2]))
    #出来高量確認
    volave=np.tile(n010(stmin(stdatas[:,8,:],[50])),(1,7,1))
    #過去のを参照したいときはnp.roll(XXX,プラス,XXX)
    #５０日「連続で安値が期間安値７５より大きい
    min75=stmin(stdatas[:,3,:],[75])
    minmin75=stc(stdatas[:,3,:],min75,[50])
    #終値の位置７５が０と３０の範囲内
    max75=stmax(stdatas[:,2,:],[75])
    oichi75=((stdatas[:,4,:]-min75)/(max75-min75)*100)
    #欠損値のあるところはシグナル出ないようにする
    maxperiod=[75]
    #移動平均剥離率25
    ave25h=((n010(stdatas[:,4,:])/n010(stmean(stdatas[:,4,:],[25]))-1)*100)

    #nosigiが0なら欠損値なし
    nosig=np.tile(n010(stc3(min75,maxperiod)),(1,7,1))+np.tile(n010(stc3(stdatas[:,4,:],[20])),(1,7,1))
    #[sig,買い売り、成り行き指値、指値価格、逆指値価格]
    #期間安値（５０）−６％で指値
    min50=stmin(stdatas[:,3,:],[50])
    sig1=np.concatenate([sigonec,sigzeroc,sigonec,n010(min50*0.938),sigzeroc,n010(min50*0.938),ave25h],axis=1)
    sig2=np.concatenate([sigzeroc,sigzeroc,sigzeroc,sigzeroc,sigzeroc,sigzeroc,sigzeroc],axis=1)
    #stopLはstdatas[:,5,:],stophはstdatas[:,6,:]
    #１→始値、２→高値、３→安値、４→終値
    sig=np.where(((np.tile(n010(minmin75),(1,7,1))==50)&
                 (np.tile(n010(stdatas[:,4,:]),(1,7,1))<np.tile(n010(min50*1.02),(1,7,1)))&
                 (np.tile(n010(oichi75),(1,7,1))>=0)&
                 (np.tile(n010(oichi75),(1,7,1))<=25)&
                 (nosig==0)&
                 (volave>0))
                  ,sig1,sig2)
    sig=np.where((np.tile(n011(stc5(sig[:,0,:])),(1,7,stdatas.shape[2]))>=12&
                 (volave>50))
                ,sig,sig2)
    return sig


def Low22sell(stdatas,buysk):

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
    Bupper25=Bupper(stdatas[:,4,:],[25],2)
    for i in range(len(hoji[0,:])):
        
        #買い執行手番の一日を追加
        sellsig=np.zeros((zdate[i],5))
        #保有日数
        zdate1=alldate-hoji[0,i]
        holddays=(np.arange(zdate1))
        #sig作成用
        sigzero1=np.zeros(zdate[i])
        sigone1=np.ones(zdate[i])
        #移動平均２５と安値の２日連続文
        #売りシグナル計算→執行関数とは別
        #１→始値、２→高値、３→安値、４→終値
        #保有日数５日以内＆
     #   sellsiga=np.array([sigone1,sigone1,sigone1,sigone1,sigzero1]).T
        sellsiga=np.array([sigone1,sigone1,sigone1,(Bupper25[hoji[0,i]::,0,hoji[2,i]]*1.03),sigzero1]).T
        sellsig=np.where((np.tile(n01(holddays),(1,5))<=5)
                        ,sellsiga,sellsig)
    
        #保有日数５日より大きい
        sellsiga=np.array([sigone1,sigone1,sigone1*2,sigzero1,stdatas[hoji[0,i]::,4,hoji[2,i]]-stdatas[hoji[0,i]::,7,hoji[2,i]]]).T
        sellsig=np.where(np.tile(n01(holddays),(1,5))>5
                        ,sellsiga,sellsig)
        sigm1=sellsk(stdatas,sellsig,sigm1,hoji,i,stdatas1day)
    return sigm1
