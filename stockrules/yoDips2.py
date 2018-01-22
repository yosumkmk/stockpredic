#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 21:58:48 2018

@author: yosu
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 19:11:28 2018

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

def Dips2(stdatas):

    #sig作成用
    sigzero=np.zeros(stdatas.shape[0])
    sigone=np.ones(stdatas.shape[0])
    #sigzerocode,sigonecode
    sigzeroc=np.tile(n011(sigzero),(1,1,stdatas.shape[2]))
    sigonec=np.tile(n011(sigone),(1,1,stdatas.shape[2]))
    #出来高量確認
    volave=np.tile(n010(stmin(stdatas[:,8,:],[50])),(1,7,1))
    #過去のを参照したいときはnp.roll(XXX,プラス,XXX)
    stdatas1p=np.roll(stdatas,1,axis=0)
    #前日比率
    hiritu=((stdatas/stdatas1p-1)*100)
    #移動平均剥離率7
    ave7h=((n010(stdatas[:,4,:])/n010(stmean(stdatas[:,4,:],[7]))-1)*100)
    #移動平均剥離率21
    ave21h=np.tile(((n010(stdatas[:,4,:])/n010(stmean(stdatas[:,4,:],[21]))-1)*100),(1,7,1))
    ave21h1p=np.roll(ave21h,1,axis=0)
    #移動平均剥離率15
    ave15h=np.tile(((n010(stdatas[:,4,:])/n010(stmean(stdatas[:,4,:],[15]))-1)*100),(1,7,1))
    ave15h1p=np.roll(ave15h,1,axis=0)
    #過去４０日間に移動平均１３が終値より小さい日（同じ含む）が２２以上存在する→終値が移動平均より大きいと同意義
    ave13h=stmean(stdatas[:,4,:],[13])
    stac=np.tile(n010(stc(stdatas[:,4,:],ave13h,[40])),(1,7,1))
    #ボリンジャー１日前７日
    stBlower7=np.tile(Blower(stdatas[:,4,:],[7],2),(1,7,1))
    stBlower71p=np.roll(stBlower7,1,axis=0)
    #欠損値のあるところはシグナル出ないようにする
    maxperiod=[21]
    #nosigiが0なら欠損値なし
    nosigp=stmean(stdatas[:,4,:],maxperiod)
    nosig=np.tile(n010(stc3(nosigp,maxperiod)),(1,7,1))+np.tile(n010(stc3(stdatas[:,4,:],[20])),(1,7,1))


    #[sig,買い売り、成り行き指値、指値価格、逆指値価格]
    #前日比率が０より大きいとき
    sig1=np.concatenate([sigonec,sigzeroc,sigonec,n010(stdatas[:,4,:]*0.965),sigzeroc,n010(stdatas[:,4,:]*0.965),ave7h],axis=1)
    sig=np.concatenate([sigzeroc,sigzeroc,sigzeroc,sigzeroc,sigzeroc,sigzeroc,sigzeroc],axis=1)
    #stopLはstdatas[:,5,:],stophはstdatas[:,6,:]
    #１→始値、２→高値、３→安値、４→終値
    sig=np.where((((np.tile(n010(hiritu[:,4,:]),(1,7,1)))>0)&
                 (ave21h1p<-15)&
                 (ave15h1p<-16)&
                 (np.tile(n010(stdatas1p[:,4,:]),(1,7,1))<stBlower71p)&
                 (stac<22)&
                 (nosig==0)&
                 (volave>0))
                ,sig1,sig)
    #前日比率が0以下の時
    sig1=np.concatenate([sigonec,sigzeroc,sigonec,n010(stdatas[:,4,:]*0.94),sigzeroc,n010(stdatas[:,4,:]*0.94),ave7h],axis=1)
    sig=np.where((((np.tile(n010(hiritu[:,4,:]),(1,7,1)))<=0)&
                 (ave21h1p<-15)&
                 (ave15h1p<-16)&
                 (np.tile(n010(stdatas1p[:,4,:]),(1,7,1))<stBlower71p)&
                 (stac<22)&
                 (nosig==0)&
                 (volave>0))
                ,sig1,sig)
    sigz=np.concatenate([sigzeroc,sigzeroc,sigzeroc,sigzeroc,sigzeroc,sigzeroc,sigzeroc],axis=1)        
    sig=np.where(((np.tile(n011(stc5(sig[:,0,:])),(1,7,stdatas.shape[2]))>=6)&
                 (volave>50))
                ,sig,sigz)

    return sig


def Dips2sell(stdatas,buysk):

    #sellsigは売りシグナル、この３、４列目に保有日数と執行価格を入れる
    #sig作成用
    codes=stdatas.shape[2]
    sigzero=np.zeros(stdatas.shape[0])
    sigone=np.ones(stdatas.shape[0])
    #sigzerocode,sigonecode
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
        

        zdate1=alldate-hoji[0,i]
        holddays=(np.arange(zdate1))
        #sig作成用
        sigzero1=np.zeros(zdate[i])
        sigone1=np.ones(zdate[i])
    
        #移動平均２５と安値の２日連続文
        #売りシグナル計算→執行関数とは別
        #１→始値、２→高値、３→安値、４→終値
        #保有日数５日以内＆２日連続で安値が移動平均２５より小さい
        #sellsigの大きさ
        sellsig=np.zeros((zdate[i],5))
    
        #保有日数3日より大きい
        sellsiga=np.array([sigone1,sigone1,sigone1*2,sigzero1,stdatas[hoji[0,i]::,4,hoji[2,i]]-stdatas[hoji[0,i]::,7,hoji[2,i]]]).T
        sellsig=np.where(np.tile(n01(holddays),(1,5))>3
                        ,sellsiga,sellsig)

        sigm1=sellsk(stdatas,sellsig,sigm1,hoji,i,stdatas1day)
    return sigm1
