#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 22:12:51 2018

@author: yosu
"""

import yopath
import numpy as np
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
from stmam import stmin
from stmam import stmax
from stmam import stmean
from stmam import stc
from stmam import stc2
from stmam import stc3
from yosellsk import sellsk

def UPtrend(stdatas):

    #sig作成用
    sigzero=np.zeros(stdatas.shape[0])
    sigone=np.ones(stdatas.shape[0])
    #sigzerocode,sigonecode
    sigzeroc=np.tile(n011(sigzero),(1,1,stdatas.shape[2]))
    sigonec=np.tile(n011(sigone),(1,1,stdatas.shape[2]))
    #過去のを参照したいときはnp.roll(XXX,プラス,XXX)
    stdatas1p=np.roll(stdatas,1,axis=0)
    #出来高量確認
    volave=np.tile(n010(stmin(stdatas[:,8,:],[50])),(1,7,1))
    #欠損値のあるところはシグナル出ないようにする
    maxperiod=[15]
    minmp=stmin(stdatas[:,2,:],maxperiod)
    #nosigiが0なら欠損値なし
    nosig=np.tile(n010(stc3(minmp,maxperiod)),(1,7,1))+np.tile(n010(stc3(stdatas[:,4,:],[20])),(1,7,1))
    #移動平均剥離率7
    ave15h=((n010(stdatas[:,4,:])/n010(stmean(stdatas[:,4,:],[15]))-1)*100)

    #前日との比率
    hiritu=((stdatas/stdatas1p-1)*100)
    #移動平均剥離率１３
    ave13h=np.tile(((n010(stdatas[:,4,:])/n010(stmean(stdatas[:,4,:],[13]))-1)*100),(1,7,1))
    #翌日寄付ギャップ率
    yyg=np.tile(((n010(stdatas[:,1,:])/n010(stdatas1p[:,4,:])-1)*100),(1,7,1))
    #期間高値、安値の比率
    tyhiritu=np.tile(n010(((stmax(stdatas[:,2,:],[75])/stmin(stdatas[:,3,:],[75])))),(1,7,1))
    #終値→高値
    omigit=np.tile(n010(((stdatas[:,2,:]/stdatas[:,4,:]-1)*100)),(1,7,1))
    #[sig,買い売り、成り行き指値、指値価格、逆指値価格]
    sig1=np.concatenate([sigonec,sigzeroc,sigonec*2,sigzeroc,n010(stdatas[:,4,:]*1.03),n010(stdatas[:,4,:]*1.05),ave15h],axis=1)
    sig2=np.concatenate([sigzeroc,sigzeroc,sigzeroc,sigzeroc,sigzeroc,sigzeroc,sigzeroc],axis=1)
    #stopLはstdatas[:,5,:],stophはstdatas[:,6,:]
    #１→始値、２→高値、３→安値、４→終値
    sig=np.where(((np.tile(n010(hiritu[:,4,:]),(1,7,1))<5)&
                 (np.tile(n010(hiritu[:,4,:]),(1,7,1))>-5)&
                 (np.tile(n010(stdatas1p[:,6,:]),(1,7,1))<np.tile(n010(stdatas[:,2,:]),(1,7,1)))&
                 (ave13h<10)&
                 (ave13h>-6)&
                 (yyg>0)&
                 (tyhiritu>=2)&
                 (omigit<3)&
                 (volave>50)&
                 (nosig==0))
                  ,sig1,sig2)
    #[sig,買い売り、成り行き指値、指値価格、逆指値価格]
    sig1=np.concatenate([sigonec,sigzeroc,sigonec*2,sigzeroc,n010(stdatas[:,2,:]+stdatas1p[:,7,:]),n010(stdatas[:,2,:]+stdatas[:,7,:]),ave15h],axis=1)

    sig=np.where(((np.tile(n010(hiritu[:,4,:]),(1,7,1))<5)&
                 (np.tile(n010(hiritu[:,4,:]),(1,7,1))>-5)&
                 (np.tile(n010(stdatas1p[:,6,:]),(1,7,1))<np.tile(n010(stdatas[:,2,:]),(1,7,1)))&                 
                 (ave13h<10)&
                 (ave13h>-6)&
                 (yyg>0)&
                 (tyhiritu>=2)&
                 (omigit>=3)&
                 (volave>50)&
                 (nosig==0))
                  ,sig1,sig)

    return sig

def UPtrendsell(stdatas,buysk):

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
        sellsiga=np.array([sigone1,sigone1,sigone1*2,sigzero1,stdatas[hoji[0,i]::,4,hoji[2,i]]-stdatas[hoji[0,i]::,7,hoji[2,i]]]).T
        sellsig=sellsiga
        sigm1=sellsk(stdatas,sellsig,sigm1,hoji,i,stdatas1day)
    return sigm1