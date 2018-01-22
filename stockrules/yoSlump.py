#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 17:05:23 2018

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

def Slump(stdatas):

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
    #５０日「連続で安値が期間安値７５より大きい
    min5=stmin(stdatas[:,3,:],[5])
    #終値の位置７５が０と３０の範囲内
    #欠損値のあるところはシグナル出ないようにする
    maxperiod=[5]
    #nosigiが0なら欠損値なし
    
    nosig=np.tile(n010(stc3(min5,maxperiod)),(1,7,1))+np.tile(n010(stc3(stdatas[:,4,:],[20])),(1,7,1))
    #移動平均剥離率21
    ave15h=((n010(stdatas[:,4,:])/n010(stmean(stdatas[:,4,:],[15]))-1)*100)
    #始値→終値
    OCgap=np.tile(n010(((stdatas[:,4,:]/stdatas[:,1,:]-1)*100)),(1,7,1))
    #前日との比率
    hiritu=((stdatas/stdatas1p-1)*100)
    #期間安値高値の中間値
    median2=((stmax(stdatas[:,2,:],[2])+stmin(stdatas[:,3,:],[2]))/2)
    median5=((stmax(stdatas[:,2,:],[5])+stmin(stdatas[:,3,:],[5]))/2)
    #暴落銘柄カウント
    #sigの大きさ決め
    slump1=np.ones((stdatas.shape[0],7,stdatas.shape[2]))
    slump0=np.zeros((stdatas.shape[0],7,stdatas.shape[2]))

    stslump=np.where(((np.tile(n010(hiritu[:,4,:]),(1,7,1))>5)&
                     (OCgap>5)&
                     (np.tile(n010(median2),(1,7,1))<np.tile(n010(median5*0.9),(1,7,1))))
                     ,slump1,slump0)
    stslumpi=np.tile(n011(np.sum(stslump[:,0,:],axis=1)),(1,7,stdatas.shape[2]))
    #[sig,買い売り、成り行き指値、指値価格、逆指値価格]
    #期間安値（５０）−６％で指値
    sig1=np.concatenate([sigonec,sigzeroc,sigzeroc,sigzeroc,sigzeroc,n010(stdatas[:,4,:]),ave15h],axis=1)
    sig=np.concatenate([sigzeroc,sigzeroc,sigzeroc,sigzeroc,sigzeroc,sigzeroc,sigzeroc],axis=1)
    #stopLはstdatas[:,5,:],stophはstdatas[:,6,:]
    #１→始値、２→高値、３→安値、４→終値
    sig=np.where(((stslump==slump1)&
                 (stslumpi>50)&
                 (nosig==0)&
                 (volave>50))
                ,sig1,sig)
    return sig


def Slumpsell(stdatas,buysk):

    #sellsigは売りシグナル、この３、４列目に保有日数と執行価格を入れる
    #sig作成用
    codes=stdatas.shape[2]
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
        zdate=alldate-hoji[0,i]
        #保有日数
        holddays=(np.arange(zdate))
        #sig作成用
        sigzero1=np.zeros(zdate)
        sigone1=np.ones(zdate)
    
        #移動平均２５と安値の２日連続文
        #売りシグナル計算→執行関数とは別
        #１→始値、２→高値、３→安値、４→終値
        #保有日数５日以内＆２日連続で安値が移動平均２５より小さい
        #sellsigの大きさ
        sellsig=np.zeros((zdate,5))
    
        #保有日数3日より大きい
        sellsiga=np.array([sigone1,sigone1,sigone1*2,sigzero1,stdatas[hoji[0,i]::,4,hoji[2,i]]-stdatas[hoji[0,i]::,7,hoji[2,i]]]).T
        sellsig=np.where(np.tile(n01(holddays),(1,5))>3
                        ,sellsiga,sellsig)
        sigm1=sellsk(stdatas,sellsig,sigm1,hoji,i,stdatas1day)
    return sigm1
