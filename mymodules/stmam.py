#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 31 13:18:32 2017

@author: yosu
"""

import yopath
import numpy as np
import pandas as pd

#安値、終値とかそのレベルで与える→stdafは時間と株種類の2次元
def stmax(stdaf,periods):
    stmax=np.zeros(stdaf.shape)
    for code in range(stdaf.shape[1]):
        for period in periods:
            pdst=pd.Series(list(stdaf[:,code]))
            stmax[:,code]=np.array(pdst.rolling(window=period, min_periods=period).max())
    stmax[np.isnan(stmax)]=0
    return stmax

def stmean(stdaf,periods):
    stmean=np.zeros(stdaf.shape)
    for code in range(stdaf.shape[1]):
        for period in periods:
            pdst=pd.Series(list(stdaf[:,code]))
            stmean[:,code]=np.array(pdst.rolling(window=period, min_periods=period).mean())
    stmean[np.isnan(stmean)]=0
    return stmean

def stmin(stdaf,periods):
    stmin=np.zeros(stdaf.shape)
    for code in range(stdaf.shape[1]):
        for period in periods:
            pdst=pd.Series(list(stdaf[:,code]))
            stmin[:,code]=np.array(pdst.rolling(window=period, min_periods=period).min())
    stmin[np.isnan(stmin)]=0
    return stmin

def stc(st1,st2,periods):
    #stc1>stc2の日数をカウント
    #cnumc=移動平均（１００）より大きいとき1
    #cnumc=np.zeros((stsize))
    cnumc=np.where(st1 > st2,1,0)
    #カウント結果作成　cnumcと同サイズ、期間追加時変更だが、
    #平均日数とカウント日数で膨大になるから関数化推奨
    #過去のデータから計算してることを確認
    stc=np.zeros((cnumc.shape))
    for code in range(st1.shape[1]):
        for period in periods:
            cnumcs=pd.Series(cnumc[:,code])
            stc[:,code]=np.array(cnumcs.rolling(window=period, min_periods=period).sum())
    return stc

def stc2(st1,st2,periods,eq):
    #stc1>stc2の日数をカウント
    #cnumc=移動平均（１００）より大きいとき1
    #cnumc=np.zeros((stsize))
    cnumc=np.where(eval(eq),1,0)
    #カウント結果作成　cnumcと同サイズ、期間追加時変更だが、
    #平均日数とカウント日数で膨大になるから関数化推奨
    #過去のデータから計算してることを確認
    stc=np.zeros((cnumc.shape))
    for code in range(st1.shape[1]):
        for period in periods:
            cnumcs=pd.Series(cnumc[:,code])
            stc[:,code]=np.array(cnumcs.rolling(window=period, min_periods=period).sum())
    return stc

def stc3(st1,periods):
    #値が０の時カウント→株価入ってない
    #cnumc=np.zeros((stsize))
    st2=np.zeros((st1.shape))
    cnumc=np.where(st1==st2,1,0)
    #カウント結果作成　cnumcと同サイズ、期間追加時変更だが、
    #平均日数とカウント日数で膨大になるから関数化推奨
    #過去のデータから計算してることを確認
    stc=np.zeros((cnumc.shape))
    for code in range(st1.shape[1]):
        for period in periods:
            cnumcs=pd.Series(cnumc[:,code])
            stc[:,code]=np.array(cnumcs.rolling(window=period, min_periods=period).sum())
    stc[np.isnan(stc)]=1
    return stc

def stc4(st1,st2,st3,periods):
    #stc1>stc2の日数をカウント
    #cnumc=移動平均（１００）より大きいとき1
    #cnumc=np.zeros((stsize))
    cnumc=np.where((st1 >= st2)&
                   (st1<=st3),1,0)
    #カウント結果作成　cnumcと同サイズ、期間追加時変更だが、
    #平均日数とカウント日数で膨大になるから関数化推奨
    #過去のデータから計算してることを確認
    stc=np.zeros((cnumc.shape))
    for code in range(st1.shape[1]):
        for period in periods:
            cnumcs=pd.Series(cnumc[:,code])
            stc[:,code]=np.array(cnumcs.rolling(window=period, min_periods=period).sum())
    return stc

def stc5(st1):
    #シグナルカウント
    #cnumc=np.zeros((stsize))
    st2=np.ones((st1.shape))
    cnumc=np.where(st1==st2,1,0)
    stc=np.sum(cnumc,axis=1)
    return stc