#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 16:02:23 2017

@author: yosu
"""
import yopath
import numpy as np
import math
import talib as ta
from copy import deepcopy
from newaxis1 import *
from ta_yo import *
from newaxis1 import *
from stmam import *

def sellsk(stdatas,sellsig,sigm1,hoji,i,stdatas1day):

    #sellsigは売りシグナル、この３、４列目に保有日数と執行価格を入れる
    #sig作成用
    #sigzerocode,sigonecode
    #残り日数
    alldate=stdatas.shape[0]
    zdate=alldate-hoji[0,:]

    #翌日,2日後の価格（買いシグナル基準）
    #未来の参照はマイナスを入れる
    #保有日数（買執行日から何日か、買い執行日は一日目）,最低保有日数はhdate+2

    #注文別、[sig,売買、成指区分、指値価格、逆指値価格]
    #sellsigsは時間、上の５種類、株コード
    #sellsigの大きさは(zdate[i],5)
    #執行シグナルごとに売りシグナル出すから、株コードと執行シグナル別に

    #sig作成用
    sigone1=np.ones(zdate[i])
    #執行時[執行sig,執行価格]
    sellsigsk=np.zeros((zdate[i],2))
    #売り執行用[シグナル＝１、執行価格]
    #成り行き
    sellsigska=np.array([sigone1,stdatas1day[hoji[0,i]::,1,hoji[2,i]]]).T
                       
    sellsigsk=np.where(((np.tile(n01(sellsig[:,0]),(1,2))==1)
                        &(np.tile(n01(sellsig[:,2]),(1,2))==0)
                        &((hoji[0,i]+2)<=stdatas.shape[0])
                        &(np.tile(n01(stdatas1day[hoji[0,i]::,4,hoji[2,i]]),(1,2))>0))
                       ,sellsigska,sellsigsk)
    #指値１（指値＜高値　＆　指値＞始値）
    sellsigska=np.array([sigone1,sellsig[:,3]]).T
                       
    sellsigsk=np.where(((np.tile(n01(sellsig[:,0]),(1,2))==1)
                        &(np.tile(n01(sellsig[:,2]),(1,2))==1)
                        &(np.tile(n01(sellsig[:,3]),(1,2))<np.tile(n01(stdatas1day[hoji[0,i]::,2,hoji[2,i]]),(1,2)))
                        &(np.tile(n01(sellsig[:,3]),(1,2))>np.tile(n01(stdatas1day[hoji[0,i]::,1,hoji[2,i]]),(1,2)))
                        &((hoji[0,i]+2)<=stdatas.shape[0])
                        &(np.tile(n01(stdatas1day[hoji[0,i]::,4,hoji[2,i]]),(1,2))>0))
                       ,sellsigska,sellsigsk)
    #指値2（指値<始値）                                   
    sellsigska=np.array([sigone1,stdatas1day[hoji[0,i]::,1,hoji[2,i]]]).T
                       
    sellsigsk=np.where(((np.tile(n01(sellsig[:,0]),(1,2))==1)
                        &(np.tile(n01(sellsig[:,2]),(1,2))==1)
                        &(np.tile(n01(sellsig[:,3]),(1,2))<np.tile(n01(stdatas1day[hoji[0,i]::,1,hoji[2,i]]),(1,2)))
                        &((hoji[0,i]+2)<=stdatas.shape[0])
                        &(np.tile(n01(stdatas1day[hoji[0,i]::,4,hoji[2,i]]),(1,2))>0))
                       ,sellsigska,sellsigsk)
                               
    #逆指値＞始値(成り行き)
    sellsigska=np.array([sigone1,stdatas1day[hoji[0,i]::,1,hoji[2,i]]]).T
    #正確には、執行価格＝始値にはならない→逆指値なので
    sellsigsk=np.where(((np.tile(n01(sellsig[:,0]),(1,2))==1)
                        &(np.tile(n01(sellsig[:,2]),(1,2))==2)
                        &(np.tile(n01(sellsig[:,3]),(1,2))==0)
                        &(np.tile(n01(sellsig[:,4]),(1,2))>np.tile(n01(stdatas1day[hoji[0,i]::,1,hoji[2,i]]),(1,2)))
                        &((hoji[0,i]+2)<=stdatas.shape[0])
                        &(np.tile(n01(stdatas1day[hoji[0,i]::,4,hoji[2,i]]),(1,2))>0))
                       ,sellsigska,sellsigsk)
    #逆指値<始値,逆指値＞安値(成り行き)
    sellsigska=np.array([sigone1,sellsig[:,4]]).T
    #正確には、執行価格＝始値にはならない→逆指値なので
    sellsigsk=np.where(((np.tile(n01(sellsig[:,0]),(1,2))==1)
                        &(np.tile(n01(sellsig[:,2]),(1,2))==2)
                        &(np.tile(n01(sellsig[:,3]),(1,2))==0)
                        &(np.tile(n01(sellsig[:,4]),(1,2))<np.tile(n01(stdatas1day[hoji[0,i]::,1,hoji[2,i]]),(1,2)))
                        &(np.tile(n01(sellsig[:,4]),(1,2))>np.tile(n01(stdatas1day[hoji[0,i]::,3,hoji[2,i]]),(1,2)))
                        &((hoji[0,i]+2)<=stdatas.shape[0])
                        &(np.tile(n01(stdatas1day[hoji[0,i]::,4,hoji[2,i]]),(1,2))>0))
                       ,sellsigska,sellsigsk)
    #逆指値＆指値(高値＞指値、安値＜逆指値で包括可能)
    sellsigska=np.array([sigone1,sellsig[:,3]]).T
    #正確には、執行価格＝始値にはならない→逆指値なので
    sellsigsk=np.where(((np.tile(n01(sellsig[:,0]),(1,2))==1)
                        &(np.tile(n01(sellsig[:,2]),(1,2))==2)
                        &(np.tile(n01(sellsig[:,3]),(1,2))>0)
                        &(np.tile(n01(sellsig[:,3]),(1,2))<np.tile(n01(stdatas1day[hoji[0,i]::,2,hoji[2,i]]),(1,2)))
                        &(np.tile(n01(sellsig[:,4]),(1,2))>np.tile(n01(stdatas1day[hoji[0,i]::,3,hoji[2,i]]),(1,2)))
                        &((hoji[0,i]+2)<=stdatas.shape[0])
                        &(np.tile(n01(stdatas1day[hoji[0,i]::,4,hoji[2,i]]),(1,2))>0))
                       ,sellsigska,sellsigsk)

    #買ったばかりで売りシグナル出ないタイムミングは−１
    #sellsigsk=np.where(((np.tile(n01(np.arange(len(sigzero1))),(1,2))+hoji[0,i])
    #                    >stdatas.shape[0])
    #                   ,sigm1,sellsigsk)
    
    #保有日数未満は０ 0:1で買い執行日のみ保有
    sellsigsk[0,:]=0
    #買い執行日に売りシグナルはオーケー、買いシグナルの日には出せないので０↑
    #執行日の前日（執行される売りシグナルが入っている）
    z=np.array(deepcopy(np.where(sellsigsk[:,0]==1)))
    if z.sum()>=1:
        sigm1[hoji[0,i],2,hoji[2,i]]=z[0,0]+1
        sigm1[hoji[0,i],3,hoji[2,i]]=sellsigsk[z[0,0],1]
    return sigm1