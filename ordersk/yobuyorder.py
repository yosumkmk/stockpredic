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
from ta_yo import *
from newaxis1 import *
from yoRSI import *

def buyorder(buysignal):
    #exesignalに売り執行が入ってないものをピックアップ
    #↑に限定すれば、「：、：、ccode,rules]に入る→同じルールで買ってるものはどうする？
    #[orderNo,code,,price,amount,///]にする
    #hoji[(日付、コード、ルール）、（シグナル執行順＝合計が買い回数)）
    #買い執行＆売り執行なし
    buyNo=deepcopy(np.array(np.where(buysignal[-1,0,:,:]==1)))
    orderbuy=np.zeros((len(buyNo[0,:]),5))
    for i in np.arange(len(buyNo[0,:])):
        orderbuy[i,0]=buyNo[0,i]
        orderbuy[i,1]=buysignal[-1,1,buyNo[0,i],buyNo[1,i]]
        orderbuy[i,2]=buysignal[-1,2,buyNo[0,i],buyNo[1,i]]
        orderbuy[i,3]=buysignal[-1,3,buyNo[0,i],buyNo[1,i]]
        orderbuy[i,4]=buysignal[-1,4,buyNo[0,i],buyNo[1,i]]
    return orderbuy