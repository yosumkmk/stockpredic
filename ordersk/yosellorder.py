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
from ta_yo import *
from newaxis1 import *
from yoRSI import *
from yoRSI2 import *
from yoRSI3 import *
from yobfkai3 import *
from yobfkai41 import *
from yoDips import *
from yoDips2 import *
from yoDips21 import *
from yoDips31 import *
from yoDips31sasi import *
from yoDipssasi import *
from yoDipstr import *
from yoHiddenLine import *
from yoSlump import *
from yoLow22 import *
from yobuysk import *
from yosellsk import *
from stmam import *

def sellorder(stdatas,exesignal,sellsigs):
    #exesignalに売り執行が入ってないものをピックアップ
    #↑に限定すれば、「：、：、ccode,rules]に入る→同じルールで買ってるものはどうする？
    #[orderNo,code,,price,amount,///]にする
    #hoji[(日付、コード、ルール）、（シグナル執行順＝合計が買い回数)）
    #買い執行＆売り執行なし
    hoji=deepcopy(np.array(np.where((exesignal[:,0,:,:][:,np.newaxis,:,:]==1)&(exesignal[:,2,:,:][:,np.newaxis,:,:]==0))))
    #翌日,2日後の価格（買いシグナル基準）
    #未来の参照はマイナスを入れる
    stdatas1day=np.roll(stdatas,-1,axis=0)
    #保有日数（買執行日から何日か、買い執行日は一日目）,最低保有日数はhdate+2
    ordersell=np.zeros((len(hoji[0,:]),5))
    for i in range(len(hoji[0,:])):
        
        #注文別、[sig,売買、成指区分、指値価格、逆指値価格]
        #sellsigsは時間、上の５種類、株コード
        #sellsigの大きさは(zdate[i],5)
        #執行シグナルごとに売りシグナル出すから、株コードと執行シグナル別に
        sellsig=eval(sellsigs)(stdatas1day,hoji,i)
        ordersell[i,0]=hoji[2,i]
        ordersell[i,1]=sellsig[-1,1]
        ordersell[i,2]=sellsig[-1,2]
        ordersell[i,3]=sellsig[-1,3]
        ordersell[i,4]=sellsig[-1,4]

    return ordersell