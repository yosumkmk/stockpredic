#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 18:39:01 2018

@author: yosu
"""

import sys
import os

sd = os.path.dirname(__file__)
sys.path.append(sd)
files = os.listdir(sd)
for f in files :
    if os.path.isdir(os.path.join(sd, f)):
        sys.path.append(os.path.join(sd,f))
    