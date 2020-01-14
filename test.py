# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 20:55:43 2018

@author: 罗锦林
"""
import numpy as np

trade_records = []

record0 = [0,'luojinlin',37]
record1 = [1,'lvxiaocheng',31]
record2 = [2,'luoyizhi',1]

trade_records.append(record0)
trade_records.append(record1)
trade_records.append(record2)

np.savetxt('family.csv', trade_records, delimiter=',', fmt='%s')



