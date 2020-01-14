# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 21:32:44 2018

@author: 罗锦林
"""

#import math
import pandas as pd
import numpy as np
import copy
#import tushare as ts
from datetime import datetime
from matplotlib.dates import date2num
import matplotlib.pyplot as plt
import stockstats

from matplotlib.dates import DateFormatter 
from matplotlib.dates import DayLocator 
from matplotlib.dates import MonthLocator 

from matplotlib.finance import candlestick_ohlc
import weeklyLineJudge0228 as weekLine

def timestr2num(s): 
    strptime    = datetime.strptime(s.decode('ascii'), "%Y/%m/%d  %H:%M")
#    date1      = strptime.date()
#    date       = date2num(date1)
    timestamp   = strptime.timestamp()
    return timestamp

def datestr2num1(s): 
    date1       = datetime.strptime(s.decode('ascii'), "%Y/%m/%d  %H:%M").date()
    date        = date2num(date1)
    return date
########################################## 一、折线图 ########################################
# 折线图函数：getLineChart(quotes) ###########################################################
def getLineChart(quotes):
    quotes_fg               = [[],[],[],[]]
    #2.1 找到极值点 #################################################################
    quotes_fg0_date         = []         #峰谷的日期
    quotes_fg0_price        = []        #峰谷的价格
    quotes_fg0_index        = []        #峰谷在原时间序列中的位置
    quotes_fg0_direction    = []    #值为1表示波峰，值为0表示波谷
    for i in range( len(quotes) )[1:-1]:
        #波峰：
        if quotes[i][4]>quotes[i-1][4] and quotes[i][4]>quotes[i+1][4]:
            quotes_fg0_date.append(quotes[i][0])
            quotes_fg0_price.append(quotes[i][4])
            quotes_fg0_index.append(i)
            quotes_fg0_direction.append(1)
        #波谷：
        if quotes[i][4]<quotes[i-1][4] and quotes[i][4]<quotes[i+1][4]:
            quotes_fg0_date.append(quotes[i][0])
            quotes_fg0_price.append(quotes[i][4])
            quotes_fg0_index.append(i)
            quotes_fg0_direction.append(0)

    #2.2 补充缺失极值点 #################################################################
    #情况1：两个波谷之间，当两个紧挨着的收盘价为最高价时，该波谷极值会丢失，在此需以第一个最高价补上
    #情况2：两个波峰之间，当两个紧挨着的收盘价为最低价时，该波峰极值会丢失，在此需以第一个最低价补上
    for i in range( len(quotes_fg0_direction) )[0:-1]:
        if quotes_fg0_direction[i] == quotes_fg0_direction[i+1] == 0: #情况1：两个波谷连在了一起
            quotes_wave_begin   = quotes_fg0_index[i]
            quotes_wave_end     = quotes_fg0_index[i+1]
            quotes_wave         = []
            for j in range(quotes_wave_begin, quotes_wave_end + 1):
                quotes_wave.append(quotes[j][4])
            quotes_wave_f_index = np.argmax(quotes_wave)
            quotes_f_index      = quotes_wave_begin + quotes_wave_f_index #新波峰在quotes的序号
            #在quotes_fg0系列数组中插入新波峰：
            quotes_fg0_date.insert(i+1, quotes[quotes_f_index][0])
            quotes_fg0_price.insert(i+1, quotes[quotes_f_index][4])
            quotes_fg0_index.insert(i+1, quotes_f_index)
            quotes_fg0_direction.insert(i+1, 1)
        elif quotes_fg0_direction[i] == quotes_fg0_direction[i+1] == 1:#情况2：两个波峰连在了一起
            quotes_wave_begin   = quotes_fg0_index[i]
            quotes_wave_end     = quotes_fg0_index[i+1]
            quotes_wave         = []
            for j in range(quotes_wave_begin, quotes_wave_end + 1):
                quotes_wave.append(quotes[j][4])
            quotes_wave_g_index = np.argmin(quotes_wave)
            quotes_g_index      = quotes_wave_begin + quotes_wave_g_index #新波谷在quotes的序号
            #在quotes_fg0系列数组中插入新波峰：
            quotes_fg0_date.insert(i+1, quotes[quotes_g_index][0])
            quotes_fg0_price.insert(i+1, quotes[quotes_g_index][4])
            quotes_fg0_index.insert(i+1, quotes_g_index)
            quotes_fg0_direction.insert(i+1, 0)
            
    #2.3 修正极值点 #################################################################
    
    #第1步：对于波峰（波谷），用本周最大值（最小值）代替收盘价
    j = 0 # 记录quotes_fg0_price中元素的坐标
    for i in range( len(quotes_fg0_index) ):
        fg0_index = quotes_fg0_index[i]
        #本极值点为波谷，用最低价代替收盘价
        if quotes_fg0_direction[i] == 0:
            quotes_fg0_price[j] = quotes[fg0_index][3]
        #开盘价<收盘价，本极值点为波峰，用最高价代替收盘价
        if quotes_fg0_direction[i] == 1:
            quotes_fg0_price[j] = quotes[fg0_index][2]
        j = j + 1 # quotes_fg0_price下一个元素的坐标
    
    #第2步：根据规则，用新峰谷时间点代替部分原峰谷时间点
    quotes_fg_date      = copy.deepcopy(quotes_fg0_date)
    quotes_fg_price     = copy.deepcopy(quotes_fg0_price)
    quotes_fg_index     = copy.deepcopy(quotes_fg0_index)
    quotes_fg_direction = copy.deepcopy(quotes_fg0_direction)    #值为1表示波峰，值为0表示波谷
    
    #先根据波峰确定波次，调整波谷
    for j in range(len(quotes_fg_index)):
        if quotes_fg_direction[j] == 1 and  j + 2 < len(quotes_fg_index):#本波峰后面还有一个波峰，能构成完整波次,才在本波次内调整波谷
            quotes_fwave_begin  = quotes_fg_index[j]    #第一个波峰在quotes的序号
            quotes_fwave_end    = quotes_fg_index[j+2]  #第二个波峰在quotes的序号
            quotes_wave_low     = []
            for i in range(quotes_fwave_begin, quotes_fwave_end + 1):
                quotes_wave_low.append(quotes[i][3])
            quotes_wave_low_min_index   = np.argmin(quotes_wave_low)
            quotes_wave_low_min         = quotes_wave_low[ quotes_wave_low_min_index ]
            quotes_g_index              = quotes_fwave_begin + quotes_wave_low_min_index #新波谷在quotes的序号
            if quotes_wave_low_min < quotes_fg_price[j+1]:
                # 用新波谷代替旧波谷:
                quotes_fg_date[j+1]     = quotes[quotes_g_index][0]
                quotes_fg_price[j+1]    = quotes[quotes_g_index][3]
                quotes_fg_index[j+1]    = quotes_g_index

    #再根据波谷确定波次，调整波峰
    for j in range(len(quotes_fg_index)):
        if quotes_fg_direction[j] == 0 and  j + 2 < len(quotes_fg_index):#本波谷后面还有一个波谷，能构成完整波次,才在本波次内调整波峰
            quotes_gwave_begin  = quotes_fg_index[j]    #第一个波谷在quotes的序号
            quotes_gwave_end    = quotes_fg_index[j+2]  #第二个波谷在quotes的序号
            quotes_wave_high    = []
            for i in range(quotes_gwave_begin, quotes_gwave_end + 1):
                quotes_wave_high.append(quotes[i][2])
            quotes_wave_high_max_index  = np.argmax(quotes_wave_high)
            quotes_wave_high_max        = quotes_wave_high[ quotes_wave_high_max_index ]
            quotes_f_index              = quotes_gwave_begin + quotes_wave_high_max_index #新波峰在quotes的序号
            if quotes_wave_high_max > quotes_fg_price[j+1]:
                # 用新波峰代替旧波峰:
                quotes_fg_date[j+1]     = quotes[quotes_f_index][0]
                quotes_fg_price[j+1]    = quotes[quotes_f_index][2]
                quotes_fg_index[j+1]    = quotes_f_index

    quotes_fg[0] = copy.deepcopy( quotes_fg_date )
    quotes_fg[1] = copy.deepcopy( quotes_fg_price )
    quotes_fg[2] = copy.deepcopy( quotes_fg_index )
    quotes_fg[3] = copy.deepcopy( quotes_fg_direction )

    return quotes_fg

######################################### 二、图形化显示K线 ########################################
###################################################################################################
#alldays = DayLocator()  
#months = MonthLocator() 
#month_formatter = DateFormatter("%b %Y")  
#fig = plt.figure( "Week {0}'s 15M Line".format(week_N), figsize=(18, 9) ) 
#ax = fig.add_subplot(121)
#ax = fig.add_subplot(111)
#ax.xaxis.set_major_locator(months) 
#ax.xaxis.set_minor_locator(alldays) 
#ax.xaxis.set_major_formatter(month_formatter) 

#plt.title("Week {0}'s 15M Line".format(week_N) )

#plt.xlabel("15M Time")
#plt.ylabel("Price")

#将两端端点加入波峰、波谷图中：
#if quotes_fg15m[3][0] == 0: #第一个极值为波谷,将quotes_15m_new第一个蜡烛图的最高价看成波峰加入数组
#    quotes_fg15m[0] = np.append(quotes_15m_new[0][0], quotes_fg15m[0])#date
#    quotes_fg15m[1] = np.append(quotes_15m_new[0][2], quotes_fg15m[1])#price，最高价做波峰
#    quotes_fg15m[2] = np.append(quotes_15m_new[0][0], quotes_fg15m[2])#index，峰谷在原系列中的位置
#    quotes_fg15m[3] = np.append(1, quotes_fg15m[3])#direction
#elif quotes_fg15m[3][0] == 1: #第一个极值为波峰
#    quotes_fg15m[0] = np.append(quotes_15m_new[0][0], quotes_fg15m[0])#date
#    quotes_fg15m[1] = np.append(quotes_15m_new[0][3], quotes_fg15m[1])#price，最低价做波谷
#    quotes_fg15m[2] = np.append(quotes_15m_new[0][0], quotes_fg15m[2])#index，峰谷在原系列中的位置
#    quotes_fg15m[3] = np.append(0, quotes_fg15m[3])#direction
#if quotes_fg15m[3][-1] == 0: #最后一个极值为波谷
#    quotes_fg15m[0] = np.append(quotes_fg15m[0], quotes_15m_new[-1][0])#date
#    quotes_fg15m[1] = np.append(quotes_fg15m[1], quotes_15m_new[-1][2])#price，最高价做波峰
#    quotes_fg15m[2] = np.append(quotes_fg15m[2], quotes_15m_new[-1][0])#index，峰谷在原系列中的位置
#    quotes_fg15m[3] = np.append(quotes_fg15m[3], 1)#direction
#elif quotes_fg15m[3][-1] == 1: #最后一个极值为波峰
#    quotes_fg15m[0] = np.append(quotes_fg15m[0], quotes_15m_new[-1][0])#date
#    quotes_fg15m[1] = np.append(quotes_fg15m[1], quotes_15m_new[-1][3])#price，最高价做波峰
#    quotes_fg15m[2] = np.append(quotes_fg15m[2], quotes_15m_new[-1][0])#index，峰谷在原系列中的位置
#    quotes_fg15m[3] = np.append(quotes_fg15m[3], 0)#direction

#图形1--K线图：
#candlestick_ohlc(ax,quotes_15m_new,width=0.2, colorup='r',colordown='g') 

#图形2--折线图：
#plt.plot(quotes_fg15m[0], quotes_fg15m[1], 'r', lw=2.0, label='Line chart') #画出峰谷折线图
#ax.scatter(quotes_fg15m[0][1:-1], quotes_fg15m[1][1:-1], alpha=0.5) #除去两端端点，画出峰谷散点图

##图形3--MA指标：
#plt.plot(stockStat.index.values, stockStat.close_5_sma.values, 'k', lw=1.0, label='MA5') # MA5
#plt.plot(stockStat.index.values, stockStat.close_10_sma.values, 'c', lw=1.0, label='MA10') # MA10
#plt.plot(stockStat.index.values, stockStat.close_20_sma.values, 'r', lw=1.0, label='MA20') # MA20

##图形4--BOLL指标：
#plt.plot(stockStat.index.values, stockStat.boll.values, 'b--', lw=1.0, label='Line K') # boll线
#plt.plot(stockStat.index.values, stockStat.boll_ub.values, 'r--', lw=1.0, label='Line D') # boll_ub线
#plt.plot(stockStat.index.values, stockStat.boll_lb.values, 'r--', lw=1.0, label='Line J') # boll_lb线

##图形5--KDJ指标：
#ax2 = fig.add_subplot(221)
#plt.plot(stockStat.index.values, stockStat.kdjk.values, 'b', lw=1.0, label='Line K') # K线
#plt.plot(stockStat.index.values, stockStat.kdjd.values, 'c', lw=1.0, label='Line D') # D线
#plt.plot(stockStat.index.values, stockStat.kdjj.values, 'red', lw=1.0, label='Line J') #J线

##图形6--MACD指标：
##ax2 = fig.add_subplot(221)
#plt.plot(stockStat.index.values, stockStat.macd.values, 'b', lw=1.0, label='macd') # macd
#plt.plot(stockStat.index.values, stockStat.macds.values, 'r', lw=1.0, label='macds') # macds
#plt.bar(stockStat.index.values, stockStat.macdh.values, color='b') # macdh

#plt.savefig("RM801_Week {0}'s 15M_MACD.png".format(week_N))
#for i in range(len(quotes_15m_new)):
#    plt.text(stockStat.index.values[i], stockStat.high.values[i], stockStat.index.values[i], alpha=0.5)
    
#plt.show()

########################################## 三、定义函数(未开始) ########################################
###############################################################################################

# 1、MA指标函数：getMATrend(stockStat, M15_NO)#############################################
# 判断15分钟线第15M_NO个节点的MA指标多空， maTrend = -1表示指标空，=1表示指标多，=0表示无法进行多空判断
def getMATrend(stockStat, M15_NO):
    maTrend    = 0
    if stockStat.close_5_sma.values[M15_NO] >= stockStat.close_10_sma.values[M15_NO] >= stockStat.close_20_sma.values[M15_NO]:
        maTrend = 1 #MA多
    elif stockStat.close_5_sma.values[M15_NO] <= stockStat.close_10_sma.values[M15_NO] <= stockStat.close_20_sma.values[M15_NO]:
        maTrend = -1 #MA空
    
    return maTrend

# 2、BOLL指标函数： getBOLLTrend(stockStat, M15_NO, latestPrice, tick)#############################################
# 判断15分钟线第15M_NO个节点的BOLL指标多空， bollTrend = -1表示指标空，=1表示指标多，=0表示无法进行多空判断
def getBOLLTrend(stockStat, M15_NO, latestPrice, tick):
    bollTrend    = 0
    if latestPrice >= stockStat.boll_ub.values[ M15_NO ] + 3*tick:
        bollTrend   = 1 #MA多
    elif latestPrice <= stockStat.boll_lb.values[ M15_NO ] - 3*tick:
        bollTrend   = -1 #MA空
    
    return bollTrend

# 3、KDJ指标函数： getKDJTrend(stockStat, M15_NO)#############################################
# 判断15分钟线第15M_NO个节点的KDJ指标多空， kdjTrend = -1表示指标空，=1表示指标多，=0表示无法进行多空判断
def getKDJTrend(stockStat, M15_NO):
    kdjTrend        = 0
    if stockStat.kdjk.values[M15_NO] <= 35:
        kdjTrend    = 1
    elif stockStat.kdjk.values[M15_NO] >= 65:
        kdjTrend    = -1
    
    return kdjTrend

# 4、MACD指标函数： getMACDTrend(stockStat, M15_NO)#############################################
# 判断15分钟线第15M_NO个节点的MACD指标多空， macdTrend = -1表示指标空，=1表示指标多，=0表示无法进行多空判断
def getMACDTrend(stockStat, M15_NO):
    macdTrend       = 0
    if stockStat.macdh.values[M15_NO] >= stockStat.macdh.values[M15_NO - 1] >= stockStat.macdh.values[M15_NO - 2]:
        macdTrend   = 1
    elif stockStat.macdh.values[M15_NO] <= stockStat.macdh.values[M15_NO - 1] <= stockStat.macdh.values[M15_NO - 2]:
        macdTrend   = -1
    
    return macdTrend

# 5、K线突破函数： getKBreakTrend()#############################################
# 以K线突破函数判断15分钟线第M15_NO个节点的多空， kBreakTrend = -1表示2个指标空，=1表示2个指标多，
# = -2表示3个及以上指标空，=2表示3个及以上指标多，=0表示无法进行多空判断
def getKBreakTrend(stockStat, quotes_fg15m, M15_NO, latestPrice, tick):
    kBreakTrend = 0
    maTrend     = getMATrend(stockStat, M15_NO)
    bollTrend   = getBOLLTrend(stockStat, M15_NO, latestPrice, tick)
    kdjTrend    = getKDJTrend(stockStat, M15_NO)
    macdTrend   = getMACDTrend(stockStat, M15_NO)
    TrendList   = [maTrend, bollTrend, kdjTrend, macdTrend]
    
    fg_index    = np.max(np.where( quotes_fg15m[0] < M15_NO )) # fg_index是离 week_N 最近的波峰/波谷下标
    f_index     = -1 #最近波峰
    g_index     = -1 #最近波谷
    if quotes_fg15m[3][fg_index] == 1:
        f_index = fg_index
        g_index = fg_index - 1
    elif quotes_fg15m[3][fg_index] == 0:
        g_index = fg_index
        f_index = fg_index - 1

    if latestPrice > quotes_fg15m[1][f_index] + 1*tick and TrendList.count(1) == 2:
        kBreakTrend = 1 #K线突破多
    elif latestPrice > quotes_fg15m[1][f_index] + 1*tick and TrendList.count(1) >= 3:
        kBreakTrend = 2 #K线突破多
    elif latestPrice < quotes_fg15m[1][g_index] - 1*tick and TrendList.count(-1) == 2:
        kBreakTrend = -1 #K线突破空
    elif latestPrice < quotes_fg15m[1][g_index] - 1*tick and TrendList.count(-1) >= 3:
        kBreakTrend = -2 #K线突破空
        
    quote_f15m = getLatestF(quotes_fg15m, M15_NO, latestPrice)
    quote_g15m = getLatestG(quotes_fg15m, M15_NO, latestPrice)
    
    if latestPrice > quote_f15m[1] + 1*tick and TrendList.count(1) >= 2:
        kBreakTrend = 1 #K线突破多
    elif latestPrice < quote_g15m[1] - 1*tick and TrendList.count(-1) >= 2:
        kBreakTrend = -1 #K线突破空
        
    return kBreakTrend

# 5_1、最近波峰的价格： getLatestF()#############################################
def getLatestF(quotes_fg15m, M15_NO, latestPrice):
    fg_index    = np.max(np.where( quotes_fg15m[0] < M15_NO )) # fg_index是离 week_N 最近的波峰/波谷下标
    f_index     = -1 #最近波峰
    if quotes_fg15m[3][fg_index] == 1:
        f_index = fg_index
    elif quotes_fg15m[3][fg_index] == 0:
        f_index = fg_index - 1
        
    quote_f15m  = [quotes_fg15m[0][f_index], quotes_fg15m[1][f_index], quotes_fg15m[2][f_index], quotes_fg15m[3][f_index]]

    return quote_f15m

# 5_2、最近波峰的价格： getLatestG()#############################################
def getLatestG(quotes_fg15m, M15_NO, latestPrice):
    fg_index    = np.max(np.where( quotes_fg15m[0] < M15_NO )) # fg_index是离 week_N 最近的波峰/波谷下标
    g_index     = -1 #最近波谷
    if quotes_fg15m[3][fg_index] == 1:
        g_index = fg_index - 1
    elif quotes_fg15m[3][fg_index] == 0:
        g_index = fg_index
        
    quote_g15m  = [quotes_fg15m[0][g_index], quotes_fg15m[1][g_index], quotes_fg15m[2][g_index], quotes_fg15m[3][g_index]]

    return quote_g15m

# 6、K线突破函数： getKReverseBull()、getKReverseBear()#############################################
# 因为据需求文档，可同时满足K线反转多、K线反转空条件，所以将判断函数拆开成两个。
# 以K线反转多函数判断15分钟线第M15_NO个节点的多空， getKReverseBull =1表示2个指标多，=2表示3个及以上指标多，=0表示得不到“多”的结论。
def getKReverseBull(stockStat, quotes_fg15m, M15_NO, latestPrice, tick):
    kReverseBull    = 0
    maTrend         = getMATrend(stockStat, M15_NO)
    bollTrend       = getBOLLTrend(stockStat, M15_NO, latestPrice, tick)
    kdjTrend        = getKDJTrend(stockStat, M15_NO)
    macdTrend       = getMACDTrend(stockStat, M15_NO)
    TrendList       = [maTrend, bollTrend, kdjTrend, macdTrend]

    down_shadow     = min(stockStat.open.values[ M15_NO ], stockStat.close.values[ M15_NO ]) - stockStat.low.values[ M15_NO ] #下影线
    entity          = abs( stockStat.open.values[ M15_NO ] - stockStat.close.values[ M15_NO ] )
    
    if (down_shadow >= 6*tick and down_shadow >= 2*entity) and TrendList.count(1) == 2:
        kReverseBull = 1
    elif (down_shadow >= 6*tick and down_shadow >= 2*entity) and TrendList.count(1) >= 3:
        kReverseBull = 2

    return kReverseBull

# 以K线反转空函数判断15分钟线第M15_NO个节点的多空， getKReverseBear =1表示2个指标空，=2表示3个及以上指标空，=0表示得不到“空”的结论。
def getKReverseBear(stockStat, quotes_fg15m, M15_NO, latestPrice, tick):
    kReverseBear    = 0
    maTrend         = getMATrend(stockStat, M15_NO)
    bollTrend       = getBOLLTrend(stockStat, M15_NO, latestPrice, tick)
    kdjTrend        = getKDJTrend(stockStat, M15_NO)
    macdTrend       = getMACDTrend(stockStat, M15_NO)
    TrendList       = [maTrend, bollTrend, kdjTrend, macdTrend]

    up_shadow       = stockStat.high.values[ M15_NO ] - max(stockStat.open.values[ M15_NO ], stockStat.close.values[ M15_NO ]) #上影线
    entity          = abs( stockStat.open.values[ M15_NO ] - stockStat.close.values[ M15_NO ] )
    
    if (up_shadow >= 6*tick and up_shadow >= 2*entity) and TrendList.count(-1) == 2:
        kReverseBear = 1
    elif (up_shadow >= 6*tick and up_shadow >= 2*entity) and TrendList.count(-1) >= 3:
        kReverseBear = 2
        
    return kReverseBear

#code:期货代码; 挂单操作; order_k:挂单时的K线; order_price:挂单开仓价; ship:挂单开仓量；; action:实际操作，; 开仓时间，; 开仓价格，; 利润
def order(code, order_a, order_k, order_p, ship, target, stop, retreat_k):
    action      = ''
    open_k      = 0 
    open_p      = 0 
    close_k     = 0 
    close_p     = 0 
    profit      = 0
    trade_records.append([code, order_a, order_k, order_p, ship, target, stop, retreat_k, action, open_k, open_p, close_k, close_p, profit])

########################################## 四、准备数据 ########################################
###############################################################################################
m15_tick    = weekLine.tick
m15_quotes  = weekLine.quotes_pre
stop        = 5 * m15_tick
target      = 5 * m15_tick
capital     = 1000000 #自有资本，单位（元）
#retreat_m  = 3 #单子未成交，3分钟撤单
retreat_k   = 1 #单子未成交，1个15分钟节点内撤单
shiprate    = 0
week_N      = 0
code        = 'RU1801'

#期货代码，挂单操作，挂单时间，  挂单开仓价， 挂单开仓量；止盈，     止损，  撤单时间，   实际操作，开仓时间，  开仓价格，  平仓时间，平仓价格， 利润
#[[code],  [order_a], [order_k], [order_p], [ship],     [target], [stop], [retreat_k], [action], [open_k], [open_p],  [close_k],[close_p],[profit]]
#trade_record = [[,  , , , , , , , , , , , ,] ]
trade_records = []

#start = datetime.strptime("2017/7/17  9:00", "%Y/%m/%d  %H:%M").timestamp()
#end = datetime.strptime("2017/7/17  23:59", "%Y/%m/%d  %H:%M").timestamp()

quotes_pre_15m  = np.loadtxt(code+'_15M.csv', delimiter=',', usecols=(0,1,2,3,4,), converters={0:datestr2num1}, unpack=True)
quotes_15m      = []

for i in range( len(quotes_pre_15m[0]) ):
    datas       = (quotes_pre_15m[0][i],quotes_pre_15m[1][i],quotes_pre_15m[2][i],quotes_pre_15m[3][i],quotes_pre_15m[4][i])
    quotes_15m.append(datas)

cols            = ["date","open","high","low","close"]
df              = pd.DataFrame(quotes_15m, columns=cols)

#if start is not None:
#    df = df[df.date >= start]
#if end is not None:
#    df = df[df.date <= end]

df["date"]      = df.index.values #增加日期列。
df              = df.sort_index(0) # 将数据按照日期排序下。
stockStat       = stockstats.StockDataFrame.retype(df)

#1，MA指标：    ###
stockStat[['close','close_5_sma','close_10_sma','close_20_sma']]
#2，KDJ指标：    ###
stockStat[['close','kdjk','kdjd','kdjj']]
#3，MACD指标：    ###
stockStat[['close','macd','macds','macdh']]
#4，BOLL指标：    ###
stockStat[['close','boll','boll_ub','boll_lb']]

quotes_15m_new  = []
for i in range( len(stockStat.index.values) ):
    datas_new = (stockStat.index.values[i],stockStat.open.values[i],stockStat.high.values[i],stockStat.low.values[i],
                 stockStat.close.values[i],stockStat.close_5_sma.values[i],stockStat.close_10_sma.values[i],stockStat.close_20_sma.values[i],
                 stockStat.rsv_9.values[i],stockStat.kdjk.values[i],stockStat.kdjd.values[i],stockStat.kdjj.values[i],
                 stockStat.macd.values[i],stockStat.macds.values[i],stockStat.macdh.values[i],stockStat.boll.values[i],
                 stockStat.boll_ub.values[i],stockStat.boll_lb.values[i])
    quotes_15m_new.append(datas_new)

#得到调整后的波峰、波谷：
quotes_fg15m    = weekLine.getLineChart(quotes_15m_new) # quotes_fg15m数组内容依次为：date、price、index、direction

########################################## 五、挂单操作 ########################################
###############################################################################################
order_p     = 0
ship        = 0
m15_no_list = []
for week_N in range(21,39):
    weekTrend       = weekLine.getFinalTrend( week_N )

    m15_no_begin    = np.min(np.where( m15_quotes[0][week_N-1] < quotes_pre_15m[0] )) # fg_index是离 week_N 最近的波峰/波谷下标
    m15_no_end      = np.max(np.where( quotes_pre_15m[0] <= m15_quotes[0][week_N] ))

    if m15_no_begin < 6:#本例子中，第6个节点前不包括完整的波峰、波谷，所以从第6周开始计算。M15_NO为当前时间节点
        m15_no_list = stockStat.index.values[6 : m15_no_end + 1]
    else:
        m15_no_list = stockStat.index.values[ m15_no_begin : m15_no_end + 1]

    for m15_no in m15_no_list:#本例子中，第6个节点前不包括完整的波峰、波谷，所以从第6周开始计算。M15_NO为当前时间节点
        #currentTime = stockStat.index.values[m15_no]
        current_k       = m15_no
        latestPrice     = stockStat.open.values[m15_no]
        kBreakTrend     = getKBreakTrend(stockStat, quotes_fg15m, m15_no - 1, latestPrice, m15_tick)
        kReverseBull    = getKReverseBull(stockStat, quotes_fg15m, m15_no - 1, latestPrice, m15_tick)
        kReverseBear    = getKReverseBear(stockStat, quotes_fg15m, m15_no - 1, latestPrice, m15_tick)
        max_3weeks_h    = max( m15_quotes[2][week_N-1], m15_quotes[2][week_N-2], m15_quotes[2][week_N-3] ) #最近三周最高价
        min_3weeks_l    = min( m15_quotes[3][week_N-1], m15_quotes[3][week_N-2], m15_quotes[3][week_N-3] ) #最近三周最底价
        mean_3weeks     = ( max_3weeks_h + min_3weeks_l )/2
        quote_f15m      = getLatestF(quotes_fg15m, m15_no, latestPrice)
        quote_g15m      = getLatestG(quotes_fg15m, m15_no, latestPrice)
        
        if kBreakTrend in (2,-2) or kReverseBull == 2 or kReverseBear == 2:#实际上，只要3个指标中某个绝对值为2，另外两个绝对值必定也为2
            shiprate = 0.8
        elif kBreakTrend in (1,-1) or kReverseBull == 1 or kReverseBear == 1:
            shiprate = 0.5
        
        if weekTrend in (1,2):          #(1) W(D)=1,2
            if kBreakTrend in (1,2):            #K线突破多
                #期货代码，挂单操作，挂单时间，  挂单开仓价， 挂单开仓量；止盈，     止损，  撤单时间，
                #code, order_a, order_k, order_p, ship, target, stop, retreat_k
                order_a     = "buy"
                order_p     = quote_f15m[1] + 2*m15_tick
                ship        = capital * shiprate//order_p
                order(code, order_a, current_k, order_p, ship, target, stop, retreat_k)
            elif kReverseBear in (1,2):         #K线反转空
                order_a     = "sell"
                order_p     = stockStat.low.values[current_k-1] - 1*m15_tick
                ship        = capital * shiprate//order_p
                order(code, order_a, current_k, order_p, ship, target, stop, retreat_k)
        elif weekTrend in (-1,-2):      #(2) W(D)=-1,-2
            if kBreakTrend in (-1,-2):           #K线突破空
                order_a     = "sell"
                order_p     = quote_g15m[1] - 2*m15_tick
                ship        = capital * shiprate//order_p
                order(code, order_a, current_k, order_p, ship, target, stop, retreat_k)
            elif kReverseBull in (1,2):         #K线反转多
                order_a     = "buy"
                order_p     = stockStat.high.values[current_k-1] + 1*m15_tick
                ship        = capital * shiprate//order_p
                order(code, order_a, current_k, order_p, ship, target, stop, retreat_k)
        elif weekTrend in (0,10) and latestPrice <= mean_3weeks:#(3) W(D)=0,10
            if kBreakTrend in (1,2):            #K线突破多
                order_a     = "buy"
                order_p     = quote_f15m[1] + 2*m15_tick
                ship        = capital * shiprate//order_p
                order(code, order_a, current_k, order_p, ship, target, stop, retreat_k)
            elif kReverseBull in (1,2):         #K线反转多
                order_a     = "buy"
                order_p     = stockStat.high.values[current_k-1] + 1*m15_tick
                ship        = capital * shiprate//order_p
                order(code, order_a, current_k, order_p, ship, target, stop, retreat_k)
        elif weekTrend in (0,10) and latestPrice >= mean_3weeks:#(4) W(D)=0,10
            if kBreakTrend in (-1,-2):           #K线突破空
                order_a     = "sell"
                order_p     = quote_g15m[1] - 2*m15_tick
                ship        = capital * shiprate//order_p
                order(code, order_a, current_k, order_p, ship, target, stop, retreat_k)
            elif kReverseBear in (1,2):         #K线反转空
                order_a     = "sell"
                order_p     = stockStat.low.values[current_k-1] - 1*m15_tick
                ship        = capital * shiprate//order_p
                order(code, order_a, current_k, order_p, ship, target, stop, retreat_k)
np.savetxt(code+'_order_1.csv', trade_records, delimiter=',', fmt='%s')
########################################## 六、交易操作 ########################################
###############################################################################################
#期货代码，挂单操作，挂单时间，  挂单开仓价， 挂单开仓量；止盈，     止损，  撤单时间，   实际操作，开仓时间，  开仓价格，  平仓时间，平仓价格， 利润
#[[code,  order_a, order_k,    order_p,    ship,      target,    stop ,  retreat_k ,  action ,  open_k ,  open_p ,   close_k , close_p , profit] ]
#trade_record = [[,  , , , , , , , , , , , ,] ]

for m15_no in stockStat.index.values:#利用15分钟K线模拟时间轴
    for i in range(len(trade_records)):
        trade_record= copy.deepcopy(trade_records[i])
        code        = trade_record[0]
        order_a     = trade_record[1]
        order_k     = trade_record[2]
        order_p     = trade_record[3]
        ship        = trade_record[4]
        target      = trade_record[5]
        stop        = trade_record[6]
        retreat_k   = trade_record[7]
        action      = trade_record[8]
        open_k      = trade_record[9]
        open_p      = trade_record[10]
        close_k     = trade_record[11]
        close_p     = trade_record[12]
        profit      = trade_record[13]
        if action == '':#情况1：该挂单没被操作
            if m15_no <= order_k < m15_no + retreat_k and stockStat.low.values[m15_no] <= order_p <= stockStat.high.values[m15_no]:#执行时间范围内，达到了挂单价格，进行操作
                trade_record[8]     = order_a
                trade_record[9]     = m15_no
                trade_record[10]    = order_p
                trade_records[i]    = trade_record #更新trade_records数组中第i条挂单记录
                action = order_a #更新当前挂单状态
                open_k = m15_no 
                open_p = order_p #假设按挂单价格成交
            elif m15_no <= order_k < m15_no + retreat_k and (order_p > stockStat.high.values[m15_no] or order_p < stockStat.low.values[m15_no]): #假设15分钟内没达到要求的价格，则撤单
                trade_record[8]     = 'retreat'
                trade_records[i]    = trade_record #更新trade_records数组中第i条挂单记录
                
        if action == 'buy':#情况2：该挂单已开仓，未平仓。假设挂单都能按指令价格止损、止盈
            if stockStat.high.values[m15_no] >= open_p + target:#触动止盈操作，进行平仓
                trade_record[8]     = 'close'
                trade_record[11]    = m15_no
                trade_record[12]    = open_p + target
                trade_record[13]    = ship * target 
                trade_records[i]    = trade_record #更新trade_records数组中第i条挂单记录
            elif stockStat.low.values[m15_no] <= open_p - stop:#触动止损操作，进行平仓
                trade_record[8]     = 'close'
                trade_record[11]    = m15_no
                trade_record[12]    = open_p - stop
                trade_record[13]    = - ship * stop 
                trade_records[i]    = trade_record #更新trade_records数组中第i条挂单记录
        elif action == 'sell':
            if  stockStat.high.values[m15_no] <= open_p - target:#触动止盈操作，进行平仓
                trade_record[8]     = 'close'
                trade_record[11]    = m15_no
                trade_record[12]    = open_p - target
                trade_record[13]    = ship * target 
                trade_records[i]    = trade_record #更新trade_records数组中第i条挂单记录
            elif stockStat.low.values[m15_no] >= open_p + stop:#触动止损操作，进行平仓
                trade_record[8]     = 'close'
                trade_record[11]    = m15_no
                trade_record[12]    = open_p - stop
                trade_record[13]    = - ship * stop 
                trade_records[i]    = trade_record #更新trade_records数组中第i条挂单记录
np.savetxt(code+'_action_1.csv', trade_records, delimiter=',', fmt='%s')