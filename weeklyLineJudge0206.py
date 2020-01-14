# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 09:40:13 2017

@author: Administrator
"""
import numpy as np
import copy
import matplotlib.pyplot as plt
#import sys 
from matplotlib.dates import DateFormatter 
from matplotlib.dates import DayLocator 
from matplotlib.dates import MonthLocator 
from matplotlib.dates import date2num
from matplotlib.finance import candlestick_ohlc
from datetime import datetime

#import tushare as ts

def datestr2num(s): 
    date1 = datetime.strptime(s.decode('ascii'), "%Y/%m/%d").date()
    date = date2num(date1)
    return date

###############################################################################
############################ 一、画出峰谷折线图 ################################

# 1、提取数据并绘FG0图 #########################################################
alldays = DayLocator()  
months = MonthLocator() 
month_formatter = DateFormatter("%b %Y")  

#quotes的内容(time, open, high, low, close)
#quotes_pre = np.loadtxt('RU1801WeeklyData.csv', delimiter=',', usecols=(0,1,2,3,4,), converters={0:datestr2num}, unpack=True) 
#quotes_pre = np.loadtxt('RU1709WeeklyData.csv', delimiter=',', usecols=(0,1,2,3,4,), converters={0:datestr2num}, unpack=True) 
#quotes_pre = np.loadtxt('Y1801WeeklyData.csv', delimiter=',', usecols=(0,1,2,3,4,), converters={0:datestr2num}, unpack=True) 
quotes_pre = np.loadtxt('RM801WeeklyData.csv', delimiter=',', usecols=(0,1,2,3,4,), converters={0:datestr2num}, unpack=True) 
#quotes_pre = np.loadtxt('RB1801WeeklyData.csv', delimiter=',', usecols=(0,1,2,3,4,), converters={0:datestr2num}, unpack=True) 
#quotes_pre = np.loadtxt('J1801WeeklyData.csv', delimiter=',', usecols=(0,1,2,3,4,), converters={0:datestr2num}, unpack=True) 
quotes = []
tick = 5 #期货橡胶ru的刻度，即一手价格变动的最低价

global g_resistanceList_wk         #[0] W(N)中的 周数N 列表
global g_resistanceList            #[1] W(N)的 resistanceList 列表
global g_resceAbateList            #[0] W(N)失效阻力的周序数及类型，结构为[[weekNo0,resceType0],[weekNo1,resceType1],[weekNo2,resceType2]]
global g_selfAdjustTrend_wk        #[0] W(N)中的 周数N 列表
global g_selfAdjustTrend           #[1] W(N)的 selfAdjustTrend 列表
global g_synthesizeTrend_wk        #[0] W(N)中的 synthesizeTrend 列表
global g_synthesizeTrend           #[1] W(N)的 synthesizeTrend 列表
global g_week_trend_wk             #[0] W(N)中的 周数N 列表
global g_week_trend                #[1] W(N)的 FinalTrend 列表
g_resistanceList_wk = []
g_resistanceList = []
g_resceAbateList = []
g_selfAdjustTrend_wk = []
g_selfAdjustTrend = []
g_synthesizeTrend_wk = []
g_synthesizeTrend = []
g_week_trend_wk = []
g_week_trend = []

for i in range( len(quotes_pre[0]) ):
    datas = (quotes_pre[0][i],quotes_pre[1][i],quotes_pre[2][i],quotes_pre[3][i],quotes_pre[4][i])
    quotes.append(datas)

#alldays = DayLocator()  
#months = MonthLocator() 
#month_formatter = DateFormatter("%b %Y") 

#fig = plt.figure(figsize=(18, 9)) 
#ax = fig.add_subplot(111) 
#ax.xaxis.set_major_locator(months) 
#ax.xaxis.set_minor_locator(alldays) 
#ax.xaxis.set_major_formatter(month_formatter) 

#plt.title("Weekly Line")
#plt.xlabel("Week")
#plt.ylabel("Price")
#candlestick_ohlc(ax,quotes,width=0.2, colorup='r',colordown='g') 
#plt.plot(quotes_pre[0], quotes_pre[4], 'r', lw=1.0) #画出收盘价折线图

# 2、找到极值点 ################################################################
#需考虑临值相等的情况 #
quotes_fg0_date = []         #峰谷的日期
quotes_fg0_price = []        #峰谷的价格
quotes_fg0_index = []        #峰谷在原时间序列中的位置
quotes_fg0_direction = []    #值为1表示波峰，值为0表示波谷
for i in range( len(quotes_pre[0]) )[1:-1]:
    #波峰：
    if quotes_pre[4][i]>quotes_pre[4][i-1] and quotes_pre[4][i]>quotes_pre[4][i+1]:
        quotes_fg0_date.append(quotes_pre[0][i])
        quotes_fg0_price.append(quotes_pre[4][i])
        quotes_fg0_index.append(i)
        quotes_fg0_direction.append(1)
    #波谷：
    if quotes_pre[4][i]<quotes_pre[4][i-1] and quotes_pre[4][i]<quotes_pre[4][i+1]:
        quotes_fg0_date.append(quotes_pre[0][i])
        quotes_fg0_price.append(quotes_pre[4][i])
        quotes_fg0_index.append(i)
        quotes_fg0_direction.append(0)

#3、修正极值点 #################################################################

#第1步：对于波峰（波谷），用本周最大值（最小值）代替收盘价
j = 0 # 记录quotes_fg0_price中元素的坐标
for i in range( len(quotes_fg0_index) ):
    fg0_index = quotes_fg0_index[i]
    #本极值点为波谷，用最低价代替收盘价
    if quotes_fg0_direction[i] == 0:
        quotes_fg0_price[j] = quotes_pre[3][fg0_index]
    #开盘价<收盘价，本极值点为波峰，用最高价代替收盘价
    if quotes_fg0_direction[i] == 1:
        quotes_fg0_price[j] = quotes_pre[2][fg0_index]
    j = j + 1 # quotes_fg0_price下一个元素的坐标
#plt.plot(quotes_fg0_date, quotes_fg0_price, 'y', lw=1.5) #画出极值点折线图
#ax.scatter(quotes_fg0_date, quotes_fg0_price, alpha=0.5) #画出极值点散点图


#第2步：根据规则，用新峰谷时间点代替部分原峰谷时间点
quotes_fg_date = copy.deepcopy(quotes_fg0_date)
quotes_fg_price = copy.deepcopy(quotes_fg0_price)
quotes_fg_index = copy.deepcopy(quotes_fg0_index)
quotes_fg_direction = copy.deepcopy(quotes_fg0_direction)    #值为1表示波峰，值为0表示波谷

#先根据波峰确定波次，调整波谷
for j in range(len(quotes_fg_index)):
    if quotes_fg_direction[j] == 1 and  j + 2 < len(quotes_fg_index):#本波峰后面还有一个波峰，能构成完整波次,才在本波次内调整波谷
        quotes_fwave_begin = quotes_fg_index[j]    #第一个波峰在quotes的序号
        quotes_fwave_end   = quotes_fg_index[j+2]  #第二个波峰在quotes的序号
        quotes_wave_low = copy.deepcopy(quotes_pre[3][quotes_fwave_begin: quotes_fwave_end + 1])
        quotes_wave_low_min_index = np.argmin(quotes_wave_low)
        quotes_wave_low_min = quotes_wave_low[ quotes_wave_low_min_index ]
        quotes_g_index = quotes_fwave_begin + quotes_wave_low_min_index #新波谷在quotes的序号
        if quotes_wave_low_min < quotes_fg_price[j+1]:
            # 用新波谷代替旧波谷:
            quotes_fg_date[j+1] = quotes_pre[0][quotes_g_index]
            quotes_fg_price[j+1] = quotes_pre[3][quotes_g_index]
            quotes_fg_index[j+1] = quotes_g_index

#再根据波谷确定波次，调整波峰
for j in range(len(quotes_fg_index)):
    if quotes_fg_direction[j] == 0 and  j + 2 < len(quotes_fg_index):#本波谷后面还有一个波谷，能构成完整波次,才在本波次内调整波峰
        quotes_gwave_begin = quotes_fg_index[j]    #第一个波谷在quotes的序号
        quotes_gwave_end   = quotes_fg_index[j+2]  #第二个波谷在quotes的序号
        quotes_wave_high = copy.deepcopy(quotes_pre[2][quotes_gwave_begin: quotes_gwave_end + 1])
        quotes_wave_high_max_index = np.argmax(quotes_wave_high)
        quotes_wave_high_max = quotes_wave_high[ quotes_wave_high_max_index ]
        quotes_f_index = quotes_gwave_begin + quotes_wave_high_max_index #新波峰在quotes的序号
        if quotes_wave_high_max > quotes_fg_price[j+1]:
            # 用新波峰代替旧波峰:
            quotes_fg_date[j+1] = quotes_pre[0][quotes_f_index]
            quotes_fg_price[j+1] = quotes_pre[2][quotes_f_index]
            quotes_fg_index[j+1] = quotes_f_index
    
#plt.plot(quotes_fg_date, quotes_fg_price, 'b--', lw=2.0) #画出调整后的峰谷折线图
#ax.scatter(quotes_fg_date, quotes_fg_price, alpha=0.5) #画出调整后的峰谷散点图

#标注各周编号
#wk_no = range(len(quotes_pre[0]))  # 周序号
#for i in range(len(quotes_pre[0])):
#    wk_number = wk_no[i] + 1
#    plt.text(quotes_pre[0][i], quotes_pre[2][i], wk_number, alpha=0.5)

#fig.autofmt_xdate() 
#plt.savefig("wave.png")
#plt.show()

###############################################################################
############################ 二、定义函数 ######################################
# 本节定义这些函数：反转周函数getReverse、突破周函数getBreak、阻力定义函数getResistance、
# 反转首周定义函数getFirstReverse、突破首周定义函数getFirstBreak、本身K线含义函数getSelfTrend、
# 阻力空间函数getBreakTrend、本身定义调整函数getSelfAdjustTrend

# 1、本身K线含义函数：getSelfTrend(week_N)######################################
# 通过K线本身判断week_N的趋势，selfTrend = -1
# 表示本身K线趋势空，=1表示本身K线趋势多，=10 表示本身K线宽幅震荡。
def getSelfTrend(week_N):
    selfTrend = 10
    week_N_rise = 0
    week_N_rise_rate = 0
    # week_N的涨幅(跌幅):
    if week_N == 1:
        week_N_rise = (quotes_pre[4][week_N-1] - quotes_pre[1][week_N-1]) #上涨（下跌）值
        week_N_rise_rate = week_N_rise / quotes_pre[1][week_N-1] #涨幅(跌幅)
    elif week_N >= 2:
        week_N_rise = (quotes_pre[4][week_N-1] - quotes_pre[4][week_N-2]) #上涨（下跌）值
        week_N_rise_rate = week_N_rise / quotes_pre[4][week_N-2] #涨幅(跌幅)
    
    # week_N的上（下）影线:
    up_shadow = quotes_pre[2][week_N-1] - max(quotes_pre[1][week_N-1], quotes_pre[4][week_N-1]) #上影线
    down_shadow = min(quotes_pre[1][week_N-1], quotes_pre[4][week_N-1]) - quotes_pre[3][week_N-1] #下影线
    if week_N_rise_rate >= 0 :
        if week_N_rise_rate>0.03 : selfTrend =1
        elif week_N_rise_rate>0.012 and up_shadow < week_N_rise: selfTrend =1
        elif week_N_rise_rate>0.012 and up_shadow >= week_N_rise: selfTrend =10
        elif week_N_rise_rate>=0 : selfTrend =10

    if week_N_rise_rate < 0 :
        if abs(week_N_rise_rate)>0.03 : selfTrend =-1
        elif abs(week_N_rise_rate)>0.012 and down_shadow < abs(week_N_rise): selfTrend =-1
        elif abs(week_N_rise_rate)>0.012 and down_shadow >= abs(week_N_rise): selfTrend =10
        elif abs(week_N_rise_rate)>0 : selfTrend =10
    
    return selfTrend

# 2、反转周函数：getReverse(week_N)#############################################
# 判断第N周是否为反转周，reverse_wk = 0表示非反转周，=1表示反转周
def getReverse(week_N):
    reverse_wk = 0
    if quotes_pre[1][week_N-2] > quotes_pre[4][week_N-2] and quotes_pre[1][week_N-1] < quotes_pre[4][week_N-1]:
        reverse_wk = 1
        
    if quotes_pre[1][week_N-2] < quotes_pre[4][week_N-2] and quotes_pre[1][week_N-1] > quotes_pre[4][week_N-1]:
        reverse_wk = 1
    
    return reverse_wk

# 3、突破周函数：getBreak(week_N)###############################################
# 判断第N周是否为突破周，break_wk = 0表示非突破周，=1/-1表示突破周(趋势多/趋势空)
# 返回值break_list：[0]为week_N、[1]为break_wk(可取值：0,1，-1)、[2]突破的波峰f1(或波谷g1)在quotes中的序列
def getBreak(week_N):
    break_wk = 0
    break_list = [week_N,break_wk,-1] # 分别为week_N、 break_wk(可取值：0,1，-1)、 波峰f1(或波谷g1)
    
    # week_N的涨幅(跌幅):
    week_N_rise = (quotes_pre[4][week_N-1] - quotes_pre[4][week_N-2]) #上涨（下跌）值
    week_N_rise_rate = week_N_rise / quotes_pre[4][week_N-2] #涨幅(跌幅)
    # week_N的上（下）影线:
    up_shadow = quotes_pre[2][week_N-1] - max(quotes_pre[1][week_N-1], quotes_pre[4][week_N-1]) #上影线
    down_shadow = min(quotes_pre[1][week_N-1], quotes_pre[4][week_N-1]) - quotes_pre[3][week_N-1] #下影线
   
    quotes_f_list = [[],[]] #将所有波峰，按时间从近到远进行排序，[0]存储价格，[1]存储波峰属于哪一周
    quotes_g_list = [[],[]] #将所有波谷，按时间从近到远进行排序，[0]存储价格，[1]存储波峰属于哪一周
    for i in range( len(quotes_fg_price) )[::-1]:
        if( quotes_fg_direction[i] == 1 and quotes_fg_index[i] < week_N - 1 ):# week_N 之前的波峰
            quotes_f_list[0].append( quotes_fg_price[i] )
            quotes_f_list[1].append( quotes_fg_index[i] )
        elif( quotes_fg_direction[i] == 0 and quotes_fg_index[i] < week_N - 1 ):# week_N 之前的波谷
            quotes_g_list[0].append( quotes_fg_price[i] )
            quotes_g_list[1].append( quotes_fg_index[i] )

    last_f_price = 0 #记录上一个可能的阻力值
    last_g_Price = 200000 #记录可能的阻力值
    if week_N_rise_rate > 0.03 or ( 0.012 < week_N_rise_rate <= 0.03 and up_shadow < week_N_rise ):
        #找符合约束条件的波峰：
        quotes_f_index = -1 # quotes_f_index即为所求波峰
        j1 = 0 # 记录遍历符合条件波峰的个数

        for i in range( len(quotes_f_list[0]) ):
            quotes_f_index = quotes_f_list[1][i]#波峰周在quotes序列中的下标
            if j1 == 5: #如果已查找了5个波峰，直接结束循环
                break
            
            if quotes_f_list[0][i] > last_f_price: #找比f1之前更高的波峰f2
                if quotes_pre[2][week_N-2] < quotes_pre[2][quotes_f_index] <= quotes_pre[2][week_N-1] - 5*tick and \
                quotes_pre[4][quotes_f_index] <= quotes_pre[4][week_N-1] - 3*tick:#找到了符合条件的波峰
                    break_wk = 1 # 本周为突破周趋势多
                    break_list = [week_N, break_wk, quotes_f_index + 1]
                    break
                else:
                    last_f_price = quotes_f_list[0][i] #将当前波峰价格设为f2，以此为基础找f3
            j1 = j1 + 1

    if week_N_rise_rate < -0.03 or ( -0.03 <= week_N_rise_rate < -0.012 and down_shadow < abs(week_N_rise) ):
        #找符合约束条件的波谷：
        quotes_g_index = -1 # quotes_g_index即为所求波谷
        j1 = 0 # 记录遍历符合条件波谷的个数

        for i in range( len(quotes_g_list[0]) ):
            quotes_g_index = quotes_g_list[1][i]#波谷周在quotes序列中的下标
            if j1 == 5: #如果已查找了5个波谷，直接结束循环
                break
            
            if quotes_g_list[0][i] < last_g_Price: #找比g1之前更低的波谷g2
                if quotes_pre[3][week_N-2] > quotes_pre[3][quotes_g_index] >= quotes_pre[3][week_N-1] + 5*tick and \
                quotes_pre[4][quotes_g_index] >= quotes_pre[4][week_N-1] + 3*tick:#找到了符合条件的波峰
                    break_wk = -1 # 本周为突破周趋势空
                    break_list = [week_N, break_wk, quotes_g_index + 1]
                    break
                else:
                    last_g_Price = quotes_g_list[0][i] #将当前波峰价格设为g2，以此为基础找g3
            j1 = j1 + 1

    return break_list

# 4、本身定义调整函数：getSelfAdjustTrend(week_N)##############################
# 通过week_N-1的首周判断函数，得到week_N的本身定义调整函数,selfAdjustTrend=-1表示趋势明显空;# =1表示趋势明显多;=10表示宽幅震荡。
def getSelfAdjustTrend(week_N):
    week_N_trend = 10
    week_1N_trend = 10
    
    if week_N in [1, 2, 3]: #前3周，用本身函数代替本身调整函数
        week_N_trend = getSelfTrend(week_N)
        return week_N_trend
    
    # 数据准备1：week_2N，即W(N-2)：
    week_2N_rise = (quotes_pre[4][week_N-3] - quotes_pre[4][week_N-4]) #上涨（下跌）值
    week_2N_rise_rate = week_2N_rise / quotes_pre[4][week_N-4] #涨幅(跌幅)
    
    # 数据准备2：week_1N，即W(N-1)：
    week_1N_rise = (quotes_pre[4][week_N-2] - quotes_pre[4][week_N-3]) #上涨（下跌）值
    week_1N_rise_rate = week_1N_rise / quotes_pre[4][week_N-3] #涨幅(跌幅)
    week_1N_upShadow = quotes_pre[2][week_N-2]-max(quotes_pre[1][week_N-2], quotes_pre[4][week_N-2]) #上影线
    week_1N_lowShadow = min(quotes_pre[1][week_N-2], quotes_pre[4][week_N-2]) - quotes_pre[3][week_N-2] #下影线
    
    if week_N-1 in g_week_trend_wk: #如全局变量列表中已有数据，则使用全局变量
        index_g_finalTrend = g_week_trend_wk.index( week_N-1 )
        week_1N_trend = copy.deepcopy( g_week_trend[index_g_finalTrend] )
    else:
        week_1N_trend = getFinalTrend(week_N-1)        
   
    # 数据准备3：week_N，即W(N)：
    week_N_rise = (quotes_pre[4][week_N-1] - quotes_pre[4][week_N-2]) #上涨（下跌）值
    week_N_rise_rate = week_N_rise / quotes_pre[4][week_N-2] #涨幅(跌幅)
    week_N_upShadow = quotes_pre[2][week_N-1]-max(quotes_pre[1][week_N-1], quotes_pre[4][week_N-1]) #上影线
    week_N_lowShadow = min(quotes_pre[1][week_N-1], quotes_pre[4][week_N-1]) - quotes_pre[3][week_N-1] #下影线
    
    #情况（01）时W（N）周线定义：#####################
    if week_1N_trend == 2:
        if week_N_rise_rate >= 0:
            if quotes_pre[3][week_N-1] >= quotes_pre[3][week_N - 2]:
                week_N_trend = 1
            elif quotes_pre[3][week_N-1] < quotes_pre[3][week_N - 2] and quotes_pre[4][week_N-1] > quotes_pre[3][week_N - 2]:
                week_N_trend = 1
            elif quotes_pre[3][week_N-1] < quotes_pre[3][week_N - 2] and quotes_pre[4][week_N-1] <= quotes_pre[2][week_N - 2]:
                week_N_trend = 10
                
        elif week_N_rise_rate < 0:
            if week_N_rise_rate < 0 and week_1N_rise_rate > 0:
                if abs(week_N_rise_rate) < week_1N_rise_rate * 0.5:
                    week_N_trend = 1
                elif week_1N_rise_rate * 0.5 <= abs(week_N_rise_rate) <= week_1N_rise_rate:
                    week_N_trend = 10
                elif abs(week_N_rise_rate) > week_1N_rise_rate:
                    week_N_trend = -1
    
    #情况（02）时W（N）周线定义：#####################
    if week_1N_trend == 1 and week_1N_rise_rate > 0.03:
        if week_N_rise_rate >= 0:
            if quotes_pre[3][week_N-1] >= quotes_pre[3][week_N - 2]:
                week_N_trend = 1
            elif quotes_pre[3][week_N-1] < quotes_pre[3][week_N - 2] and quotes_pre[4][week_N-1] > quotes_pre[2][week_N - 2] :
                week_N_trend = 1
            elif quotes_pre[3][week_N-1] < quotes_pre[3][week_N - 2] and quotes_pre[4][week_N-1] <= quotes_pre[2][week_N - 2] :
                week_N_trend = 10
                
        elif week_N_rise_rate < 0:
            if week_1N_rise_rate > 0:
                if abs(week_N_rise_rate) < week_1N_rise_rate/3:
                    week_N_trend = 1
                elif week_1N_rise_rate/3 <= abs(week_N_rise_rate) <= week_1N_rise_rate * 0.5:
                    week_N_trend = 10
                elif abs(week_N_rise_rate) > week_1N_rise_rate * 0.5:
                    week_N_trend = -1
        
    #情况（03）时W（N）周线定义：#####################
    if week_1N_trend == 1 and 0.012 < week_1N_rise_rate < 0.03 and week_1N_upShadow < week_1N_rise:
        if week_N_rise_rate > 0:
            if quotes_pre[3][week_N-1] >= quotes_pre[3][week_N - 2]:
                week_N_trend = 1
            elif quotes_pre[3][week_N-1] < quotes_pre[3][week_N - 2] and quotes_pre[4][week_N-1] > quotes_pre[2][week_N - 2]:
                week_N_trend = 1
            elif quotes_pre[3][week_N-1] < quotes_pre[3][week_N - 2] and quotes_pre[4][week_N-1] <= quotes_pre[2][week_N - 2]:
                week_N_trend = 10
                
        elif week_N_rise_rate < 0:
            if week_1N_rise_rate > 0:
                if abs(week_N_rise_rate) < week_1N_rise_rate/3:
                    week_N_trend = 1
                elif week_1N_rise_rate/3 <= abs(week_N_rise_rate) <= week_1N_rise_rate * 0.5:
                    week_N_trend = 10
                elif abs(week_N_rise_rate) > week_1N_rise_rate * 0.5:
                    week_N_trend = -1
            
    #情况（04）时W（N）周线定义：#####################
    if week_1N_trend == 1 and 0.012 < week_1N_rise_rate < 0.03 and week_1N_upShadow >= week_1N_rise:
        if week_N_rise_rate > 0:
            if quotes_pre[3][week_N-1] >= quotes_pre[3][week_N - 2]:
                week_N_trend = 1
            elif quotes_pre[3][week_N-1] < quotes_pre[3][week_N - 2] and quotes_pre[4][week_N-1] > quotes_pre[2][week_N - 2]:
                week_N_trend = 1
            elif quotes_pre[3][week_N-1] < quotes_pre[3][week_N - 2] and quotes_pre[4][week_N-1] <= quotes_pre[2][week_N - 2]:
                week_N_trend = 10
        elif week_N_rise_rate < 0:
            if abs(week_N_rise_rate) > 0.03:
                week_N_trend = 1
            elif 0.012 < abs(week_N_rise_rate) <= 0.03 and week_N_lowShadow < abs(week_N_rise):
                if abs(week_N_rise_rate) > week_1N_rise_rate:
                    week_N_trend = -1
                elif abs(week_N_rise_rate) <= week_1N_rise_rate:
                    week_N_trend = 10
            elif 0.012 < abs(week_N_rise_rate) <= 0.03 and week_N_lowShadow >= abs(week_N_rise):
                week_N_trend = 10
            elif abs(week_N_rise_rate) <= 0.012:
                week_N_trend = 10
            
    #情况（05）时W（N）周线定义：#####################
    if week_1N_trend == 1 and 0 <= week_1N_rise_rate <= 0.012:
        if week_N_rise_rate > 0:
            if week_N_rise_rate > 0.03:
                week_N_trend = 1
            elif 0.012 < week_N_rise_rate <= 0.03 and week_N_upShadow < week_N_rise:
                week_N_trend = 1
            elif 0.012 < week_N_rise_rate <= 0.03 and week_N_upShadow >= week_N_rise and \
            quotes_pre[2][week_N-1] > quotes_pre[2][week_N - 2]:
                week_N_trend = 1
            elif 0.012 < week_N_rise_rate <= 0.03 and week_N_upShadow >= week_N_rise and \
            quotes_pre[2][week_N-1] <= quotes_pre[2][week_N - 2]:
                week_N_trend = 10
            elif 0 < week_N_rise_rate <= 0.012 and quotes_pre[2][week_N-1] > quotes_pre[2][week_N - 2]:
                week_N_trend = 1
            elif 0 < week_N_rise_rate <= 0.012 and quotes_pre[2][week_N-1] <= quotes_pre[2][week_N - 2]:
                week_N_trend = 10
        elif week_N_rise_rate < 0:
            week_N_trend = getSelfTrend(week_N )
            
    #情况（06）时W（N）周线定义：#####################
    if week_1N_trend == 1 and -0.012 <= week_1N_rise_rate < 0:
        if week_N_rise_rate > 0:
            if week_N_rise_rate > 0.03:
                week_N_trend = 1
            elif 0.012 < week_N_rise_rate <= 0.03 and week_N_upShadow < week_N_rise and \
            quotes_pre[4][week_N-1] >= quotes_pre[2][week_N - 2]:
                week_N_trend = 1
            elif 0.012 < week_N_rise_rate <= 0.03 and week_N_upShadow < week_N_rise and \
            quotes_pre[4][week_N-1] < quotes_pre[2][week_N - 2]:
                week_N_trend = 10
            elif 0.012 < week_N_rise_rate <= 0.03 and week_N_upShadow >= week_N_rise and \
            quotes_pre[4][week_N-1] >= quotes_pre[2][week_N - 2]:
                week_N_trend = 1
            elif 0.012 < week_N_rise_rate <= 0.03 and week_N_upShadow >= week_N_rise and \
            quotes_pre[4][week_N-1] < quotes_pre[2][week_N - 2]:
                week_N_trend = 10
            elif week_N_rise_rate <= 0.012 :
                week_N_trend = 10
        elif week_N_rise_rate < 0:
            if abs(week_N_rise_rate) > 0.03 and abs(week_N_rise_rate) > week_2N_rise_rate * 0.5 > 0:
                week_N_trend = -1
            elif abs(week_N_rise_rate) > 0.03 and abs(week_N_rise_rate) <= week_2N_rise_rate * 0.5:
                week_N_trend = 10
            elif 0.012 < abs(week_N_rise_rate) <= 0.03 and week_N_lowShadow < abs(week_N_rise) and \
            abs(week_N_rise_rate) > week_2N_rise_rate * 0.5 > 0:
                week_N_trend = -1
            elif 0.012 < abs(week_N_rise_rate) <= 0.03 and week_N_lowShadow < abs(week_N_rise) and \
            abs(week_N_rise_rate) < week_2N_rise_rate * 0.5:
                week_N_trend = 10
            elif 0.012 < abs(week_N_rise_rate) <= 0.03 and week_N_lowShadow >= abs(week_N_rise):
                week_N_trend = 10
            elif abs(week_N_rise_rate) <= 0.012 :
                week_N_trend = 10
            
            
    #情况（07）时W（N）周线定义：#####################
    if week_1N_trend == 1 and week_1N_rise_rate < 0 and \
    0.012< abs(week_1N_rise_rate) <0.03 and week_1N_lowShadow > abs(week_1N_rise):
        if week_N_rise_rate > 0:
            if week_N_rise_rate > 0.03 and quotes_pre[2][week_N-1] > quotes_pre[2][week_N - 2]:
                week_N_trend = 1
            elif week_N_rise_rate > 0.03 and quotes_pre[2][week_N-1] <= quotes_pre[2][week_N - 2]:
                week_N_trend = 10
            elif 0.012 < week_N_rise_rate <= 0.03 and week_N_upShadow < week_N_rise and \
            quotes_pre[4][week_N-1] >= quotes_pre[2][week_N - 2]:
                week_N_trend = 1
            elif 0.012 < week_N_rise_rate <= 0.03 and week_N_upShadow < week_N_rise and \
            quotes_pre[4][week_N-1] < quotes_pre[2][week_N - 2]:
                week_N_trend = 10
            elif 0.012 < week_N_rise_rate <= 0.03 and week_N_upShadow >= week_N_rise and \
            quotes_pre[4][week_N-1] >= quotes_pre[2][week_N - 2]:
                week_N_trend = 1
            elif 0.012 < week_N_rise_rate <= 0.03 and week_N_upShadow >= week_N_rise and \
            quotes_pre[4][week_N-1] < quotes_pre[2][week_N - 2]:
                week_N_trend = 10
            elif week_N_rise_rate <= 0.012:
                week_N_trend = 10
        elif week_N_rise_rate < 0:
            if abs(week_N_rise_rate) > 0.03 and abs(week_N_rise_rate) > week_2N_rise_rate * 0.5 > 0: 
                week_N_trend = -1
            elif abs(week_N_rise_rate) > 0.03 and abs(week_N_rise_rate) <= week_2N_rise_rate * 0.5: 
                week_N_trend = 10
            elif 0.012 < abs(week_N_rise_rate) <= 0.03 and week_N_lowShadow < abs(week_N_rise) and \
            abs(week_N_rise) > week_2N_rise * 0.5 > 0:
                week_N_trend = -1
            elif 0.012 < abs(week_N_rise_rate) <= 0.03 and week_N_lowShadow < abs(week_N_rise) and \
            abs(week_N_rise) <= week_2N_rise * 0.5:
                week_N_trend = 10
            elif 0.012 < abs(week_N_rise_rate) <= 0.03 and week_N_lowShadow >= abs(week_N_rise):
                week_N_trend = 10
            elif abs(week_N_rise_rate) <= 0.012:
                week_N_trend = 10

    #情况（08）时W（N）周线定义：#####################
    if week_1N_trend == 1 and week_1N_rise_rate < 0 and 0.012 < abs(week_1N_rise_rate) <= 0.03 and week_1N_lowShadow < abs(week_1N_rise):
        if week_N_rise_rate > 0:
            if week_N_rise_rate > 0.03 and quotes_pre[2][week_N-1] > quotes_pre[2][week_N - 2]:
                week_N_trend = 1
            elif week_N_rise_rate > 0.03 and quotes_pre[2][week_N-1] <= quotes_pre[2][week_N - 2]:
                week_N_trend = 10
            elif 0.012 < week_N_rise_rate <= 0.03 and week_N_upShadow < week_N_rise and quotes_pre[4][week_N-1] > quotes_pre[2][week_N - 2]:
                week_N_trend = 1
            elif 0.012 < week_N_rise_rate <= 0.03 and week_N_upShadow < week_N_rise and quotes_pre[4][week_N-1] <= quotes_pre[2][week_N - 2]:
                week_N_trend = 10
            elif 0.012 < week_N_rise_rate <= 0.03 and week_N_upShadow >= week_N_rise and quotes_pre[4][week_N-1] > quotes_pre[2][week_N - 2]:
                week_N_trend = 1
            elif 0.012 < week_N_rise_rate <= 0.03 and week_N_upShadow >= week_N_rise and quotes_pre[4][week_N-1] < quotes_pre[2][week_N - 2]:
                week_N_trend = 10
            elif 0 < week_N_rise_rate <= 0.012:
                week_N_trend = 10                
        elif week_N_rise_rate < 0:
            if abs(week_N_rise_rate) > 0.03 and abs(week_N_rise_rate) >= week_2N_rise_rate * 0.5 > 0:
                week_N_trend = -1
            elif abs(week_N_rise_rate) > 0.03 and abs(week_N_rise_rate) <= week_2N_rise_rate * 0.5:
                week_N_trend = 10
            elif 0.012 < abs(week_N_rise_rate) <= 0.03 and week_N_lowShadow < abs(week_N_rise) and \
            abs(week_N_rise) > week_2N_rise * 0.5 > 0:
                week_N_trend = -1
            elif 0.012 < abs(week_N_rise_rate) <= 0.03 and week_N_lowShadow < abs(week_N_rise) and \
            abs(week_N_rise) <= week_2N_rise * 0.5:
                week_N_trend = 10
            elif 0.012 < abs(week_N_rise_rate) <= 0.03 and week_N_lowShadow >= abs(week_N_rise):
                week_N_trend = 10
            elif abs(week_N_rise_rate) <= 0.012:
                week_N_trend = 10
                
    #情况（09）时W（N）周线定义：#####################
    if week_1N_trend == 1 and week_1N_rise_rate < 0 and abs(week_1N_rise_rate) > 0.03:
        if week_N_rise_rate > 0:
            if week_N_rise_rate > 0.03 and quotes_pre[2][week_N-1] > quotes_pre[2][week_N - 2]:
                week_N_trend = 1
            elif week_N_rise_rate > 0.03 and quotes_pre[2][week_N-1] <= quotes_pre[2][week_N - 2]:
                week_N_trend = 10
            elif 0.012 < week_N_rise_rate <= 0.03 and week_N_upShadow < week_N_rise:
                week_N_trend = 10
            elif 0.012 < week_N_rise_rate <= 0.03 and week_N_upShadow >= week_N_rise:
                week_N_trend = 10
            elif week_N_rise_rate <= 0.012:
                week_N_trend = 10                
        elif week_N_rise_rate < 0:
            if abs(week_N_rise_rate) > 0.03 and abs(week_N_rise_rate) > week_2N_rise_rate * 0.5 > 0:
                week_N_trend = -1
            elif abs(week_N_rise_rate) > 0.03 and abs(week_N_rise_rate) <= week_2N_rise_rate * 0.5:
                week_N_trend = 10
            elif 0.012 < abs(week_N_rise_rate) <= 0.03 and week_N_lowShadow < abs(week_N_rise) and \
            abs(week_N_rise) > week_2N_rise * 0.5 > 0:
                week_N_trend = -1
            elif 0.012 < abs(week_N_rise_rate) <= 0.03 and week_N_lowShadow < abs(week_N_rise) and \
            abs(week_N_rise) <= week_2N_rise * 0.5:
                week_N_trend = 10
            elif 0.012 < abs(week_N_rise_rate) <= 0.03 and week_N_lowShadow >= abs(week_N_rise):
                week_N_trend = 10
            elif abs(week_N_rise_rate) <= 0.012:
                week_N_trend = 10
                
                
    #情况（10）时W（N）周线定义：#####################
    if week_1N_trend == -2:
        if week_N_rise_rate < 0:
            if quotes_pre[2][week_N-1] <= quotes_pre[2][week_N - 2]:
                week_N_trend = -1
            elif quotes_pre[2][week_N-1] > quotes_pre[2][week_N - 2] and quotes_pre[4][week_N-1] < quotes_pre[3][week_N - 2]:
                week_N_trend = -1
            elif quotes_pre[2][week_N-1] > quotes_pre[2][week_N - 2] and quotes_pre[4][week_N-1] >= quotes_pre[3][week_N - 2]:
                week_N_trend = 10
        elif week_N_rise_rate > 0:
            if week_1N_rise_rate < 0 and week_N_rise_rate < abs(week_1N_rise_rate) * 0.5:
                week_N_trend = -1
            elif week_1N_rise_rate < 0 and abs(week_1N_rise_rate) * 0.5 < week_N_rise_rate < abs(week_1N_rise_rate):
                week_N_trend = 10
            elif week_1N_rise_rate < 0 and week_N_rise_rate > abs(week_1N_rise_rate):
                week_N_trend = 1            
            
    #情况（11）时W（N）周线定义：#####################
    if week_1N_trend == -1 and week_1N_rise_rate < 0 and abs(week_1N_rise_rate) > 0.03:
        if week_N_rise_rate < 0:
            if quotes_pre[2][week_N-1] <= quotes_pre[2][week_N - 2]:
                week_N_trend = -1
            elif quotes_pre[2][week_N-1] > quotes_pre[2][week_N - 2] and  quotes_pre[4][week_N-1] < quotes_pre[3][week_N - 2]:
                week_N_trend = -1
            elif quotes_pre[2][week_N-1] > quotes_pre[2][week_N - 2] and  quotes_pre[4][week_N-1] >= quotes_pre[3][week_N - 2]:
                week_N_trend = 10
        elif week_N_rise_rate > 0:
            if week_N_rise_rate < abs(week_1N_rise_rate)/3:
                week_N_trend = -1
            elif abs(week_1N_rise_rate)/3 <= week_N_rise_rate <= abs(week_1N_rise_rate) * 0.5:
                week_N_trend = 10
            elif week_N_rise_rate > abs(week_1N_rise_rate) * 0.5:
                week_N_trend = 1

    #情况（12）时W（N）周线定义：#####################
    if week_1N_trend == -1 and week_1N_rise_rate < 0 and 0.012 < abs(week_1N_rise_rate) < 0.03 and week_1N_lowShadow < abs(week_1N_rise):
        if week_N_rise_rate < 0:
            if quotes_pre[2][week_N-1] <= quotes_pre[2][week_N - 2]:
                week_N_trend = -1
            elif quotes_pre[2][week_N-1] > quotes_pre[2][week_N - 2] and quotes_pre[4][week_N-1] < quotes_pre[3][week_N - 2]:
                week_N_trend = -1
            elif quotes_pre[2][week_N-1] > quotes_pre[2][week_N - 2] and quotes_pre[4][week_N-1] >= quotes_pre[3][week_N - 2]:
                week_N_trend = 10
        elif week_N_rise_rate > 0:
            if week_N_rise_rate < abs(week_1N_rise_rate)/3:
                week_N_trend = -1
            elif abs(week_1N_rise_rate)/3 <= week_N_rise_rate <= abs(week_1N_rise_rate) * 0.5:
                week_N_trend = 10
            elif week_N_rise_rate > abs(week_1N_rise_rate) * 0.5:
                week_N_trend = 1
            
    #情况（13）时W（N）周线定义：#####################
    if week_1N_trend == -1 and week_1N_rise_rate < 0 and 0.012 < abs(week_1N_rise_rate) < 0.03 and week_1N_lowShadow > abs(week_1N_rise):
        if week_N_rise_rate < 0:
            if quotes_pre[2][week_N-1] <= quotes_pre[2][week_N - 2]:
                week_N_trend = -1
            elif quotes_pre[2][week_N-1] > quotes_pre[2][week_N - 2] and quotes_pre[4][week_N-1] < quotes_pre[3][week_N - 2]:
                week_N_trend = -1
            elif quotes_pre[2][week_N-1] > quotes_pre[2][week_N - 2] and quotes_pre[4][week_N-1] >= quotes_pre[3][week_N - 2]:
                week_N_trend = 10
        elif week_N_rise_rate > 0:
            if week_N_rise_rate > 0.03:
                week_N_trend = 1
            elif 0.012 < week_N_rise_rate < 0.03 and week_N_upShadow < week_N_rise and week_N_rise_rate > abs(week_1N_rise_rate):
                week_N_trend = 1
            elif 0.012 < week_N_rise_rate < 0.03 and week_N_upShadow < week_N_rise and week_N_rise_rate <= abs(week_1N_rise_rate):
                week_N_trend = 10
            elif 0.012 < week_N_rise_rate < 0.03 and week_N_upShadow >= week_N_rise:
                week_N_trend = 10
            elif week_N_rise_rate <= 0.012:
                week_N_trend = 10


    #情况（14）时W（N）周线定义：#####################
    if week_1N_trend == -1 and week_1N_rise_rate < 0 and abs(week_1N_rise_rate) < 0.012:
        if week_N_rise_rate < 0:
            if abs(week_N_rise_rate) > 0.03:
                week_N_trend = -1
            elif 0.012 < abs(week_N_rise_rate) <= 0.03 and week_N_lowShadow < abs(week_N_rise):
                week_N_trend = -1
            elif 0.012 < abs(week_N_rise_rate) <= 0.03 and week_N_lowShadow >= abs(week_N_rise) and quotes_pre[3][week_N-1] < quotes_pre[3][week_N - 2]:
                week_N_trend = -1
            elif 0.012 < abs(week_N_rise_rate) <= 0.03 and week_N_lowShadow >= abs(week_N_rise) and quotes_pre[3][week_N-1] >= quotes_pre[3][week_N - 2]:
                week_N_trend = 10
            elif abs(week_N_rise_rate) <= 0.012 and quotes_pre[3][week_N-1] < quotes_pre[3][week_N - 2]:
                week_N_trend = -1
            elif abs(week_N_rise_rate) <= 0.012 and quotes_pre[3][week_N-1] >= quotes_pre[3][week_N - 2]:
                week_N_trend = 10
        elif week_N_rise_rate > 0:
            week_N_trend = getSelfTrend(week_N)
            
    #情况（15）时W（N）周线定义：#####################
    if week_1N_trend == -1 and 0 < week_1N_rise_rate <= 0.012:
        if week_N_rise_rate < 0:
            if abs(week_N_rise_rate) > 0.03:
                week_N_trend = -1
            elif 0.012 < abs(week_N_rise_rate) <= 0.03 and week_N_lowShadow < abs(week_N_rise) and \
            quotes_pre[4][week_N-1] <= quotes_pre[3][week_N - 2]:
                week_N_trend = -1
            elif 0.012 < abs(week_N_rise_rate) <= 0.03 and week_N_lowShadow < abs(week_N_rise) and \
            quotes_pre[4][week_N-1] > quotes_pre[3][week_N - 2]:
                week_N_trend = 10
            elif 0.012 < abs(week_N_rise_rate) <= 0.03 and week_N_lowShadow >= abs(week_N_rise) and \
            quotes_pre[4][week_N-1] <= quotes_pre[3][week_N - 2]:
                week_N_trend = -1
            elif 0.012 < abs(week_N_rise_rate) <= 0.03 and week_N_lowShadow >= abs(week_N_rise) and \
            quotes_pre[4][week_N-1] > quotes_pre[3][week_N - 2]:
                week_N_trend = 10
            elif abs(week_N_rise_rate) <= 0.012:
                week_N_trend = 10
        elif week_N_rise_rate > 0:
            if week_N_rise_rate > 0.03 and week_2N_rise_rate < 0 and week_N_rise_rate > abs(week_2N_rise_rate) * 0.5:
                week_N_trend = 1
            elif week_N_rise_rate > 0.03 and week_2N_rise_rate < 0 and week_N_rise_rate <= abs(week_2N_rise_rate) * 0.5:
                week_N_trend = 10
            elif 0.012 < week_N_rise_rate <= 0.03 and week_N_upShadow < week_N_rise and \
            week_2N_rise_rate < 0 and week_N_rise_rate > abs(week_2N_rise_rate) * 0.5:
                week_N_trend = 1
            elif 0.012 < week_N_rise_rate <= 0.03 and week_N_upShadow < week_N_rise and \
            week_2N_rise_rate < 0 and week_N_rise_rate <= abs(week_2N_rise_rate) * 0.5:
                week_N_trend = 10
            elif 0.012 < week_N_rise_rate <= 0.03 and week_N_upShadow >= week_N_rise:
                week_N_trend = 10
            elif week_N_rise_rate <= 0.012:
                week_N_trend = 10
            
    #情况（16）时W（N）周线定义：#####################
    if week_1N_trend ==-1 and 0.012 < week_1N_rise_rate <= 0.03 and week_1N_upShadow > week_1N_rise:
        if week_N_rise_rate < 0:
            if abs(week_N_rise_rate) > 0.03 and quotes_pre[3][week_N-1] < quotes_pre[3][week_N - 2]:
                week_N_trend = -1
            elif abs(week_N_rise_rate) > 0.03 and quotes_pre[3][week_N-1] >= quotes_pre[3][week_N - 2]:
                week_N_trend = 10
            elif 0.012 < abs(week_N_rise_rate) <= 0.03 and week_N_lowShadow < abs(week_N_rise) and \
            quotes_pre[4][week_N-1] <= quotes_pre[3][week_N - 2]:
                week_N_trend = -1
            elif 0.012 < abs(week_N_rise_rate) <= 0.03 and week_N_lowShadow < abs(week_N_rise) and \
            quotes_pre[4][week_N-1] > quotes_pre[3][week_N - 2]:
                week_N_trend = 10
            elif 0.012 < abs(week_N_rise_rate) <= 0.03 and week_N_lowShadow >= abs(week_N_rise) and \
            quotes_pre[4][week_N-1] <= quotes_pre[3][week_N - 2]:
                week_N_trend = -1
            elif 0.012 < abs(week_N_rise_rate) <= 0.03 and week_N_lowShadow >= abs(week_N_rise) and \
            quotes_pre[4][week_N-1] > quotes_pre[3][week_N - 2]:
                week_N_trend = 10
            elif abs(week_N_rise_rate) <= 0.012:
                week_N_trend = 10
        elif week_N_rise_rate > 0:
            if week_N_rise_rate > 0.03 and week_2N_rise_rate < 0 and week_N_rise_rate > abs(week_2N_rise_rate) * 0.5:
                week_N_trend = 1
            elif week_N_rise_rate > 0.03 and week_2N_rise_rate < 0 and week_N_rise_rate <= abs(week_2N_rise_rate) * 0.5:
                week_N_trend = 10
            elif 0.012 < week_N_rise_rate <= 0.03 and week_N_upShadow < week_N_rise and \
            week_2N_rise_rate < 0 and week_N_rise_rate > abs(week_2N_rise_rate) * 0.5:
                week_N_trend = 1
            elif 0.012 < week_N_rise_rate <= 0.03 and week_N_upShadow < week_N_rise and \
            week_2N_rise_rate < 0 and week_N_rise_rate <= abs(week_2N_rise_rate) * 0.5:
                week_N_trend = 10
            elif 0.012 < week_N_rise_rate <= 0.03 and week_N_upShadow > week_N_rise:
                week_N_trend = 10
            if week_N_rise_rate <= 0.012:
                week_N_trend = 10

    #情况（17）时W（N）周线定义：#####################
    if week_1N_trend ==-1 and 0.012 < week_1N_rise_rate <= 0.03 and week_1N_upShadow < week_1N_rise:
        if week_N_rise_rate < 0:
            if abs(week_N_rise_rate) > 0.03 and quotes_pre[3][week_N-1] < quotes_pre[3][week_N - 2]:
                week_N_trend = -1
            elif abs(week_N_rise_rate) > 0.03 and quotes_pre[3][week_N-1] >= quotes_pre[3][week_N - 2]:
                week_N_trend = 10
            elif 0.012 < abs(week_N_rise_rate) <= 0.03 and week_N_lowShadow < abs(week_N_rise) and \
            quotes_pre[4][week_N-1] <= quotes_pre[3][week_N - 2]:
                week_N_trend = -1
            elif 0.012 < abs(week_N_rise_rate) <= 0.03 and week_N_lowShadow < abs(week_N_rise) and \
            quotes_pre[4][week_N-1] > quotes_pre[3][week_N - 2]:
                week_N_trend = 10
            elif 0.012 < abs(week_N_rise_rate) <= 0.03 and week_N_lowShadow >= abs(week_N_rise) and \
            quotes_pre[4][week_N-1] <= quotes_pre[3][week_N - 2]:
                week_N_trend = -1
            elif 0.012 < abs(week_N_rise_rate) <= 0.03 and week_N_lowShadow >= abs(week_N_rise) and \
            quotes_pre[4][week_N-1] > quotes_pre[3][week_N - 2]:
                week_N_trend = 10
            elif abs(week_N_rise_rate) <= 0.012:
                week_N_trend = 10
        elif week_N_rise_rate > 0:
            if week_N_rise_rate > 0.03 and week_2N_rise_rate < 0 and week_N_rise_rate > abs(week_2N_rise_rate) * 0.5:
                week_N_trend = 1
            elif week_N_rise_rate > 0.03 and week_2N_rise_rate < 0 and week_N_rise_rate <= abs(week_2N_rise_rate) * 0.5:
                week_N_trend = 10
            elif 0.012 < week_N_rise_rate <= 0.03 and week_N_upShadow < week_N_rise and \
            week_2N_rise_rate < 0 and week_N_rise_rate > abs(week_2N_rise_rate) * 0.5:
                week_N_trend = 1
            elif 0.012 < week_N_rise_rate <= 0.03 and week_N_upShadow < week_N_rise and \
            week_2N_rise_rate < 0 and week_N_rise_rate <= abs(week_2N_rise_rate) * 0.5:
                week_N_trend = 10
            elif 0.012 < week_N_rise_rate <= 0.03 and week_N_upShadow >= week_N_rise:
                week_N_trend = 10
            if week_N_rise_rate <= 0.012:
                week_N_trend = 10
            
    #情况（18）时W（N）周线定义：#####################
    if week_1N_trend == -1 and week_1N_rise_rate > 0.03:
        if week_N_rise_rate < 0:
            if abs(week_N_rise_rate) > 0.03 and quotes_pre[3][week_N-1] < quotes_pre[3][week_N - 2]:
                week_N_trend = -1
            elif abs(week_N_rise_rate) > 0.03 and quotes_pre[3][week_N-1] >= quotes_pre[3][week_N - 2]:
                week_N_trend = 10
            elif 0.012 < abs(week_N_rise_rate) <= 0.03 and week_N_lowShadow < abs(week_N_rise) :
                week_N_trend = 10
            elif 0.012 < abs(week_N_rise_rate) <= 0.03 and week_N_lowShadow >= abs(week_N_rise) :
                week_N_trend = 10
            elif abs(week_N_rise_rate) <= 0.012:
                week_N_trend = 10
        elif week_N_rise_rate > 0:
            if week_N_rise_rate > 0.03 and week_2N_rise_rate < 0 and week_N_rise_rate > abs(week_2N_rise_rate) * 0.5:
                week_N_trend = 1
            elif week_N_rise_rate > 0.03 and week_2N_rise_rate < 0 and week_N_rise_rate <= abs(week_2N_rise_rate) * 0.5:
                week_N_trend = 10
            elif 0.012 < week_N_rise_rate <= 0.03 and week_N_upShadow < week_N_rise and \
            week_2N_rise_rate < 0 and week_N_rise_rate > abs(week_2N_rise_rate) * 0.5:
                week_N_trend = 1
            elif 0.012 < week_N_rise_rate <= 0.03 and week_N_upShadow < week_N_rise and \
            week_2N_rise_rate < 0 and week_N_rise_rate <= abs(week_2N_rise_rate) * 0.5:
                week_N_trend = 10
            elif 0.012 < week_N_rise_rate <= 0.03 and week_N_upShadow >= week_N_rise:
                week_N_trend = 10
            if week_N_rise_rate <= 0.012:
                week_N_trend = 10

    #情况（19、20）时W（N）周线定义：#####################
    if week_1N_trend in [0, 10]:
        week_N_trend = getSelfTrend(week_N)

#将W(N)周的 selfAdjustTrend 存入全局变量：
    if week_N not in g_selfAdjustTrend_wk: #如全局变量列表中没有数据，则将该值存入全局变量中
        g_selfAdjustTrend_wk.append( week_N )
        g_selfAdjustTrend.append( week_N_trend )
    return week_N_trend

# 5、阻力定义函数： getResistanceList(week_N)##############################################
# 获得第N周的阻力 resistanceList, [0]阻力价格列表，[1]阻力类型，[2]存储阻力来源。阻力由小到达进行排序
def getResistanceList(week_N):

    resistancePrice = [] #存储 week_N 周所有的阻力值。
    resistanceOrigin = [] #存储 week_N 周所有阻力类型。=11，反转阻力；=22，突破阻力；=33，重现阻力。
    resistanceWeekNo = [] #存储 week_N 周所有阻力的来源周。
    resistanceList = [[],[],[]] #阻力列表:[0]存储resistancePrice, [1]存储resistanceOrigin,[2]存储阻力的来源周

    resistancePrice_1N = [] 
    resistanceOrigin_1N = [] 
    resistanceWeekNo_1N = [] 
    resistanceList_1N = [[],[]] 
    
    # week_N的涨幅(跌幅):
    week_N_rise = (quotes_pre[4][week_N-1] - quotes_pre[4][week_N-2]) #上涨（下跌）值
    week_N_rise_rate = week_N_rise / quotes_pre[4][week_N-2] #涨幅(跌幅)
    # week_N的上（下）影线:
    up_shadow = quotes_pre[2][week_N-1] - max(quotes_pre[1][week_N-1], quotes_pre[4][week_N-1]) #上影线
    down_shadow = min(quotes_pre[1][week_N-1], quotes_pre[4][week_N-1]) - quotes_pre[3][week_N-1] #下影线

    quotes_f_list = [[],[]] #将所有波峰，按时间从近到远进行排序，[0]存储价格，[1]存储波峰属于哪一周
    quotes_g_list = [[],[]] #将所有波谷，按时间从近到远进行排序，[0]存储价格，[1]存储波峰属于哪一周
    for i in range( len(quotes_fg_price) )[::-1]:
        if( quotes_fg_direction[i] == 1 and quotes_fg_index[i] < week_N - 2 ):# week_1N 之前（不含week_1N）的波峰
            quotes_f_list[0].append( quotes_fg_price[i] )
            quotes_f_list[1].append( quotes_fg_index[i] )
        elif( quotes_fg_direction[i] == 0 and quotes_fg_index[i] < week_N - 2 ):# week_1N 之前（不含week_1N）的波谷
            quotes_g_list[0].append( quotes_fg_price[i] )
            quotes_g_list[1].append( quotes_fg_index[i] )

    if week_N in [1, 2, 3]:#第三周的阻力，设为默认值为本周阻力：
        resistancePrice.append(0)
        resistanceOrigin.append(33)
        resistanceWeekNo.append(week_N)
        resistancePrice.append(200000)
        resistanceOrigin.append(33)
        resistanceWeekNo.append(week_N)
    elif week_N > 3:#第三周之后某周的阻力，用递归方法得到本周阻力：
#步骤1：得到本周前一周阻力
        if week_N -1 in g_resistanceList_wk:#如全局变量列表中已有数据，则使用全局变量
            index_g_resceList = g_resistanceList_wk.index(week_N -1 )
            resistanceList_1N = copy.deepcopy( g_resistanceList[index_g_resceList] )
        else:            
            resistanceList_1N = getResistanceList(week_N -1)
            
        resistancePrice_1N = copy.deepcopy( resistanceList_1N[0] )
        resistanceOrigin_1N = copy.deepcopy( resistanceList_1N[1] )
        resistanceWeekNo_1N = copy.deepcopy( resistanceList_1N[2] )
        existedResceList = [] #以[[weekNo0,resceType0],,,,]结构存储已有阻力
        for i in range( len(resistancePrice_1N) ):
            existedResceList.append( [resistanceWeekNo_1N[i],resistanceOrigin_1N[i]] )
        
#步骤2：得到本周失效的阻力
        #第一类失效：反转了阻力11，
        reverse = getReverse(week_N)
        if reverse==1: #删除11类阻力
            if 11 in resistanceOrigin_1N:
                resistanceIndex = resistanceOrigin_1N.index(11) # 11类阻力最多只有一个
                g_resceAbateList.append( [resistanceWeekNo_1N[resistanceIndex],11] )
                resistancePrice_1N.pop(resistanceIndex) #删除11阻力价格
                resistanceOrigin_1N.pop(resistanceIndex) #删除11阻力类型
                resistanceWeekNo_1N.pop(resistanceIndex) #删除11阻力类型

        #第二类失效：击穿了阻力11、22、33
        for i in range(len(resistancePrice_1N))[::-1]: #resistancePrice_1N.pop()函数会改变resistancePrice_1N的长度，导致坐标i错位，所以逆序删除。
            resistancePrice = resistancePrice_1N[i]
            if resistancePrice in [0,200000]:#默认阻力[0,200000]不会被击穿
                continue
            elif resistancePrice < quotes_pre[4][week_N - 2]:#阻力小于W(N-1)收盘价，属于W(N)的下阻力
                if quotes_pre[3][week_N - 1] < resistancePrice - 20*tick:
                    g_resceAbateList.append( [resistanceWeekNo_1N[i],resistanceOrigin_1N[i]] )
                    resistancePrice_1N.pop(i)
                    resistanceOrigin_1N.pop(i)
                    resistanceWeekNo_1N.pop(i)
            elif resistancePrice >= quotes_pre[4][week_N - 2]:#阻力大于W(N-1)收盘价，属于W(N)的上阻力
                if quotes_pre[2][week_N - 1] > resistancePrice + 20*tick:
                    g_resceAbateList.append( [resistanceWeekNo_1N[i],resistanceOrigin_1N[i]] )
                    resistancePrice_1N.pop(i)
                    resistanceOrigin_1N.pop(i)
                    resistanceWeekNo_1N.pop(i)

#步骤3：得到本周新增的阻力
        resistancePrice = copy.deepcopy( resistancePrice_1N )
        resistanceOrigin = copy.deepcopy( resistanceOrigin_1N )
        resistanceWeekNo = copy.deepcopy( resistanceWeekNo_1N )
        resceIndex = 0
        
        #第一类阻力来源：产生新反转类阻力11
        if reverse == 1:
            if week_N_rise_rate > 0 and len(quotes_f_list[0]) > 0:
                resceIndex = quotes_f_list[1][0]
                if quotes_f_list[0][0] > quotes_pre[2][week_N-1] - 20*tick and [resceIndex,11] not in existedResceList and [resceIndex,11] not in g_resceAbateList:
                    resistancePrice.append( quotes_f_list[0][0] )
                    resistanceOrigin.append(11)
                    resistanceWeekNo.append( resceIndex )
            elif week_N_rise_rate < 0 and len(quotes_g_list[0]) > 0:
                resceIndex = quotes_g_list[1][0]
                if quotes_g_list[0][0] < quotes_pre[3][week_N-1] + 20*tick and [resceIndex,11] not in existedResceList and [resceIndex,11] not in g_resceAbateList:
                    resistancePrice.append( quotes_g_list[0][0] )
                    resistanceOrigin.append(11)
                    resistanceWeekNo.append( resceIndex )

        #第二类阻力来源：产生新突破类阻力22
        poss_f_Price = 0 #记录上一个可能的阻力值
        poss_g_Price = 200000 #记录可能的阻力值
        if week_N_rise_rate > 0.03 or ( 0.012 < week_N_rise_rate <= 0.03 and up_shadow < week_N_rise ):
            #找符合约束条件的波峰：
            quotes_f_index = -1 # quotes_f_index即为所求波峰
            j1 = 0 # 记录遍历符合条件波峰的个数

            for i in range( len(quotes_f_list[0]) ):
                quotes_f_index = quotes_f_list[1][i]#波峰周在quotes序列中的下标
                if j1 == 5: #如果已查找了5个波峰，直接结束循环
                    break
                
                if quotes_f_list[0][i] > poss_f_Price: #找比f1之前更高的波峰f2
                    if quotes_pre[2][week_N-2] < quotes_pre[2][quotes_f_index] <= quotes_pre[2][week_N-1] - 40*tick and \
                    quotes_pre[4][quotes_f_index] <= quotes_pre[4][week_N-1] - 3*tick:#找到了符合条件的波峰
                        poss_f_Price = quotes_pre[2][quotes_f_index] #将当前波峰价格设为f2，以此为基础找f3
                
                j1 = j1 + 1
        
        if poss_f_Price != 0 and [quotes_f_index,22] not in existedResceList and [quotes_f_index,22] not in g_resceAbateList: #当前可能的阻力不是默认值0，该阻力即为要找的22类阻力
            resistancePrice.append( poss_f_Price )
            resistanceOrigin.append(22)
            resistanceWeekNo.append( quotes_f_index )

        if week_N_rise_rate < -0.03 or ( -0.03 <= week_N_rise_rate < -0.012 and down_shadow < abs(week_N_rise) ):
            #找符合约束条件的波谷：
            quotes_g_index = -1 # quotes_g_index即为所求波谷
            j1 = 0 # 记录遍历符合条件波谷的个数

            for i in range( len(quotes_g_list[0]) ):
                quotes_g_index = quotes_g_list[1][i]#波谷周在quotes序列中的下标
                if j1 == 5: #如果已查找了5个波谷，直接结束循环
                    break
                
                if quotes_g_list[0][i] < poss_g_Price: #找比g1之前更低的波谷g2
                    if quotes_pre[3][week_N-2] > quotes_pre[3][quotes_g_index] >= quotes_pre[3][week_N-1] + 40*tick and \
                    quotes_pre[4][quotes_g_index] >= quotes_pre[4][week_N-1] + 3*tick:#找到了符合条件的波峰
                        poss_g_Price = quotes_pre[3][quotes_g_index] #将当前波谷价格设为g2，以此为基础找g3
                
                j1 = j1 + 1
        if poss_g_Price != 200000 and [quotes_g_index,22] not in existedResceList and [quotes_g_index,22] not in g_resceAbateList: #当前可能的阻力即要找的22类阻力
            resistancePrice.append( poss_g_Price )
            resistanceOrigin.append(22)
            resistanceWeekNo.append( quotes_g_index )
            
        #第三类阻力来源：产生新重现类阻力33的“上阻力”
        f_list_np = np.array(quotes_f_list) # 将阻力数组(list)转换成numpy数组(array)
        f_list_np_order = f_list_np.T[np.lexsort(f_list_np[::-1,:])].T #按第1行价格值，从小到大进行排序
        f_list_np_order_diff = [[],[]]
        f_list_np_order_diff[0] = np.diff(f_list_np_order[0]) 
        f_list_np_order_diff[0] = np.append(f_list_np_order_diff[0], [0]) # f_list_np_order_diff 比 f_list_np_order 少一个元素，用0补上空位。
        f_list_np_order_diff[1] = copy.deepcopy( f_list_np_order[1] )
        
        for i in range( len(f_list_np_order_diff[1]) - 1 ):
            if f_list_np_order_diff[0][i] < 10 * tick: # 即 0<=峰值2-峰值1<10，波峰1是阻力
                # 检查两个波峰间的元素是否符合要求：
                begin = int( min( f_list_np_order_diff[1][i] , f_list_np_order_diff[1][i+1] ) )
                end = int( max( f_list_np_order_diff[1][i] , f_list_np_order_diff[1][i+1] ) )
                quotes_high_between = quotes_pre[2][ begin+1 : end] # 提取在两个波峰间的最高价
                quotes_high_between_np = np.array(quotes_high_between) # 将阻力数组(list)转换成numpy数组(array)
                if len(quotes_high_between_np) == 0 :
                    continue
                elif quotes_high_between_np.max() < f_list_np_order[0][i] and [end,33] not in existedResceList and [end,33] not in g_resceAbateList:
                    #产生一个阻力值:
                    resistancePrice.append( f_list_np_order[0][i] ) #波谷数组中存的价格即为当日最高价
                    resistanceOrigin.append( 33 )
                    resistanceWeekNo.append( end )

        #第三类阻力来源：产生新重现类阻力33的“下阻力”
        g_list_np = np.array(quotes_g_list) # 将阻力数组(list)转换成numpy数组(array)
        g_list_np_order = g_list_np.T[np.lexsort(g_list_np[::-1,:])].T #按第1行价格值，从小到大进行排序
        g_list_np_order_diff = [[],[]]
        g_list_np_order_diff[0] = np.diff(g_list_np_order[0]) 
        g_list_np_order_diff[0] = np.append(g_list_np_order_diff[0], [0]) # g_list_np_order_diff 比 g_list_np_order 少一个元素，用0补上空位。
        g_list_np_order_diff[1] = copy.deepcopy( g_list_np_order[1] )

        for i in range( len(g_list_np_order_diff[1]) - 1 ):
            if g_list_np_order_diff[0][i] < 10 * tick: # 即 0<=谷值2-谷值1<10，波谷2是阻力
                # 检查两个波谷间的元素是否符合要求：
                begin = int( min( g_list_np_order_diff[1][i] , g_list_np_order_diff[1][i+1] ) )
                end = int( max( g_list_np_order_diff[1][i] , g_list_np_order_diff[1][i+1] ) )
                quotes_low_between = quotes_pre[3][ begin+1 : end] # 提取在两个波谷间各点的最低价
                quotes_low_between_np = np.array(quotes_low_between) # 将阻力数组(list)转换成numpy数组(array)
                if len(quotes_low_between_np) == 0 :
                    continue
                elif quotes_low_between_np.min() > g_list_np_order[0][i+1] and [end,33] not in existedResceList and [end,33] not in g_resceAbateList:
                    #产生一个阻力值:
                    resistancePrice.append(g_list_np_order[0][i+1]) #波谷数组中存的价格即为当日最高价
                    resistanceOrigin.append( 33 )
                    resistanceWeekNo.append( end )
                    
    resistanceList[0] = copy.deepcopy( resistancePrice )
    resistanceList[1] = copy.deepcopy( resistanceOrigin )
    resistanceList[2] = copy.deepcopy( resistanceWeekNo )

#将W(N)周的 resistanceList 存入全局变量：
    if week_N not in g_resistanceList_wk: #如全局变量列表中没有数据，则将该值存入全局变量中
        g_resistanceList_wk.append( week_N )
        g_resistanceList.append( resistanceList )
    return resistanceList

# 6、综合周线判断函数：getSynthesizeTrend(week_N)########################################
# 通过W(N-1)的综合周线判断，结合W(N)的阻力空间，判断出W(N)的综合周线判断。
# SynthesizeTrend = =-1表示趋势明显空，=1表示趋势明显多，=10表示宽幅震荡。
def getSynthesizeTrend(week_N):
    synthesizeTrend = 10  #综合周线判断结果,默认为宽幅震荡
    selfAdjustTrend_N = 10
    selfTrend_N = 10
    synthesizeTrend_1N = 10
    
    resistancePrice = [] #上阻力数组
    supportPrice = []    #下阻力数组

    resistanceList = getResistanceList(week_N)
    for i in resistanceList[0]:
        if i > quotes_pre[4][week_N-1]: #阻力大于收盘价，为本周上阻力
            resistancePrice.append(i)
        elif i <= quotes_pre[4][week_N-1]: #阻力小于等于收盘价，为本周下阻力
            supportPrice.append(i)

    selfTrend_N = getSelfTrend( week_N )                  # W(N)的本身函数结果
    if week_N in g_selfAdjustTrend_wk:
        index_g_selfAdjustTrend = g_selfAdjustTrend_wk.index( week_N )
        selfAdjustTrend_N = copy.deepcopy( g_selfAdjustTrend[index_g_selfAdjustTrend] )
    else:
        selfAdjustTrend_N = getSelfAdjustTrend( week_N )      # 得到W(N)的本身调整函数结果

    if week_N in [ 2, 3]: #前3周，用本身函数代替综合周线判断
        synthesizeTrend_1N = getSelfTrend(week_N - 1)
    elif week_N in g_synthesizeTrend_wk:#如全局变量列表中已有数据，则使用全局变量
        index_g_synTrend = g_synthesizeTrend_wk.index( week_N -1 )
        synthesizeTrend_1N = copy.deepcopy( g_synthesizeTrend[index_g_synTrend] )
    else:
        synthesizeTrend_1N = getSynthesizeTrend( week_N - 1 ) # W(N-1)的综合周线判断
    
    effective_resistance = 0 #W(N)的综合阻力，即唯一生效阻力
    resistance_margin = 0    #W(N)的阻力空间

#计算阻力空间
    if synthesizeTrend_1N == -1 or synthesizeTrend_1N == 1 :# W(N-1)是趋势
        if selfAdjustTrend_N == 1: # 情况1，W(N)本身调整是多
            effective_resistance = min(resistancePrice) 
            resistance_margin = abs( quotes_pre[2][week_N-1] - effective_resistance)/quotes_pre[2][week_N-1]
        elif selfAdjustTrend_N == -1: # 情况2，W(N)本身调整是空
            effective_resistance = max(supportPrice) 
            resistance_margin = abs( quotes_pre[3][week_N-1] - effective_resistance)/quotes_pre[3][week_N-1]
        elif selfAdjustTrend_N == 10 and selfTrend_N == 1: # 情况3，W(N)本身调整是震荡，且W(N)本身是上涨
            effective_resistance = min(resistancePrice) 
            resistance_margin = abs( quotes_pre[2][week_N-1] - effective_resistance)/quotes_pre[2][week_N-1]
        elif selfAdjustTrend_N == 10 and selfTrend_N == -1: # 情况4，W(N)本身调整是震荡，且W(N)本身是下跌
            effective_resistance = max(supportPrice)
            resistance_margin = abs( quotes_pre[3][week_N-1] - effective_resistance)/quotes_pre[3][week_N-1]
    elif synthesizeTrend_1N == 10:# W(N-1)的综合周线判断是震荡
        if selfTrend_N == 1 : # 情况5，W(N)本身函数上涨
            effective_resistance = min(resistancePrice) 
            resistance_margin = abs( quotes_pre[2][week_N-1] - effective_resistance)/quotes_pre[2][week_N-1]
        elif selfTrend_N == -1 : # 情况6，W(N)本身函数下跌
            effective_resistance = max(supportPrice)
            resistance_margin = abs( quotes_pre[3][week_N-1] - effective_resistance)/quotes_pre[3][week_N-1]

#根据阻力空间，得到W(N)综合周线判断：
    if abs(resistance_margin) >= 0.03 and synthesizeTrend_1N == 10:
        synthesizeTrend = selfTrend_N
    elif abs(resistance_margin) >= 0.03 and ( synthesizeTrend_1N == 1 or synthesizeTrend_1N == -1 ):
        synthesizeTrend = selfAdjustTrend_N
    elif abs(resistance_margin) < 0.03 :
        synthesizeTrend = 10

#将W(N)周的 synthesizeTrend 存入全局变量：
    if week_N not in g_synthesizeTrend_wk: #如全局变量列表中没有数据，则将该值存入全局变量中
        g_synthesizeTrend_wk.append( week_N )
        g_synthesizeTrend.append( synthesizeTrend )

    return synthesizeTrend

# 6.1、综合阻力函数： getEffeResistance(week_N)########################################
# 得到W(N)的综合阻力（即唯一有效的阻力）。

def getEffeResistance(week_N):
    selfAdjustTrend_N = 10
    synthesizeTrend_1N = 10
    selfTrend_N = 10
    
    resistancePrice = [] #上阻力数组
    supportPrice = []    #下阻力数组

    resistanceList = getResistanceList(week_N)
    for i in resistanceList[0]:
        if i > quotes_pre[4][week_N-1]: #阻力大于收盘价，为本周上阻力
            resistancePrice.append(i)
        elif i <= quotes_pre[4][week_N-1]: #阻力小于等于收盘价，为本周下阻力
            supportPrice.append(i)

    selfTrend_N = getSelfTrend( week_N )                  # W(N)的本身函数结果
    if week_N in g_selfAdjustTrend_wk:
        index_g_selfAdjustTrend = g_selfAdjustTrend_wk.index( week_N )
        selfAdjustTrend_N = copy.deepcopy( g_selfAdjustTrend[index_g_selfAdjustTrend] )
    else:
        selfAdjustTrend_N = getSelfAdjustTrend( week_N )      # 得到W(N)的本身调整函数结果

    if week_N in [ 2, 3]: #前3周，用本身函数代替综合周线判断
        synthesizeTrend_1N = getSelfTrend(week_N - 1)
    elif week_N in g_synthesizeTrend_wk:#如全局变量列表中已有数据，则使用全局变量
        index_g_synTrend = g_synthesizeTrend_wk.index( week_N - 1 )
        synthesizeTrend_1N = copy.deepcopy( g_synthesizeTrend[index_g_synTrend] )
    else:
        synthesizeTrend_1N = getSynthesizeTrend( week_N - 1 ) # W(N-1)的综合周线判断
    
    effective_resistance = 0 #W(N)的综合阻力，即唯一生效阻力

#计算阻力空间
    if synthesizeTrend_1N == -1 or synthesizeTrend_1N == 1 :# W(N-1)是趋势
        if selfAdjustTrend_N == 1: # 情况1，W(N)本身调整是多
            effective_resistance = min(resistancePrice) 
        elif selfAdjustTrend_N == -1: # 情况2，W(N)本身调整是空
            effective_resistance = max(supportPrice) 
        elif selfAdjustTrend_N == 10 and selfTrend_N == 1: # 情况3，W(N)本身调整是震荡，且W(N)本身是上涨
            effective_resistance = min(resistancePrice) 
        elif selfAdjustTrend_N == 10 and selfTrend_N == -1: # 情况4，W(N)本身调整是震荡，且W(N)本身是下跌
            effective_resistance = max(supportPrice)
    elif synthesizeTrend_1N == 10:# W(N-1)的综合周线判断是震荡
        if selfTrend_N == 1 : # 情况5，W(N)本身函数上涨
            effective_resistance = min(resistancePrice) 
        elif selfTrend_N == -1 : # 情况6，W(N)本身函数下跌
            effective_resistance = max(supportPrice)

    return effective_resistance

# 7、最终周线判断函数：getFinalTrend( week_N)##############################
# 以 W(N)的综合周线判断函数为输入，先判断W(N)是宽幅震荡还是窄幅震荡，再判断W(N)是否为首周，输出最终结果finalTrend
# weekTrend=-1表示本周周线定义为趋势明显空，=-2表示首周空;
# =1表示趋势明显多，=2表示首周多;=10表示宽幅震荡，=0表示窄幅震荡。
def getFinalTrend( week_N ):
    
    finalTrend = 10 #默认为宽幅震荡
    synthesizeTrend_wn = 10
    if week_N in g_synthesizeTrend_wk: #如全局变量中已有数据，则使用全局变量
        index_g_synTrend = g_synthesizeTrend_wk.index( week_N )
        synthesizeTrend_wn = copy.deepcopy( g_synthesizeTrend[index_g_synTrend] )
    else:
        synthesizeTrend_wn = getSynthesizeTrend(week_N)
        
    reverse_wn = getReverse(week_N)
    break_list_wn = getBreak(week_N)
    # week_N的涨幅(跌幅):
    week_N_rise = (quotes_pre[4][week_N-1] - quotes_pre[4][week_N-2]) #上涨（下跌）值
    week_N_rise_rate = week_N_rise / quotes_pre[4][week_N-2] #涨幅(跌幅)
    # week_2N的涨幅(跌幅):
    week_2N_rise = (quotes_pre[4][week_N-3] - quotes_pre[4][week_N-4]) #上涨（下跌）值
    week_2N_rise_rate = week_2N_rise / quotes_pre[4][week_N-4] #涨幅(跌幅)
    
    up_shadow = quotes_pre[2][week_N-1] - max(quotes_pre[1][week_N-1], quotes_pre[4][week_N-1]) #上影线
    down_shadow = min(quotes_pre[1][week_N-1], quotes_pre[4][week_N-1]) - quotes_pre[3][week_N-1] #下影线

#先判断宽幅震荡还是窄幅震荡
    finalTrend = synthesizeTrend_wn # finalTrend 的初始值，即为synthesizeTrend_wn，后面的函数在此基础上调整finalTrend的值
    if synthesizeTrend_wn == 10:
        max_3weeks_h = max( quotes_pre[2][week_N-1], quotes_pre[2][week_N-2], quotes_pre[2][week_N-3] )
        min_3weeks_l = min( quotes_pre[3][week_N-1], quotes_pre[3][week_N-2], quotes_pre[3][week_N-3] )
        max_amplitude = (max_3weeks_h - min_3weeks_l)/min_3weeks_l
        if max_amplitude >= 0.05 :
            finalTrend = 10 #宽幅震荡
        else:
            finalTrend = 0 #窄幅震荡
#再判断是否为首周。如果判断条件中，既是突破周，又是反转周，则认作突破周
    elif synthesizeTrend_wn == 1 and break_list_wn[1] in [1, -1] : #情况3
        week_f = break_list_wn[2]
        week_f_rise = (quotes_pre[4][week_f - 1] - quotes_pre[4][week_f - 2]) #上涨（下跌）值
        week_f_rise_rate = week_f_rise / quotes_pre[4][week_f - 2] #涨幅(跌幅)
        if abs(week_f_rise_rate)<= 0.008:
            finalTrend = 2 #首周多
        elif abs(week_f_rise_rate) > 0.008 and week_N - week_f >=3:
            finalTrend = 2 #首周多

    elif synthesizeTrend_wn == -1 and break_list_wn[1] in [1, -1] :#情况4
        week_g = break_list_wn[2]
        week_g_rise = (quotes_pre[4][week_g - 1] - quotes_pre[4][week_g - 2]) #上涨（下跌）值
        week_g_rise_rate = week_g_rise / quotes_pre[4][week_g - 2] #涨幅(跌幅)
        if abs(week_g_rise_rate)<= 0.008:
            finalTrend = -2 #首周空
        elif abs(week_g_rise_rate) > 0.008 and week_N - week_g >=3:
            finalTrend = -2 #首周空
            
    elif synthesizeTrend_wn == 1 and reverse_wn == 1:#情况1
        if week_N_rise_rate>0.03 and week_2N_rise_rate < 0:
            finalTrend = 2 #首周多
        elif 0.012<week_N_rise_rate<0.03 and up_shadow < week_N_rise and week_2N_rise_rate < 0:
            finalTrend = 2 #首周多
            
    elif synthesizeTrend_wn == -1 and reverse_wn == 1:#情况2
        if week_N_rise_rate<0 and abs(week_N_rise_rate)>0.03 and week_2N_rise_rate > 0:
            finalTrend = -2 #首周空
        elif week_N_rise_rate<0 and 0.012< abs(week_N_rise_rate) <0.03 and down_shadow < week_N_rise and week_2N_rise_rate > 0:
            finalTrend = -2 #首周空
    
#将W(N)周的 finalTrend 存入全局变量：
    if week_N not in g_week_trend_wk: #如全局变量列表中没有数据，则将该值存入全局变量中
        g_week_trend_wk.append( week_N )
        g_week_trend.append( finalTrend )

    return finalTrend

###############################################################################
############################ 三、测试函数#######################################
# 测试这些函数：反转周函数getReverse、突破周函数getBreak、阻力定义函数getResistance、
# 反转首周定义函数getFirstReverse、突破首周定义函数getFirstBreak、本身K线含义函数getSelfTrend、
# 阻力空间函数getBreakTrend、本身定义调整函数getSelfAdjustTrend
alldays = DayLocator()  
months = MonthLocator() 
month_formatter = DateFormatter("%b %Y")

print("-------------------Test All Functions---------------------------")
for week_N in np.arange(3,41,1):
#for week_N in np.arange(32,34,1):
#week_N = 4
#if week_N == 4:
    reverse_wk = getReverse(week_N)
    break_list = getBreak(week_N)
    break_wk = break_list[1]
    resistance_list = getResistanceList(week_N)
    weekTrend = getFinalTrend( week_N )

    testResult = ["", "", "", "", "", ""]
    resistance_price = "resistance price:"
    
    testResult[0] = "Week {0}'s Judge:---".format(week_N)
    #1、是否反转
    if reverse_wk ==0:
        testResult[1] = "Reverse Week: No;"
    elif reverse_wk ==1:
        testResult[1] = "Reverse Week: Yes;"
        
    #2、是否突破
    if break_wk ==0:
        testResult[2] = "Break Week: No;"
    elif break_wk in [1,-1]:
        testResult[2] = "Break Week: Yes, WeekNo = {0};".format(break_list[2])

    #3、综合阻力
    if len(resistance_list[0]) != 0:
        for i in range( len(resistance_list[0]) ):
            resistance_price = resistance_price + "{0};".format(resistance_list[0][i])
        testResult[3] = resistance_price
    else:
        testResult[3] = "resistance price: none;"

    #4、周线定义
    if weekTrend == -1:
        testResult[5] = "Week Trend: negative;"
    elif weekTrend == -2:
        testResult[5] = "Week Trend: first negative;"
    elif weekTrend == 1:
        testResult[5] = "Week Trend: positive;"
    elif weekTrend == 2:
        testResult[5] = "Week Trend: first positive;"
    elif weekTrend == 10:
        testResult[5] = "Week Trend: fluctuate violently;"
    elif weekTrend == 0:
        testResult[5] = "Week Trend: fluctuate narrowly;"
    for i in range(5):
        print(testResult[i])

### 图形化显示本周属性：    ########################################
    fig = plt.figure( "Week {0}'s".format(week_N), figsize=(18, 9) ) 
    ax = fig.add_subplot(111)
    ax.xaxis.set_major_locator(months) 
    ax.xaxis.set_minor_locator(alldays) 
    ax.xaxis.set_major_formatter(month_formatter) 

    #plt.title("Week {0}'s Weekly Line".format(week_N) )
    fig_title = testResult[0] + testResult[1] + "\n" + \
                testResult[2] + "\n" + \
                testResult[3] + testResult[4] + "\n" + \
                testResult[5] + "\n"

    plt.title( fig_title )
    plt.xlabel("Week")
    plt.ylabel("Price")
    candlestick_ohlc(ax,quotes,width=2.0, colorup='r',colordown='g') 
    
    quotes_fg_date_extd = []
    quotes_fg_price_extd = []
    quotes_fg_date_extd = np.append(quotes_pre[0][0], quotes_fg_date)
    quotes_fg_date_extd = np.append(quotes_fg_date_extd, quotes_pre[0][-1])
    
    if quotes_fg_direction[0] == 0: #第一个极值为波谷
        quotes_fg_price_extd = np.append(quotes_pre[2][0], quotes_fg_price)
    elif quotes_fg_direction[0] == 1: #第一个极值为波峰
        quotes_fg_price_extd = np.append(quotes_pre[3][0], quotes_fg_price)
        
    if quotes_fg_direction[-1] == 0: #最后一个极值为波谷
        quotes_fg_price_extd = np.append(quotes_fg_price_extd, quotes_pre[2][-1])
    elif quotes_fg_direction[-1] == 1: #最后一个极值为波峰
        quotes_fg_price_extd = np.append(quotes_fg_price_extd, quotes_pre[3][-1])
        
    plt.plot(quotes_fg_date_extd, quotes_fg_price_extd, 'r', lw=1.0) #画出调整后的峰谷折线图
    ax.scatter(quotes_fg_date, quotes_fg_price, alpha=0.5) #画出调整后的峰谷散点图
    #画出阻力线：
#    if len(resistance_list[0]) > 0:
#        for i in range( len(resistance_list[0]) ):
#            if resistance_list[0][i] not in [0, 200000]:
#                plt.plot([quotes_pre[0][0],quotes_pre[0][-1]], [resistance_list[0][i],resistance_list[0][i]], 'b', lw=0.5 )#画出所有阻力线
#                plt.text(quotes_pre[0][i], resistance_list[0][i], i-1, color='black', alpha=0.5)
    effective_resistance = getEffeResistance(week_N)
    if effective_resistance not in [0, 200000]:
        plt.plot([quotes_pre[0][0],quotes_pre[0][-1]], [effective_resistance,effective_resistance], 'b', lw=0.5 )#画出有效阻力线
        plt.text(quotes_pre[0][0], effective_resistance, "resistance:{0}".format(effective_resistance), color='black', alpha=0.5)
    #标注各周编号
    wk_no = range(len(quotes_pre[0]))  # 周序号
    for i in range(len(quotes_pre[0])):
        wk_number = wk_no[i] + 1
        if i == week_N - 1:#突出标注本周编号
            plt.text(quotes_pre[0][i], quotes_pre[2][i], "({0})".format(wk_number), color='red', withdash=True)
        else:
            plt.text(quotes_pre[0][i], quotes_pre[2][i], wk_number, alpha=0.5)

    fig.autofmt_xdate()
    #plt.savefig("RU1801_Week {0}.png".format(week_N))
    #plt.savefig("RU1709_Week {0}.png".format(week_N))
    #plt.savefig("Y1801_Week {0}.png".format(week_N))
    plt.savefig("RM801_Week {0}.png".format(week_N))
    #plt.savefig("RB1801_Week {0}.png".format(week_N))
    #plt.savefig("J1801_Week {0}.png".format(week_N))
    
plt.show()




