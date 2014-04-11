
import os
import csv
import math
import numpy as np
import pandas as pd
import statsmodels.formula.api as sm
import seaborn as sns
from pylab import *

Punt = [[-2.505152, -2.087459, 1.306882, -1.127434], [0.050217, 0.087383, 0.060661, 0.080226], [0.007299, -0.001777, 0.013215, -0.016401], [-0.166807, -0.942778, -0.397772, -1.485986], [0.018914, 0.028430, 0.014893, 0.038073], [0.255890, 0.602383, 0.258313, 1.120144]]
Punt = pd.DataFrame(Punt)
Punt.columns = ["DefTD","FG","NoPoints","TD"]
Punt.index = ["intercept", "removelast_ydline", "offrankMid", "offrank31t32", "defrankMid", "defrank31t32"]

score = [[17.102051, 23.383689, 22.702956, 23.697687], [-0.189247, -0.238087, -0.204967, -0.234290], [0.006484, -0.005221, 0.007977, -0.020917], [0.023844, -0.741291, -0.249409, -1.376970], [0.010301, 0.021678, 0.010377, 0.030189], [0.458122, 0.321432, 0.048783, 0.718639]]
score = pd.DataFrame(score)
score.columns = ["DefTD","FG","NoPoints","TD"]
score.index = ["intercept", "ydline", "offrankMid", "offrank31t32", "defrankMid", "defrank31t32"]

fg_vec = [3.603047, -0.098109]

gfi_vec = [0.580768, -0.119942, -0.007568, -0.455296, 0.011190, 0.484075]

def init(ydline, ydtogo, oorank, odrank, ddrank, dorank):
    oorank1 = 2 if 5 <= oorank <= 30 else 3 if 31 <= oorank <= 32 else 0
    Ioorank = 1 if oorank1 == 0 else 1 if oorank1 == 3 else oorank
    ddrank1 = 4 if 5 <= ddrank <= 30 else 5 if 31 <= ddrank <= 32 else 0
    Iddrank = 1 if ddrank1 == 0 else 1 if ddrank1 == 3 else ddrank
    odrank1 = 4 if 5 <= odrank <= 30 else 5 if 31 <= odrank <= 32 else 0
    Iodrank = 1 if odrank1 == 0 else 1 if odrank1 == 3 else odrank
    dorank1 = 2 if 5 <= dorank <= 30 else 3 if 31 <= dorank <= 32 else 0
    Idorank = 1 if dorank1 == 0 else 1 if dorank1 == 3 else dorank
    X_off = [1, ydline - ydtogo, 0, 0, 0, 0]; X_off[oorank1] = Ioorank; X_off[ddrank1] = Iddrank
    X_def_score = [1, 100 - (ydline - ydtogo)*(2/5), 0, 0, 0, 0]; X_def_score[dorank1] = Idorank; X_def_score[odrank1] = Iodrank
    X_def_gfi_fail = [1, 100 - ydline, 0, 0, 0, 0]; X_def_gfi_fail[dorank1] = Idorank; X_def_gfi_fail[odrank1] = Iodrank
    X_def_20 = [1, 80, 0, 0, 0, 0]; X_def_20[dorank1] = Idorank; X_def_20[odrank1] = Iodrank
    X_def_fg_fail = [1, 93 - ydline, 0, 0, 0, 0]; X_def_fg_fail[dorank1] = Idorank; X_def_fg_fail[odrank1] = Iodrank
    X_punt = [1, ydline, 0, 0, 0, 0]; X_punt[dorank1] = Idorank; X_punt[odrank1] = Iodrank
    X = {'off': X_off, 'def_score': X_def_score, 'gfi_fail': X_def_gfi_fail, '20': X_def_20, 'fg_fail': X_def_fg_fail, 'punt': X_punt}   
    return X

def prob(ydline, ydtogo, oorank, odrank, ddrank, dorank):
    oorank1 = 2 if 5 <= oorank <= 30 else 3 if 31 <= oorank <= 32 else 0
    Ioorank = 1 if oorank1 == 0 else 1 if oorank1 == 3 else oorank
    ddrank1 = 4 if 5 <= ddrank <= 30 else 5 if 31 <= ddrank <= 32 else 0
    Iddrank = 1 if ddrank1 == 0 else 1 if ddrank1 == 3 else ddrank
    oorank_gfi = 2 if 5 <= oorank <= 30 else 3 if 31 <= oorank <= 32 else 0
    ddrank_gfi = 4 if 5 <= ddrank <= 30 else 5 if 31 <= ddrank <= 32 else 0
    vec = [1, ydtogo, 0, 0, 0, 0]; vec[oorank_gfi] = Ioorank; vec[ddrank_gfi] = Iddrank
    fg = exp((1*fg_vec[0] + ydline*(fg_vec[1])))/(1 + exp((fg_vec[0] + ydline*(fg_vec[1]))))
    gfi = exp((vec[0]*gfi_vec[0] + vec[1]*(gfi_vec[1]) + vec[2]*(gfi_vec[2]) * vec[3]*(gfi_vec[3]) + vec[4]*(gfi_vec[4]) + vec[5]*(gfi_vec[5])))/(1 + exp(vec[0]*gfi_vec[0] + vec[1]*(gfi_vec[1]) + vec[2]*(gfi_vec[2]) + vec[3]*(gfi_vec[3]) + vec[4]*(gfi_vec[4]) + vec[5]*(gfi_vec[5])))
    X = {'fg': fg, 'gfi': gfi}
    return X

def log_score(ydline, ydtogo, oorank, odrank, ddrank, dorank, vec):
    Xsum = (1 + exp(sum(vec*score.ix[:,0])) + exp(sum(vec*score.ix[:,1])) + exp(sum(vec*score.ix[:,2])) + exp(sum(vec*score.ix[:,3])))
    DefTD = exp(sum(vec*score.ix[:,0]))/Xsum
    FG = exp(sum(vec*score.ix[:,1]))/Xsum
    NoPoints = exp(sum(vec*score.ix[:,2]))/Xsum
    TD = exp(sum(vec*score.ix[:,3]))/Xsum
    DefSafety = 1/Xsum
    X = {'DefTD': DefTD, 'FG': FG, 'NoPoints': NoPoints, 'TD': TD, 'DefSafety': DefSafety}
    return X

def log_punt(ydline, ydtogo, oorank, odrank, ddrank, dorank, vec):
    Xsum = (1 + exp(sum(vec*Punt.ix[:,0])) + exp(sum(vec*Punt.ix[:,1])) + exp(sum(vec*Punt.ix[:,2])) + exp(sum(vec*Punt.ix[:,3])))
    DefTD = exp(sum(vec*Punt.ix[:,0]))/Xsum
    FG = exp(sum(vec*Punt.ix[:,1]))/Xsum
    NoPoints = exp(sum(vec*Punt.ix[:,2]))/Xsum
    TD = exp(sum(vec*Punt.ix[:,3]))/Xsum
    DefSafety = 1/Xsum
    X = {'DefTD': DefTD, 'FG': FG, 'NoPoints': NoPoints, 'TD': TD, 'DefSafety': DefSafety}
    return X

def gfi_expect(ydline, ydtogo, oorank, odrank, ddrank, dorank):
    
    x = init(ydline, ydtogo, oorank, odrank, ddrank, dorank)
    y = prob(ydline, ydtogo, oorank, odrank, ddrank, dorank)
    X20 = log_score(ydline, ydtogo, oorank, odrank, ddrank, dorank, x['20'])
    XDS = log_score(ydline, ydtogo, oorank, odrank, ddrank, dorank, x['def_score'])
    XGFI = log_score(ydline, ydtogo, oorank, odrank, ddrank, dorank, x['gfi_fail'])
    XOFF = log_score(ydline, ydtogo, oorank, odrank, ddrank, dorank, x['off'])

    if ydline <= ydtogo:
        E_gfi = y['gfi']*(7 - (X20['DefTD']*(-7) + X20['FG']*(3) + X20['TD']*(7) + X20['DefSafety']*(-2))) - (1 - y['gfi'])*(XGFI['DefTD']*(-7) + XGFI['FG']*(3) + XGFI['TD']*(7) + XGFI['DefSafety']*(-2))
    else:
        E_gfi = y['gfi']*((XOFF['FG']*(3) + XOFF['TD']*(7) + XOFF['DefSafety']*(-2) + XOFF['DefTD']*(-7)) - (XOFF['FG'] + XOFF['TD'] + XOFF['DefSafety'])*(X20['DefTD']*(-7) + X20['FG']*(3) + X20['TD']*(7) + X20['DefSafety']*(-2)) - XOFF['NoPoints']*(XDS['FG']*(3) + XDS['TD']*(7) + XDS['DefSafety']*(-2) + XDS['DefTD']*(-7))) - (1 - y['gfi'])*(XGFI['DefTD']*(-7) + XGFI['FG']*(3) + XGFI['TD']*(7) + XGFI['DefSafety']*(-2))
    return E_gfi

def fg_expect(ydline, ydtogo, oorank, odrank, ddrank, dorank):
    
    x = init(ydline, ydtogo, oorank, odrank, ddrank, dorank)
    y = prob(ydline, ydtogo, oorank, odrank, ddrank, dorank)
    X20 = log_score(ydline, ydtogo, oorank, odrank, ddrank, dorank, x['20'])
    XFG = log_score(ydline, ydtogo, oorank, odrank, ddrank, dorank, x['fg_fail'])
   
    E_fg = y['fg']*(3 - (X20['DefTD']*(-7) + X20['FG']*(3) + X20['TD']*(7) + X20['DefSafety']*(-2))) - (1 - y['fg'])*(XFG['DefTD']*(-7) + XFG['FG']*(3) + XFG['TD']*(7) + XFG['DefSafety']*(-2))
    return E_fg

def punt_expect(ydline, ydtogo, oorank, odrank, ddrank, dorank):
    x = init(ydline, ydtogo, oorank, odrank, ddrank, dorank)
    XP = log_punt(ydline, ydtogo, oorank, odrank, ddrank, dorank, x['punt'])
    E_punt = - (XP['DefTD']*(-7) + XP['FG']*(3) + XP['TD']*(7) + XP['DefSafety']*(-2))
    return E_punt

def decision(ydline, ydtogo, oorank, odrank, ddrank, dorank):
    gfi = gfi_expect(ydline, ydtogo, oorank, odrank, ddrank, dorank)
    fg = fg_expect(ydline, ydtogo, oorank, odrank, ddrank, dorank)
    punt = punt_expect(ydline, ydtogo, oorank, odrank, ddrank, dorank)
    dec = {'Go For It': gfi, 'Field Goal': fg, 'Punt': punt}
    return dec

def test_init():
    result = init(50, 10, 5, 5, 6, 6)
    print 'This is a test; the value of result is', result
    assert result == {'20': [1, 80, 6, 0, 5, 0], 'def_score': [1, 100, 6, 0, 5, 0], 'fg_fail': [1, 43, 6, 0, 5, 0], 'gfi_fail': [1, 50, 6, 0, 5, 0],'off': [1, 40, 5, 0, 6, 0], 'punt': [1, 50, 6, 0, 5, 0]} 
    
def test_prob():
    result = prob(50, 10, 5, 5, 6, 6)
    print 'hello, this is a test; the value of result is', result
    assert result == {'fg': 0.2137608737772787, 'gfi': 0.37054292453220239}
    
def test_log_score():
    result = log_score(50, 10, 5, 5, 6, 6, [1, 10, 5, 0, 6, 0])
    print 'hello, this is a test; the value of result is', result
    assert result == {'DefSafety': 2.2052541609623836e-10, 'DefTD': 0.00097682683621591582, 'FG': 0.32360677361954143, 'NoPoints': 0.22773441808312658, 'TD': 0.44768198124059067}

def test_log_punt():
    result = log_punt(50, 10, 5, 5, 6, 6, [1, 10, 5, 0, 6, 0])
    print 'hello, this is a test; the value of result is', result
    assert result == {'DefSafety': 0.097478634178799317, 'DefTD': 0.015281305481536707, 'FG': 0.034045546032633892, 'NoPoints': 0.77167094133367564, 'TD': 0.081523572973354372}
    
def test_gfi():
    result = gfi_expect(50, 10, 5, 5, 6, 6)
    print 'hello, this is a test; the value of result is', result
    assert result == -0.90330920543553173
    
def test_fg():
    result = fg_expect(50, 10, 5, 5, 6, 6)
    print 'hello, this is a test; the value of result is', result
    assert result == -1.9430718453559772
    
def test_punt():
    result = punt_expect(50, 10, 5, 5, 6, 6)
    print 'hello, this is a test; the value of result is', result
    assert result == -1.312820991036932
    
def test_decision():
    result = decision(50, 10, 5, 5, 6, 6)
    print 'hello, this is a test; the value of result is', result
    assert result == {'Field Goal': -1.9430718453559772, 'Go For It': -0.90330920543553173, 'Punt': -1.312820991036932}