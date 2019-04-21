
import numpy as np
import pandas as pd
from scipy.stats import norm
import time
import sobol_seq
%matplotlib inline

def GBM(Ttm, TradingDaysInAYear, NoOfPaths, UnderlyingPrice, RiskFreeRate, Volatility, \
        AntitheticPaths = True, MomentMatching = True, QuasiRandom = True):
    dt = float(Ttm) / TradingDaysInAYear
    paths = np.zeros((TradingDaysInAYear , NoOfPaths), np.float64)
    paths[0] = UnderlyingPrice

    for t in range(1, TradingDaysInAYear ):
        if AntitheticPaths:
            rand = np.random.standard_normal(NoOfPaths/2)
            rand = np.concatenate((rand, -rand))
            
        elif MomentMatching:
            rand = np.random.standard_normal(NoOfPaths)
            rand = (rand - np.mean(rand))/np.std(rand)
#             rand = rand / np.std(rand)
        
        elif QuasiRandom:
            rand = i4_sobol_generate_std_normal(1, NoOfPaths)
            lRand = []
            for i in range(len(rand)):
                a = rand[i][0]
                lRand.append(a)
            rand = np.array(lRand)

        else:
            rand = np.random.standard_normal(NoOfPaths)
        paths[t] = paths[t - 1] * np.exp((RiskFreeRate - 0.5 * Volatility ** 2) * dt + Volatility * np.sqrt(dt) * rand)
    return paths

def BlackScholes(Ttm, UnderlyingPrice, Strike, RiskFreeRate, Volatility, OptionType):

    d1 = (np.log(UnderlyingPrice/Strike) + (RiskFreeRate + 0.5 * Volatility ** 2) * Ttm)/(Volatility * np.sqrt(Ttm))
    d2 = (np.log(UnderlyingPrice/Strike) + (RiskFreeRate - 0.5 * Volatility ** 2) * Ttm)/(Volatility * np.sqrt(Ttm))

    if OptionType == 'call':
        value = (UnderlyingPrice * norm.cdf(d1, 0.0, 1.0) - \
                      Strike * np.exp(-RiskFreeRate*Ttm) * norm.cdf(d2, 0.0, 1.0))
#         elif OptionType == 'put':
#             self.value = (Strike * np.exp(-RiskFreeRate*Ttm) * norm.cdf(self.-d2, 0.0, 1.0) - \
#                           UnderlyingPrice * norm.cdf(self.-d1, 0.0, 1.0))
    return value

def Payoff(Ttm, TradingDaysInAYear, NoOfPaths, UnderlyingPrice, RiskFreeRate, Volatility, Strike, OptionType, \
           AntitheticPaths, MomentMatching, QuasiRandom):

    DiscountFactor = np.exp(-RiskFreeRate*Ttm)

    if OptionType == 'call':
        PayoffAtMaturity = np.maximum(0, GBM(Ttm, TradingDaysInAYear, NoOfPaths, UnderlyingPrice, RiskFreeRate, Volatility,\
                                             AntitheticPaths, MomentMatching, QuasiRandom)[-1] - Strike)

        AveragePayoffAtMaturity = np.average(PayoffAtMaturity)
        DiscountedPayoff = DiscountFactor * AveragePayoffAtMaturity

    return {"bs":DiscountedPayoff, "PayOffMat":PayoffAtMaturity}

def Main(Ttm, TradingDaysInAYear, NoOfPaths, UnderlyingPrice, RiskFreeRate, Volatility, Strike, OptionType, \
         AntitheticPaths, MomentMatching, QuasiRandom):

    BSPrice = BlackScholes(Ttm, UnderlyingPrice, Strike, RiskFreeRate, Volatility, OptionType)
    MCPrice = Payoff(Ttm, TradingDaysInAYear, NoOfPaths, UnderlyingPrice, RiskFreeRate, Volatility,\
                     Strike, OptionType, AntitheticPaths, MomentMatching, QuasiRandom)["bs"]
    MCStdErr = np.std(Payoff(Ttm, TradingDaysInAYear, NoOfPaths, UnderlyingPrice, RiskFreeRate, Volatility, Strike, \
                             OptionType, AntitheticPaths, MomentMatching, QuasiRandom)["PayOffMat"])/np.sqrt(NoOfPaths)
    
    return pd.DataFrame({"BS":[BSPrice], "MC":[MCPrice], "SE":[MCStdErr], "nSims":[NoOfPaths]})
	
	
# Antithetic
np.random.seed(123)
lNumberOfPaths = [100, 500, 1000, 2000, 3000, 4000, 5000, 8000]
ldf = []
for NPaths in lNumberOfPaths:
    start = time.clock()
    df = Main(T, N, NPaths, S0, r, v, K, O, True, False)
    Runtime = time.clock() - start
    df['RunTime'] = Runtime
    ldf.append(df)
    
# dfFinal = ldf[0]
# dfFinal = dfFinal.join(ldf[1:],ignore_index = True)
dfFinal = pd.concat(ldf)
dfFinal

np.random.seed(123)
lNumberOfPaths = [100, 500, 1000, 2000, 3000, 4000, 5000, 8000]
ldf = []
for NPaths in lNumberOfPaths:
    start = time.clock()
    df = Main(T, N, NPaths, S0, r, v, K, O, False, False)
    Runtime = time.clock() - start
    df['RunTime'] = Runtime
    ldf.append(df)
    
# dfFinal = ldf[0]
# dfFinal = dfFinal.join(ldf[1:],ignore_index = True)
dfFinal = pd.concat(ldf)
dfFinal

dfFinal.plot(x="nSims")

# Data
# Trade Details
S0 = 100.
K = 100.
r = 0.05
v = 0.5
T = 1
N = 252
O = "call"

# MC Config
P = 10000