import numpy as np
import pandas as pd
from scipy.stats import norm
%matplotlib inline

class TradeDetails(object):
    def __init__(self, UnderlyingPrice, Strike, RiskFreeRate, Volatility, Ttm, TradingDaysInAYear, OptionType):
        self.UnderlyingPrice = UnderlyingPrice
        self.Strike = Strike
        self.RiskFreeRate = RiskFreeRate
        self.Volatility = Volatility
        self.Ttm = Ttm
        self.TradingDaysInAYear = TradingDaysInAYear
        self.OptionType = OptionType
        
class MonteCarloConfig(object):
     def __init__(self, NoOfPaths):
         self.NoOfPaths=NoOfPaths
            
class Simulation(object):
    def __init__(self, TradeDetails, MonteCarloConfig):
        self.TradeDetails = TradeDetails
        self.MonteCarloConfig = MonteCarloConfig
    
    def GBM(self):
        self.dt = float(Ttm) / TradingDaysInAYear
        self.paths = np.zeros((TradingDaysInAYear + 1, NoOfPaths), np.float64)
        self.paths[0] = UnderlyingPrice
#         np.random.seed(123)
        for t in range(1, TradingDaysInAYear + 1):
            self.rand = np.random.standard_normal(NoOfPaths)
            self.paths[t] = self.paths[t - 1] * np.exp((RiskFreeRate - 0.5 * Volatility ** 2) * self.dt +
                                             Volatility * np.sqrt(self.dt) * self.rand)
        print "Shape of simulated Paths:", NoOfPaths
        return self.paths
    
            
class Pricing(object):
    def __init__(self, Simulation):
        self.Simulation = Simulation
        
    def BlackScholes(self):
    
        self.d1 = (np.log(UnderlyingPrice/Strike) + (RiskFreeRate + 0.5 * Volatility ** 2) * Ttm)/(Volatility * np.sqrt(Ttm))
        self.d2 = (np.log(UnderlyingPrice/Strike) + (RiskFreeRate - 0.5 * Volatility ** 2) * Ttm)/(Volatility * np.sqrt(Ttm))

        if OptionType == 'call':
            self.value = (UnderlyingPrice * norm.cdf(self.d1, 0.0, 1.0) - \
                          Strike * np.exp(-RiskFreeRate*Ttm) * norm.cdf(self.d2, 0.0, 1.0))
        return self.value

    def Payoff(self):
        
        self.DiscountFactor = np.exp(-RiskFreeRate*Ttm)
        
        if OptionType == 'call':
            self.PayoffAtMaturity = np.maximum(0, Simulation(TradeDetails, MonteCarloConfig).GBM()[-1] - Strike)
                        
            self.AveragePayoffAtMaturity = np.average(np.maximum(0, Simulation(TradeDetails, MonteCarloConfig).GBM()[-1] \
                                                                 - Strike))
            self.DiscountedPayoff = self.DiscountFactor * self.AveragePayoffAtMaturity
            
        return {"bs":self.DiscountedPayoff, "PayOffMat":self.PayoffAtMaturity}
            
        
		
# Data
# Trade Details
S0 = 200. # Underlying price
K = 100 # Strike
r = 0.05 # RF Rate
v = 0.5 # Volatility
T = 1 # Time to Maturity
N = 252 # No of trading days in a year
O = "call" # Option type

# MC Config
P = 10000 # Number of Paths


def Main(NoOfPaths2):
    #prepare the data
    Trade = TradeDetails(S0, K, r, v, T, N, O)
    print Trade.UnderlyingPrice
    MCParam = MonteCarloConfig(P)
    print MCParam.NoOfPaths 
    Paths = Simulation(Trade, MCParam)
    Prices = Pricing(Paths)
    
    Sim_path = Paths.GBM()
    BSPrice = Prices.BlackScholes()
    MCPrice = Prices.Payoff()["bs"]
    MCStdErr = np.std(Prices.Payoff()["PayOffMat"])/np.sqrt(P)
    
#     print "BS Price:", BSPrice
#     print "MC Price:", MCPrice
#     print "MC Std Err:", MCStdErr
#     print Sim_path
#     return Prices.Payoff()["PayOffMat"]

    return pd.DataFrame({"BS":[BSPrice], "MC":[MCPrice], "SE":[MCStdErr], "nSims":[P]})
    
x = Main(100)
x

lNumberOfPaths = [100, 500, 1000, 2000, 3000, 4000, 5000, 8000, 10000, 15000, 20000]
ldf = []
for NPaths in lNumberOfPaths:
    df = Main(NPaths)
    ldf.append(df)
    
# dfFinal = ldf[0]
# dfFinal = dfFinal.join(ldf[1:],ignore_index = True)
dfFinal = pd.concat(ldf)
dfFinal