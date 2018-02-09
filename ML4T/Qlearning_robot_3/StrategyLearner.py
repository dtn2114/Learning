"""
Template for implementing StrategyLearner  (c) 2016 Tucker Balch
"""
#Dung Nguyen
#dnguyen333
import numpy as np
import datetime as dt
import pandas as pd
import util as ut
import random
import QLearner as ql
import indicators as ind


class Market(object):
    class Action:
        LONG = 0
        SHORT = 1
        HOLD = 2

    def __init__(self, sym='SPY', sd=dt.datetime(2008,1, 1), \
            ed=dt.datetime(2009,12,31),sv=100000,verbose=False, experiment=False, impact=0.0):

        self.sym = sym
        self.verbose = verbose
        self.shares = 0
        self.position = 2
        self.sv =  sv
        self.cash = sv
        self.pp = 0
        self.impact = impact
        self.experiment = experiment
        #n = 3  #period
        chart = False
        #take in, normalize data
        df = ut.get_data([sym], pd.date_range(sd-dt.timedelta(100),ed))[sym]
        #print df
        #print sd, ed
        normed =  df/df.ix[0]
        #print normed
		#daily returns & combined features
        rets =  pd.DataFrame(df)
        daily_rets = rets[sym].copy()
        daily_rets[1:] = (rets[sym].ix[1:] / rets[sym].ix[:-1].values) - 1
        daily_rets.ix[0] = 0

		#Simple moving average
        sma5 = normed.rolling(5).mean()
        sma15 = normed.rolling(15).mean()
        sma20 = normed.rolling(20).mean()
        sma25 = normed.rolling(25).mean()
        sma30 = normed.rolling(30).mean()
        sma40 = normed.rolling(40).mean()
        #print sma5
        # Volatility
        vol5 = normed.rolling(5).std()
        vol10 = normed.rolling(10).std()
        vol20 = normed.rolling(20).std()
        vol30 = normed.rolling(30).std()

        # Bollinger bands
        sma_bb = normed.rolling(5).mean()
        sma_bb_std = normed.rolling(5).std()
        bb = (normed - (sma_bb - 2 * sma_bb_std)) / ((sma_bb + 2 * sma_bb_std) - (sma_bb - 2 * sma_bb_std))

        # Moving average convergence/divergence
        #ema12 = pd.ewma(np.asarray(normed), span=12)
        #ema26 = pd.ewma(np.asarray(normed), span=26)
        #macd = ema12 - ema26

        # Momentum
        momentum2 = normed / normed.shift(2) - 1
        momentum5 = normed / normed.shift(5) - 1
        momentum10 = normed / normed.shift(10) - 1

        # Combine into new dataframe
        df = pd.DataFrame(df).assign(sma5=normed / sma5 - 1).assign(\
            bb=bb).assign(momentum2=momentum2).assign(normed=normed).\
            assign(vol10=vol10).assign(vol20=vol20).assign(vol30=vol30).assign(vol10=vol10)\
            .assign(vol5=vol5).assign(\
            momentum5=momentum5).assign(momentum10=momentum10)[sd:]

        df = df.assign(dr=daily_rets)
        #print df
        # Determine optimal features for states
        corr_df = df.corr().abs()
        corr = corr_df['dr'][:]
        corr.ix[0] = 0
        corr['normed'] = 0
        icorr = np.asarray(corr)
        scorr = icorr.argsort()[-4:][::-1]  # select top 3 features and daily returns
        scorr = scorr[1:]  # remove daily returns from possible features
        optimal_ftrs = []
        for i in scorr:
            optimal_ftrs.append(corr_df.columns[i])

        if experiment:
            #print "Experiment 1 ACTIVATED"
            n = 14
            sma = ind.SMA(normed, n)
            smap = ind.standardize(normed / sma)
            bbp = ind.standardize(ind.bband(normed, n, chart)[0])
            stdev = ind.standardize(normed.rolling(window=n, min_periods=n).std())
            mom = ind.standardize(ind.momentum(normed, n, chart))
            momvol = ind.standardize(mom/stdev)
            df = pd.DataFrame(df).assign(normed = normed).assign(smap=smap).assign(stdev=stdev).assign(bbp=bbp).assign(mom=mom)[sd:]
            optimal_ftrs = ['smap', 'stdev', 'bbp', 'mom']

        #df = pd.DataFrame(df).assign(normed = normed).assign(smap=smap).assign(bbp=bbp).assign(mom=mom)[sd:]
        #df = pd.DataFrame(df).assign(normed = normed)[sd:]

        self.df = df
        self.market = df.iterrows()
        self.curr = self.market.next()
        self.action = self.Action()
        self.features  = optimal_ftrs
        #print self.features
        #print df


    '''
    def long(self):
        if self.shares > 0:
            return -10000 #
        close = self.curr[1][self.sym]
        #self.pp = pd.cut(self.df['normed'], 10, labels=False)[self.curr[0]]
        self.shares = self.shares + 1000
        self.cash = self.cash - (1000 * close)
        dr = self.shares * self.curr[1]['dr']
        return dr

    '''
    def long(self):
        self.shares = 1000
        close = self.curr[1][self.sym]
        #print close
        if self.position == 0: return -1000
        elif self.position == 1:
            self.position = 0
            return 20*(self.curr[1]['dr'] + (1+self.impact))
        elif self.position == 2:
            self.position = 0
            return 10*(self.curr[1]['dr'] * (1+self.impact))
        else:
            self.position = 2
            return -10
    '''

    def short(self):
        if self.shares < 0:
            return -10000
        if self.shares == 1000:
            self.shares += -2000
            self.cash += (2000 * self.curr[1][self.sym])
            dr = 1* (2000 * self.curr[1][self.sym])
        else:
            self.shares += -1000
            self.cash += (1000 * self.curr[1][self.sym])
            dr = 1* (1000 * self.curr[1][self.sym])
        #print dr
        return dr

    '''

    def short(self):
        self.shares = -2000
        close = self.curr[1][self.sym]
        if self.position == 2:
            self.position =1
            return  10*(self.curr[1]['dr'] * (-1*self.impact + 1))
        elif self.position == 0:
            self.position = 1
            return  20*(self.curr[1]['dr'] * (-1*self.impact + 1))
        elif self.position == 1: return -1000
        else:
            self.position = 2
            return -10

    '''

    def hold(self):
        dr = self.shares * self.curr[1]['dr']
        return 0

    '''

    def hold(self):
        if self.position == 1: return 10 * self.curr[1]['dr']
        elif self.position == 0: return 0
        elif self.position == 2: return -10 *self.curr[1]['dr']
        else:
            self.position = 0
            return -100


    def discretize(self):
        date = self.curr[0]
        #print date
        s = self.position
        for i, feature in enumerate(self.features):
            s += (10 ** (i+1) * pd.cut(self.df[feature], 10, labels=False)[date])
        return int(s)

    def reward(self, action):
        #print self.action.LONG, self.action.SHORT, self.action.HOLD
        #print self.long(), self.short(), self.hold()
        #print action

        r = {   self.action.LONG : self.long,\
                self.action.SHORT: self.short,\
                self.action.HOLD : self.hold, \
            }[action]()
        try:
            self.curr = self.market.next()
            state = self.discretize()
        except StopIteration:
            return None, None
        return state, r

    def state(self):
       cv = self.shares * self.curr[1][self.sym] + self.cash
       return cv, self.cash, self.shares, self.curr[1][self.sym]


    def author(self):
        return 'dnguyen333'

    def raw(self):
        return self.df

class StrategyLearner(object):

    # constructor
    def __init__(self, verbose = False, impact=0.0):
        self.verbose = verbose
        self.impact = impact
        self.experiment = False
        self.ql = ql.QLearner(num_states=int(100000), num_actions=3, alpha=0.1, \
                gamma=0.9, rar=0.5, radr=0.99, dyna=0, verbose=False)


    def author(self):
        return 'dnguyen333'

    # this method should create a QLearner, and train it for trading
    def addEvidence(self, symbol = "IBM", \
        sd=dt.datetime(2008,1,1), \
        ed=dt.datetime(2009,1,1), \
        sv = 100000):
        #print self.experiment
        ret = -1
        i = 0
        while i < 10:
            i+=1
            market = Market(symbol, sd, ed, sv, self.verbose,self.experiment, self.impact)
            s = market.discretize()
            a = self.ql.querysetstate(s)
            while True:
                s1, r = market.reward(a)
                if s1 is None:
                    break
                a = self.ql.query(s1,r)

            ret0 = ret
            ret = market.state()[0]
            #ret0 = ret
            #ret = market.state()[0]
            if (ret == ret0) & (i > 200):
                break
            if i > 1000:
                print 'Error: cannot converge'
                break
        #print market.raw()
        # this method should use the existing policy and test it against new data
    def testPolicy(self, symbol = "IBM", \
        sd=dt.datetime(2009,1,1), \
        ed=dt.datetime(2010,1,1), \
        sv = 100000):
        self.ql.rar = 0
        market = Market(symbol, sd, ed, sv, self.verbose, self.experiment, self.impact)
        s = market.discretize()
        a = self.ql.querysetstate(s)
        df = market.raw()
        #print df
        actions = []

        while True:
            s1, r = market.reward(a)
            if s1 is None:
                break
            a = self.ql.querysetstate(s1)
            actions.append(a)
            if self.verbose: print s1, r, a

        prev = 0
        actions = np.asarray([0] + actions)
        for i in range(1, len(actions) - 1):
            if actions[i] != 2:
                if actions[i] == prev: actions[i] = 2
                else: prev = actions[i]


        def order(x):
            if x==0: return 2000
            elif x==1: return -2000
            else: return 0


        #actions[-1] = 1 #close out last position
        df['Trades'] = actions
        #df = df[df['Trades'] != 2].copy()
        df[symbol] = df['Trades'].apply(lambda x:order(x))
        df[symbol].ix[0] = 1000
        df[symbol].ix[-1]= 1000
        df = df[df[symbol] != 0].copy()
        df[symbol].ix[0] = 0
        df[symbol].ix[-1]= 0
        df[symbol].ix[1] = df[symbol].ix[1]/2
        df = pd.DataFrame(df[symbol].copy())
        return df

    def experment(self):
        self.experiment = True

if __name__=="__main__":
    print "One does not simply think up a strategy"
