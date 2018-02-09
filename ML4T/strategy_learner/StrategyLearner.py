"""
Template for implementing StrategyLearner  (c) 2016 Tucker Balch
"""
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
            ed=dt.datetime(2009,12,31),sv=100000,verbose=False):

        self.sym = sym
        self.verbose = verbose
        self.shares = 0
        self.position = 0
        self.sv =  sv
        self.cash = sv
        self.pp = 0
        #period
        n = 28
        chart = False
        #take in, normalize data
        df = ut.get_data([sym], pd.date_range(sd-dt.timedelta(100),ed))[sym]
        normed =  df/df.ix[0]

        #get features
        sma = ind.SMA(normed, n)
        smap = ind.standardize(normed / sma)
        bbp = ind.standardize(ind.bband(normed, n, chart)[0])
        stdev = ind.standardize(normed.rolling(window=n, min_periods=n).std())
        mom = ind.standardize(ind.momentum(normed, n, chart))
        momvol = ind.standardize(mom/stdev)

        #daily returns & combined features
        rets =  pd.DataFrame(df)
        daily_rets = rets[sym].copy()
        daily_rets[1:] = (rets[sym].ix[1:] / rets[sym].ix[:-1].values) - 1
        daily_rets.ix[0] = 0
        #print df

        #df = pd.DataFrame(df).assign(normed = normed).assign(smap=smap).\
                #        assign(stdev=stdev).assign(bbp=bbp).assign(mom=mom)[sd:]
        df = pd.DataFrame(df).assign(normed = normed).assign(smap=smap).\
                assign(bbp=bbp).assign(mom=mom)[sd:]

        #print df
        daily_rets.ix[0] = 0
        df = df.assign(dr = daily_rets)

        #print df
        self.df = df
        self.market = df.iterrows()
        self.curr = self.market.next()
        self.action = self.Action()
        #self.features = ['smap', 'stdev', 'bbp', 'mom']
        self.features = ['smap', 'bbp', 'mom']


    def long(self):
        if self.shares >= 2000:
            return -10000 #
        close = self.curr[1][self.sym]
        self.pp = pd.cut(self.df['normed'], 10, labels=False)[self.curr[0]]
        self.shares = self.shares + 1000
        self.cash = self.cash - (1000 * close)
        dr = self.shares * self.curr[1]['dr']
        return dr

    '''
    def long(self):
        self.shares = 2000
        close = self.curr[1][self.sym]
        #print close
        if self.position == 1: return -1000
        elif self.position == 0 or self.position == 2:
            self.position = 1
            return 10 * self.curr[1]['dr']
        else:
            self.position = 1
            return -1000
    '''

    def short(self):
        if self.shares <= 0:
            return -10000
        self.shares = self.shares - 1000
        self.cash = self.cash + (1000 * self.curr[1][self.sym])
        self.pp = 0
        dr = (self.cash - self.sv) + (self.shares * self.curr[1][self.sym])
        return dr

    '''

    def short(self):
        self.shares = -2000
        close = self.curr[1][self.sym]
        if self.position == 1 or self.position == 0:
            self.position = 2
            return -10 * self.curr[1]['dr']
        elif self.position == 2: return -1000
        else:
            self.position = 2
            return -1000

    '''

    def hold(self):
        dr = self.shares * self.curr[1]['dr']
        return dr

    '''

    def hold(self):
        if self.position == 1: return 5 * self.curr[1]['dr']
        elif self.position == 0: return 0
        elif self.position == 2: return -5 * self.curr[1]['dr']
        else:
            self.position = 0
            return -100
    '''

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
        self.ql = ql.QLearner(num_states=int(10000), num_actions=3, alpha=0.1, \
                gamma=0.9, rar=0.5, radr=0.99, dyna=0, verbose=False)


    def author(self):
        return 'dnguyen333'

    # this method should create a QLearner, and train it for trading
    def addEvidence(self, symbol = "IBM", \
        sd=dt.datetime(2008,1,1), \
        ed=dt.datetime(2009,1,1), \
        sv = 100000):

        ret = -1
        i = 0
        while i != ret:
            #i+=1
            market = Market(symbol, sd, ed, sv, self.verbose)
            s = market.discretize()
            a = self.ql.querysetstate(s)
            while True:
                s1, r = market.reward(a)
                if s1 is None:
                    break
                a = self.ql.query(s1,r)
            ret = i
            i = market.state()[0]
            #ret0 = ret
            #ret = market.state()[0]
            #if (ret == ret0) & (i > 200):
            #    break
            #if i > 1000:
            #    print 'Error: cannot converge'
            #    break
        #print market.raw()
        # this method should use the existing policy and test it against new data
    def testPolicy(self, symbol = "IBM", \
        sd=dt.datetime(2009,1,1), \
        ed=dt.datetime(2010,1,1), \
        sv = 100000):
        self.ql.rar = 0
        market = Market(symbol, sd, ed, sv, self.verbose)
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
        df['Trades'].ix[0] = 0
        #print df
        df = df[df['Trades'] != 2].copy()
        #df['Orders'] = df['Trades'].apply(lambda x:order(x))
        df[symbol] = df['Trades'].apply(lambda x:order(x))
        #df['Share'] = 2000
        df[symbol].ix[0] = df[symbol].ix[0]/2
        #df[symbol].ix[1] = df[symbol].ix[1]/2
        df = pd.DataFrame(df[symbol].copy())
        #df.columns = [symbol]
        #print df
        return df

        # your code should return the same sort of data
        '''
        dates = pd.date_range(sd, ed)
        prices_all = ut.get_data([symbol], dates)  # automatically adds SPY
        trades = prices_all[[symbol,]]  # only portfolio symbols
        trades_SPY = prices_all['SPY']  # only SPY, for comparison later
        trades.values[:,:] = 0 # set them all to nothing
        trades.values[0,:] = 1000 # add a BUY at the start
        trades.values[40,:] = -1000 # add a SELL
        trades.values[41,:] = 1000 # add a BUY
        trades.values[60,:] = -2000 # go short from long
        trades.values[61,:] = 2000 # go long from short
        trades.values[-1,:] = -1000 #exit on the last day
        if self.verbose: print type(trades) # it better be a DataFrame!
        if self.verbose: print trades
        if self.verbose: print prices_all
        return trades
        '''
if __name__=="__main__":
    print "One does not simply think up a strategy"
