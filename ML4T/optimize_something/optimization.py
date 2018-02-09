"""MC1-P2: Optimize a portfolio."""

import pandas as pd
import numpy as np
import datetime as dt
from util import get_data, plot_data
import scipy.optimize as spo
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
pd.set_option('mode.chained_assignment','raise')
# This is the function that will be tested by the autograder
# The student must update this code to properly implement the functionality
def optimize_portfolio(sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,1,1), \
    syms=['GOOG','AAPL','GLD','XOM'], gen_plot=False):

    # Read in adjusted closing prices for given symbols, date range
    dates = pd.date_range(sd, ed)
    prices_all = get_data(syms, dates)  # automatically adds SPY
    prices_all.fillna(method='ffill', inplace=True)
    prices_all.fillna(method='bfill', inplace=True)
    prices = prices_all[syms]  # only portfolio symbols
    prices_SPY = prices_all['SPY']  # only SPY, for comparison later

##############################################################################

    # find the allocations for the optimal portfolio
    # note that the values here ARE NOT meant to be correct for a test case
    alloc_guess = [1.0/len(syms)] * len(syms)
    bnds = tuple((0.0,1.0) for x in range(len(syms)))
    fun = lambda allocs: f(allocs,prices)
    cons = ({'type':'eq', 'fun': lambda inputs: 1.0-np.sum(inputs) })
    result = spo.minimize(fun, alloc_guess, method = 'SLSQP',
        bounds = bnds, constraints = cons,  options={'disp':True})

    #print('result',result)
    allocs = result.x

    # add code here to compute stats
    cr, adr, sddr, sr = compute_portfolio_stats(get_portfolio_value(prices, allocs)
      , allocs)

    # Get daily portfolio value
    port_val = get_portfolio_value(prices, allocs) # add code here to compute daily portfolio values

    # Compare daily portfolio value with SPY using a normalized plot
    if gen_plot:
        # add code to plot here
        df_temp = pd.concat([port_val/port_val.ix[0], prices_SPY/prices_SPY.ix[0]], \
            keys=['Portfolio', 'SPY'], axis=1)
        fig = df_temp.plot(title = "Portfolio vs. SPY", fontsize = 12)
        fig.set_xlabel("Date")
        fig.set_ylabel("Normalized Price")
        plt.savefig('plot.png')

        pass


##############################################################################
    return allocs, cr, adr, sddr, sr

def f(allocs, prices):
    cr, adr, sddr, sr = compute_portfolio_stats(get_portfolio_value(prices, allocs),
        allocs)
    return sddr

def assess_portfolio(sd = dt.datetime(2008,1,1), ed = dt.datetime(2009,1,1), \
    syms = ['GOOG','AAPL','GLD','XOM'], \
    allocs=[0.1,0.2,0.3,0.4], \
    sv=1000000, rfr=0.0, sf=252.0, \
    gen_plot=False):

    # Read in adjusted closing prices for given symbols, date range
    dates = pd.date_range(sd, ed)
    prices_all = get_data(syms, dates)  # automatically adds SPY
    prices = prices_all[syms]  # only portfolio symbols
    prices.fillna(method='ffill', inplace=True)
    prices.fillna(method='bfill', inplace=True)
    prices_SPY = prices_all['SPY']  # only SPY, for comparison later
    prices_SPY.fillna(method='ffill', inplace=True)
    prices_SPY.fillna(method='bfill', inplace=True)

    # Get daily portfolio value
    port_val = get_portfolio_value(prices, allocs, sv)
    #print(port_val)

    # Get portfolio statistics (note: std_daily_ret = volatility)
    cr, adr, sddr, sr = compute_portfolio_stats(port_val,allocs, rfr, sf)

    # Compare daily portfolio value with SPY using a normalized plot
    if gen_plot:
        # add code to plot here
        df_temp = pd.concat([port_val/port_val.ix[0], prices_SPY/prices_SPY.ix[0]], keys
            =['Portfolio', 'SPY'], axis = 1)
        ax = df_temp.plot(title="port_val vs. SPY", fontsize = 2)
        ax.set_xlabel("Date")
        ax.set_ylabel("Normalized Price")
        pass

    # Add code here to properly compute end value
    ev = sv * (1 + cr)
    return cr, adr, sddr, sr, ev

def get_portfolio_value(prices, allocs, sv=1000000):
    normed = prices/prices.ix[0]
    alloced = normed * allocs
    pos_vals = alloced * sv
    port_val = pos_vals.sum(axis=1)
    return port_val

def compute_portfolio_stats(port_val, allocs, rfr=0.0, sf=252):
    # This function will not be called by the auto grader
    daily_rets = port_val.copy()
    daily_rets[1:] = (port_val[1:]/port_val[:-1].values) -1
    daily_rets = daily_rets[1:]
    cr = (port_val[-1]/port_val[0])-1
    adr = daily_rets.mean()
    sddr = daily_rets.std()
    sr = np.sqrt(sf) * (adr-rfr)/sddr
    return cr, adr, sddr, sr

def test_code():
    # This function WILL NOT be called by the auto grader
    # Do not assume that any variables defined here are available to your function/code
    # It is only here to help you set up and test your code

    # Define input parameters
    # Note that ALL of these values will be set to different values by
    # the autograder!

    start_date = dt.datetime(2008,6,1)
    end_date = dt.datetime(2009,6,1)
    symbols = ['IBM', 'X', 'GLD']

    # Assess the portfolio
    allocations, cr, adr, sddr, sr = optimize_portfolio(sd = start_date, ed = end_date,\
        syms = symbols, \
        gen_plot = True)

    # Print statistics
    print "Start Date:", start_date
    print "End Date:", end_date
    print "Symbols:", symbols
    print "Allocations:", allocations
    print "Sharpe Ratio:", sr
    print "Volatility (stdev of daily returns):", sddr
    print "Average Daily Return:", adr
    print "Cumulative Return:", cr

if __name__ == "__main__":
    # This code WILL NOT be called by the auto grader
    # Do not assume that it will be called
    test_code()
