"""Analyze a portfolio."""

import pandas as pd
import numpy as np
import datetime as dt
from util import get_data, plot_data

# This is the function that will be tested by the autograder
# The student must update this code to properly implement the functionality
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
##code added
    normed = prices/prices.ix[0]
    alloced = normed * allocs
    pos_vals = alloced *sv
    port_val = pos_vals.sum(axis=1)
    #print(port_val)

    # Get portfolio statistics (note: std_daily_ret = volatility)
    cr, adr, sddr, sr = compute_portfolio_stats(port_val,allocs, rfr, sf)

    # Compare daily portfolio value with SPY using a normalized plot
    if gen_plot:
        # add code to plot here
        df_temp = pd.concat([port_val/port_val.ix[0], prices_SPY/prices_SPY.ix[0]], keys=['Portfolio', 'SPY'], axis=1)
        ax = df_temp.plot(title="port_val vs. SPY", fontsize = 2)
	ax.set_xlabel("Date")
	ax.set_ylabel("Normalized Price")
	pass

    # Add code here to properly compute end value
    ev = sv * (1 + cr)

    return cr, adr, sddr, sr, ev

def compute_portfolio_stats(prices, allocs, rfr, sf):
    daily_rets = prices.copy()
    daily_rets[1:] = (prices[1:]/prices[:-1].values) -1
    daily_rets = daily_rets[1:]
    cr = (prices[-1]/prices[0])-1
    adr = daily_rets.mean()
    sddr = daily_rets.std()
    sr = np.sqrt(sf) * (adr-rfr)/sddr
    return cr, adr, sddr, sr

def test_code():
    # This code WILL NOT be tested by the auto grader
    # It is only here to help you set up and test your code

    # Define input parameters
    # Note that ALL of these values will be set to different values by
    # the autograder!
    start_date = dt.datetime(2010,6,1)
    end_date = dt.datetime(2010,12,31)
    symbols = ['GOOG', 'AAPL', 'GLD', 'XOM']
    allocations = [0.2, 0.3, 0.4, 0.1]
    start_val = 1000000  
    risk_free_rate = 0.0
    sample_freq = 252.0

    # Assess the portfolio
    cr, adr, sddr, sr, ev = assess_portfolio(sd = start_date, ed = end_date,\
        syms = symbols, \
        allocs = allocations,\
        sv = start_val, \
        gen_plot = False)

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
    test_code()
