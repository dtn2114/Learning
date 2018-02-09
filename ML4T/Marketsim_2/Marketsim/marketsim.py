"""MC2-P1: Market simulator."""

import pandas as pd
import numpy as np
import datetime as dt
import os
from util import get_data, plot_data

def compute_portvals(orders_file = "./orders/orders.csv", start_val = 1000000, commission=9.95, impact=0.005):
    # this is the function the autograder will call to test your code
    # NOTE: orders_file may be a string, or it may be a file object. Your
    # code should work correctly with either input
    # TODO: Your code here

    #Get orders, Read in dates and syms
    orders = pd.read_csv(orders_file, index_col='Date', parse_dates=True)
    orders.sort_index(inplace=True)
    start_date = orders.index.min()
    end_date = orders.index.max()
    syms = orders["Symbol"].unique().tolist()

    #Get PRICES
    prices = get_data(syms, pd.date_range(start_date, end_date), False)
    prices['Cash'] = 1
    prices.fillna(method='ffill', inplace=True)
    prices.fillna(method='bfill', inplace=True)
    #print prices[(prices!=0).any(1)]

    # Get TRADES
    trades = prices.copy()
    trades.ix[:,:] = 0
    for index, row in orders.iterrows():
        sym = row['Symbol']
        order = row['Order']
        share = row['Shares']
        share = share if (order == 'BUY') else share * -1
        price = prices.ix[index, sym]
        trades.ix[index,sym] += share
        trades.ix[index,'Cash'] += share * price * -1
    # print trades[(trades!=0).any(1)]

    # Get HOLDINGS
    holdings = trades.copy()
    holdings.ix[start_date, 'Cash'] += start_val
    holdings['Cash'] = holdings['Cash'].cumsum(axis=0)
    for sym in syms:
        holdings[sym] = holdings[sym].cumsum(axis=0)
    #print holdings[(holdings!=0).any(1)]

    #Get VALUES
    values = holdings * prices
    values['portval'] = values.sum(axis=1)
    # print values[(values!=0).any(1)]
    return values.ix[:,'portval']

def test_code():
    # this is a helper function you can use to test your code
    # note that during autograding his function will not be called.
    # Define input parameters

    of = "./orders/orders-02.csv"
    sv = 1000000

    # Process orders
    portvals = compute_portvals(orders_file = of, start_val = sv)
    if isinstance(portvals, pd.DataFrame):
        portvals = portvals[portvals.columns[0]] # just get the first column
    else:
        "warning, code did not return a DataFrame"

    # Get portfolio stats
    # Here we just fake the data. you should use your code from previous assignments.
    start_date = dt.datetime(2008,1,1)
    end_date = dt.datetime(2008,6,1)
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = [0.2,0.01,0.02,1.5]
    cum_ret_SPY, avg_daily_ret_SPY, std_daily_ret_SPY, sharpe_ratio_SPY = [0.2,0.01,0.02,1.5]

    # Compare portfolio against $SPX
    print "Date Range: {} to {}".format(start_date, end_date)
    print
    print "Sharpe Ratio of Fund: {}".format(sharpe_ratio)
    print "Sharpe Ratio of SPY : {}".format(sharpe_ratio_SPY)
    print
    print "Cumulative Return of Fund: {}".format(cum_ret)
    print "Cumulative Return of SPY : {}".format(cum_ret_SPY)
    print
    print "Standard Deviation of Fund: {}".format(std_daily_ret)
    print "Standard Deviation of SPY : {}".format(std_daily_ret_SPY)
    print
    print "Average Daily Return of Fund: {}".format(avg_daily_ret)
    print "Average Daily Return of SPY : {}".format(avg_daily_ret_SPY)
    print
    print "Final Portfolio Value: {}".format(portvals[-1])

if __name__ == "__main__":
    test_code()
