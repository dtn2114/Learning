"""MC2-P1: Market simulator."""

import pandas as pd
import numpy as np
import datetime as dt
import os
from util import get_data, plot_data

def author():
    return 'dnguyen333'

def compute_portvals(df_trades, start_val = 1000000, commission=9.95, impact=0.005):
    # this is the function the autograder will call to test your code
    # NOTE: orders_file may be a string, or it may be a file object. Your
    # code should work correctly with either input
    # TODO: Your code here

    #Get orders, Read in dates and syms
    #orders = pd.read_csv(orders_file, index_col='Date', parse_dates=True, \
            #                        usecols=['Date','Symbol','Order','Shares'], na_values=['nan'])
    syms = df_trades.columns.values.tolist()
    start_date = df_trades.index.min()
    end_date = df_trades.index.max()
    df_trades.ix[-1] = -1000
    #print df_trades
    df_trades = df_trades.ix[(df_trades!=0).any(axis=1)]
    df_trades.is_copy=False
    df_trades.ix[-1] = df_trades.ix[-2] * -1
    #print df_trades
    orders = []
    sym = syms[0]
    for day in df_trades.index:
        if df_trades.ix[day,sym] > 0:
            orders.append([day.date(), sym, 'BUY', df_trades.ix[day,sym]])
        elif df_trades.ix[day,sym] < 0:
            orders.append([day.date(), sym, 'SELL', df_trades.ix[day,sym]*-1])
        elif df_trades.ix[day,sym] == 0:
            orders.append([day.date(), sym, 'HOLD', df_trades.ix[day,sym]])
    orders = pd.DataFrame(orders, columns = ['Date', 'Symbol', 'Order', 'Shares']).set_index('Date')
    #print orders

    orders.sort_index(inplace=True)

    #start_date = orders.index.min()
    #end_date = orders.index.max()
    #syms = orders["Symbol"].unique().tolist()
    #print start_date
    #Get PRICES
    prices = get_data(syms, pd.date_range(start_date, end_date))
    prices.fillna(method='ffill', inplace=True)
    prices.fillna(method='bfill', inplace=True)
    prices = prices[syms]
    prices['Cash'] = 1
    #print prices

    # Get TRADES
    trades = prices.copy()
    trades.ix[:,:] = 0
    #print orders, prices
    for index, row in orders.iterrows():
        #print index, row
        sym = row['Symbol']
        order = row['Order']
        share = row['Shares']
        share = share if (order == 'BUY') else share * -1
        im = impact if (order == 'BUY') else impact * -1
        #print index
        price = prices.ix[index, sym]
        price = price * (1+im)
        trades.ix[index,sym] += share
        trades.ix[index,'Cash'] += share * price *(-1) - commission
    #print trades[(trades!=0).any(1)]

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
    #print values[(values!=0).any(1)]
    return values.ix[:,'portval']

def compute_portfolio_stats(prices, rfr=0.0, sf=252.0):
    daily_rets = prices.copy()
    daily_rets[1:] = (prices[1:]/prices[:-1].values) -1
    daily_rets = daily_rets[1:]
    cr = (prices[-1]/prices[0])-1
    adr = daily_rets.mean()
    sddr = daily_rets.std()
    sr = np.sqrt(sf) * (adr-rfr)/sddr
    return cr, adr, sddr, sr

def test_code():
    # this is a helper function you can use to test your code
    # note that during autograding his function will not be called.
    # Define input parameters

    of = "./orders/orders-02.csv"
    sv = 1000000
    of = "./orders/orders2.csv"

    # Process orders
    portvals = compute_portvals(orders_file = of, start_val = sv)
    if isinstance(portvals, pd.DataFrame):
        portvals = portvals[portvals.columns[0]] # just get the first column
    else:
        "warning, code did not return a DataFrame"

    # Get portfolio stats
    # Here we just fake the data. you should use your code from previous assignments.
    start_date = portvals.index.min()
    end_date = portvals.index.max()
    prices_SPY = get_data([], pd.date_range(start_date, end_date))
    prices_SPY.fillna(method="ffill",inplace=True)
    prices_SPY.fillna(method="bfill",inplace=True)
    prices_SPY = prices_SPY[prices_SPY.columns[0]]
    #print prices_SPY
    #print portvals
    cum_ret_SPY, avg_daily_ret_SPY, std_daily_ret_SPY, sharpe_ratio_SPY = compute_portfolio_stats(prices_SPY)
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = compute_portfolio_stats(portvals)


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
    print "Standard Deviation of SPY : {}".format(std_daily_ret_SPY)
    print
    print "Average Daily Return of Fund: {}".format(avg_daily_ret)
    print "Average Daily Return of SPY : {}".format(avg_daily_ret_SPY)
    print
    print "Final Portfolio Value: {}".format(portvals[-1])



if __name__ == "__main__":
    test_code()
