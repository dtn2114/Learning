'''
Make orders based on the following rules:
SELL when:
    Prices / SMA > 1.05
    Bollinger Band % > 1
    mom > 0
    vol > 0.3
BUY when
    Prices / SMA < 0.95
    Bollinger Band % < 0
    mom < 0
    vol < 0.1
CLOSE
    symbol crosses through its sma

'''
import marketsimcode as ms
import numpy as np
import pandas as pd
import indicator as ind
import datetime as dt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from util import get_data, plot_data

def testPolicy(symbol = "AAPL", sd=dt.datetime(2010,1,1), \
        ed = dt.datetime(2011,12,31), sv=100000):
    chart = False
    n=14
    prices = get_data(symbol, pd.date_range(sd, ed))
    #print prices
    #ffill and drop SPY
    prices.fillna(method='ffill', inplace=True)
    prices.fillna(method='bfill', inplace=True)
    prices = prices[symbol]
    #Generate indicators
    sma = ind.SMA(prices,n)
    smap = ind.standardize(prices / sma)
    bbp = ind.standardize(ind.bband(prices,n,chart)[0])
    stdev = ind.standardize(prices.rolling(window=n, min_periods=n).std())
    mom = ind.standardize(ind.momentum(prices,n,chart))
    momvol = ind.standardize(mom/stdev)
    orders = prices.copy()
    orders.ix[:,:] = np.nan

    smap_cross = pd.DataFrame(0, index=smap.index, columns=smap.columns)
    smap_cross[smap>=1] = 1
    smap_cross[1:] = smap_cross.diff()
    smap_cross.ix[0] = 0

    orders[(smap<0.95) & (bbp<0) & (stdev < 0.1) & (mom<0) ] = 1000
    orders[(smap>1.05) & (bbp>1) & (stdev > 0.3) & (mom>0) ] = -1000

    orders[(smap_cross != 0)] = 0
    orders.ffill(inplace=True)
    orders.fillna(0, inplace=True)
    orders[1:] = orders.diff()
    #orders.ix[0] = 1000
    #orders.ix[-1] = -1000
    #orders = orders.loc[(orders!=0).any(axis=1)]
    #orders.ix[0] = 0
    #orders.ix[-1] = orders.ix[-2]*-1
    #print orders
    return orders


def testCode(sym, sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31),\
        sv = 100000, chart =False, name='InSample_ManualStrategy'):
    prices = get_data(sym, pd.date_range(sd,ed))
    prices.fillna(method='ffill', inplace=True)
    prices.fillna(method='bfill', inplace=True)
    prices = prices[sym]
    orders = testPolicy(symbol=sym, sd=sd, ed=ed, sv=sv)

    #pad the orders data frame and remove days w/o orders
    #orders.ix[0] = 1000
    #orders.ix[-1] = -1000
    #orders = orders.loc[(orders!=0).any(axis=1)]
    #orders.ix[0] = 0
    #orders.ix[-1] = orders.ix[-2]*-1

    #print orders
    portvals = ms.compute_portvals(df_trades = orders, start_val = sv)
    if isinstance(portvals, pd.DataFrame):
        portvals = portvals[portvals.columns[0]] # just get the first column
    else:
        "warning, code did not return a DataFrame"
    # Benchmark
    orders_bench = orders.copy()
    orders_bench.ix[1:-1,:] = 0
    orders_bench.ix[0] = 1000
    #orders_bench.ix[-1] = 0
    orders_bench = orders_bench[(orders_bench!=0).any(axis=1)]
    portvals_bench = ms.compute_portvals(df_trades = orders_bench,\
            start_val = sv)
    if isinstance(portvals_bench, pd.DataFrame):
        portvals_bench = portvals_bench[portvals_bench.columns[0]] # just get the first column
    else:
        "warning, code did not return a DataFrame"
    if chart:
        prices = ind.normalize(prices)
        portvals = ind.normalize(portvals)
        portvals_bench  = ind.normalize(portvals_bench)
        title = name + " vs. Benchmark"
        f = portvals.plot(title = title, linewidth=1.0,\
                    label='Manual_Strategy', color='black')
        f.set_xlabel("Date")
        f.set_ylabel("Value")
        plt.grid(True)
        portvals_bench.plot(ax=f, linewidth=1.0, legend=True,\
                label='Benchmark', color='blue')
        f.legend(loc='upper left', prop={'size':5})
        #f2 = f.twinx()
        #prices.plot(ax=f2, linewidth=0.7, label = 'Prices', color='r')
        #f2.set_ylabel('Prices')
        #f2.legend(loc='upper right', prop={'size':5})
        for index, row in orders.iterrows():
            if row.values[0] == 1000:
                plt.axvline(x=index, color='g', linestyle='--', lw=0.7, linewidth=0.3)
            if row.values[0] == -1000:
                plt.axvline(x=index, color='r', linestyle='--', lw=0.7, linewidth=0.3)
        plt.savefig(name)
    # Get portfolio stats
    start_date = portvals.index.min()
    end_date = portvals.index.max()
    cum_ret_bench, avg_daily_ret_bench, std_daily_ret_bench, sharpe_ratio_bench \
                    = ms.compute_portfolio_stats(portvals_bench)
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio \
            = ms.compute_portfolio_stats(portvals)

    # Compare portfolio against $SPX
    print "Date Range: {} to {}".format(start_date, end_date)
    print
    print "Sharpe Ratio of Manual: {}".format(sharpe_ratio)
    print "Sharpe Ratio of Bench : {}".format(sharpe_ratio_bench)
    print
    print "Cumulative Return of Manual: {}".format(cum_ret)
    print "Cumulative Return of Bench : {}".format(cum_ret_bench)
    print
    print "Standard Deviation of Manual: {}".format(std_daily_ret)
    print "Standard Deviation of Bench: {}".format(std_daily_ret_bench)
    print
    print "Average Daily Return of Manual: {}".format(avg_daily_ret)
    print "Average Daily Return of Bench: {}".format(avg_daily_ret_bench)
    print
    print "Final Portfolio Value of Manual: {}".format(portvals[-1])
    print "Final Portfolio Value of Bench: {}".format(portvals_bench[-1])



if __name__ == '__main__':
    sym = ['JPM']
    chart = True
    InSample = False
    if InSample:
        sd = dt.datetime(2008,1,1)
        ed = dt.datetime(2009,12,31)
        name = 'InSample_ManualStrategy'
    else:
        sd = dt.datetime(2010,1,1)
        ed = dt.datetime(2011,12,31)
        name = 'OutofSample_ManualStrategy'
    testCode(sym=sym, sd = sd, ed = ed, chart = chart, name=name)
