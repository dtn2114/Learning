import numpy as np
import pandas as pd
import marketsimcode as ms
import datetime as dt
from util import get_data, plot_data
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import indicator as ind
def testPolicy(sym, sd=dt.datetime(2010,01,01), ed=dt.datetime(2011,12,31), \
        sv = 100000):
    prices= get_data(sym,pd.date_range(sd,ed))
    prices.fillna(method='ffill',inplace=True)
    prices.fillna(method='bfill',inplace=True)
    prices = prices[sym]
    daily_rets = prices.diff(periods=1).shift(-1)
    holding = np.sign(daily_rets) * 1000
    holding.ix[-1,:] = 0

    orders = prices.copy()
    orders.ix[:,:] = np.NaN
    orders = holding.diff(1)
    orders.ix[0,:] = holding.ix[0,:]
    #orders = orders[(orders != 0).any(axis=1)]
    #print orders
    return orders


def testCode(sym, sd=dt.datetime(2010,1,1), ed=dt.datetime(2011,12,31), sv=100000,\
                chart = False):
    # Best Possible Order
    orders = testPolicy(sym=sym, sd=sd, ed=ed, sv = sv)
    #orders = orders[(orders != 0).any(axis=1)]
    portvals = ms.compute_portvals(df_trades = orders, start_val = sv, commission=0,\
            impact=0)
    if isinstance(portvals, pd.DataFrame):
        portvals = portvals[portvals.columns[0]] # just get the first column
    else:
        "warning, code did not return a DataFrame"
    #print orders
    # Benchmark
    orders_bench = orders.copy()
    orders_bench.ix[1:-1,:] = 0
    orders_bench.ix[0] = 1000
    #print orders_bench
    orders_bench = orders_bench[(orders_bench!=0).any(axis=1)]
    #print orders
    portvals_bench = ms.compute_portvals(df_trades = orders_bench,\
            start_val = sv, commission=0, impact=0)
    if isinstance(portvals_bench, pd.DataFrame):
        portvals_bench = portvals_bench[portvals_bench.columns[0]] # just get the first column
    else:
        "warning, code did not return a DataFrame"
    if chart:
        portvals = ind.normalize(portvals)
        portvals_bench = ind.normalize(portvals_bench)
        f = portvals.plot(title = "Best Possible vs. Benchmark", linewidth=1.5,\
                    label='Best_Possible', color='black', style='-')
        f.set_xlabel("Date")
        f.set_ylabel("Value")
        plt.grid(True)
        portvals_bench.plot(ax=f, linewidth=1.5, legend=True,\
                label='Benchmark', color='blue', style='--')
        f.legend(loc='best', prop={'size':10})
        plt.savefig("Best_Possible")
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
    print "Sharpe Ratio of Best: {}".format(sharpe_ratio)
    print "Sharpe Ratio of Bench : {}".format(sharpe_ratio_bench)
    print
    print "Cumulative Return of Best: {}".format(cum_ret)
    print "Cumulative Return of Bench : {}".format(cum_ret_bench)
    print
    print "Standard Deviation of Best: {}".format(std_daily_ret)
    print "Standard Deviation of Bench: {}".format(std_daily_ret_bench)
    print
    print "Average Daily Return of Best: {}".format(avg_daily_ret)
    print "Average Daily Return of Bench : {}".format(avg_daily_ret_bench)
    print
    print "Final Portfolio Value of Best: {}".format(portvals[-1])
    print "Final Portfolio Value of Bench: {}".format(portvals_bench[-1])


if __name__ == '__main__':
    sym = ['JPM']
    sd = dt.datetime(2008,1,1)
    ed = dt.datetime(2009,12,31)
    chart = True
    testCode(sym = sym, sd = sd, ed = ed, chart=chart)


