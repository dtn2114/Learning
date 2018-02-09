import numpy as np
import pandas as pd
import datetime as dt
import os
from util import get_data, plot_data
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def SMA(price,n):
    sma = price.rolling(window = n, min_periods=n).mean()
    return sma

def SMAP(price, n, chart):
    price = normalize(price)
    sma = price.rolling(window = n, min_periods=n).mean()
    smap = price/sma
    if chart:
        plot_price = price.drop('SPY', axis=1)
        plot_smap = smap.drop('SPY', axis=1)
        plot_sma = sma.drop('SPY', axis=1)
        df_temp = pd.concat([plot_price, plot_sma, plot_smap], axis=1)
        df_temp.columns = ['Price', 'SMA', 'SMAP']
        #plot_smap.columns = ['SMAP']
        f = df_temp.plot(title = "SMAP", linewidth=0.7)
        f.set_xlabel("Date")
        f.set_ylabel("Price")
        plt.grid(True)
        #plot_price.plot(ax=f, linewidth=0.7, legend=True, label="SMAP", color='b', \
                #        style='--')
        f.legend(loc='best', prop={'size':10})
        plt.savefig("SMAP")
    return smap

def bband(price,n, chart=False ):
    price = normalize(price)
    stdev = price.rolling(window=n, min_periods=n).std()
    sma = price.rolling(window=n, min_periods=n).mean()
    top_band = sma + (2 * stdev)
    bottom_band = sma - (2 * stdev)
    bbp = (price - bottom_band) / (top_band - bottom_band)
    if chart:
        plot_price = price.drop('SPY',axis=1)
        plot_bbp = standardize(bbp.drop('SPY',axis=1))
        plot_top = top_band.drop('SPY',axis=1)
        plot_bottom = bottom_band.drop('SPY',axis=1)
        plot_sma = sma.drop('SPY',axis=1)
        fig = plt.figure()
        ax1 = fig.add_subplot(2,1,1)
        df_temp = pd.concat([plot_price, plot_top, plot_bottom, plot_sma],\
                 axis=1)
        df_temp.columns = ['JPM', 'Top Band', 'Bottom Band','SMA']
        df_temp.plot(title = "Bollinger Bands",  ax=ax1, linewidth=0.7)
        plt.grid(True)
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Price")
        ax1.legend(loc='best', prop={'size':5})

        ax2 = fig.add_subplot(2,1,2)
        #plt.legend(loc=0)
        plot_bbp.columns = ['BBP']
        plot_bbp.plot(title = "Bollinger Bands Percentage", ax=ax2, linewidth=0.7)
        plt.grid(True)
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Indicator")
        ax2.legend(loc='best', prop={'size':5})
        fig.tight_layout()
        fig.savefig("Bollinger_Band")
    return bbp, top_band, bottom_band

def momentum(price, n, chart=False):
    mom =  price.copy()
    mom = mom.divide(mom.shift(n)) -1
    if chart:
        stdev = price.rolling(window=n, min_periods=n).std()
        mom_stan = standardize(mom.drop('SPY', axis=1))
        price_norm = normalize(price.drop('SPY', axis=1))
        stdev_stan = standardize(stdev.drop('SPY', axis=1))

        df_temp = pd.concat([mom_stan, stdev_stan], axis=1)
        df_temp.columns = ['Momentum', 'Volatility']
        f = df_temp.plot(title = "Momentum vs. Vol", linewidth=0.7)
        f.set_ylabel("Momentum, Volatility")
        f.set_xlabel("Date")
        #f2 = f.twinx()
        #price_norm.plot(ax=f2, color='r', linewidth=0.7)
        #f2.set_ylabel("Price")
        f.legend(loc='upper left', prop={'size':5})
        plt.grid(True)
        #f2.legend(loc='upper right', prop={'size':5})
        plt.savefig("Momentum_Volatility")
    return mom

def vol(price, n):
    return price.rolling(window=n, min_periods=n).std()

def RSI(price, n, chart=False):
    daily_rets = price.copy()
    daily_rets.values[1:,:] = price.values[1:,:] - price.values[:-1,:]
    daily_rets.values[0,:] = np.nan

    up_rets = daily_rets[daily_rets >=0].fillna(0).cumsum()
    down_rets = -1 * daily_rets[daily_rets < 0].fillna(0).cumsum()

    up_gain = price.copy()
    up_gain.ix[:,:] = 0
    up_gain.values[n:,:] = up_rets.values[n:,:] - up_rets.values[:-n,:]

    down_loss = price.copy()
    down_loss.ix[:,:] = 0
    down_loss.values[n:,:] = down_loss.values[n:,:] - down_rets.values[:-n,:]

    rs = (up_gain / n) / (down_loss / n)
    rsi = 100 - (100 / (1 + rs))
    rsi.ix[:n,:] = np.nan
    rsi[rsi == np.inf] = 100
    if chart:
        rsi = standardize(rsi)
        f = rsi.plot(title = "RSI", color='r', linewidth=0.7)
        f.set_xlabel("Date")
        f.set_ylabel("RSI")
        f.legend(loc='best', prop={'size':5})
        plt.grid(True)
        plt.savefig("RSI")
    return rsi

def order_generator(price, n):
    chart = True
    sma = SMA(price, n)
    smap = SMAP(price,n,chart)
    bbp = bband(price,n, chart)[0]
    mom = momentum(price, n, chart)
    rsi = RSI(price, n, chart)
    orders = price.copy()
    orders.ix[:,:] = np.nan
    spy_rsi = rsi.copy()
    spy_rsi.values[:,:] = spy_rsi.ix[:,['SPY']]

    smap_cross = pd.DataFrame(0, index=smap.index, columns=smap.columns)
    smap_cross[smap>=1] = 1
    smap_cross[1:] = smap_cross.diff()
    smap_cross.ix[0] = 0

    orders[(smap < 0.95) & (bbp<0) & (rsi < 30) & (spy_rsi > 30) ] = 1000
    orders[(smap > 1.05) & (bbp>1) & (rsi > 70) & (spy_rsi < 70) ] = -1000
    orders[(smap_cross != 0)] = 0
    orders.ffill(inplace=True)
    orders.fillna(0, inplace = True)
    orders[1:] = orders.diff()
    orders.ix[0] = 0
    return orders

def normalize(df):
    return (df[0:]/df.ix[0])

def standardize(df):
    return (df-df.mean())/df.std()

def testPolicy(syms, sd =dt.datetime(2010,1,1), ed=dt.datetime(2011,12,31), sv=100000):
    #get data
    n = 14
    dates = pd.date_range(sd, ed)
    prices_all = get_data(syms, dates)
    prices_all.fillna(method='ffill', inplace=True)
    prices_all.fillna(method='bfill', inplace=True)
    #prices = prices_all.divide(prices_all.ix[0])
    prices = prices_all
    orders = order_generator(prices, n)
    del orders['SPY']
    #syms.remove('SPY')
    orders = orders.loc[(orders!=0).any(axis=1)]
    orders_list = []
    #print orders
    for day in orders.index:
        for sym in syms:
            if orders.ix[day,sym] > 0:
                orders_list.append([day.date(), sym, 'BUY',1000])
            elif orders.ix[day,sym] < 0:
                orders_list.append([day.date(), sym, 'SELL',1000])
    for order in orders_list:
        print "      ".join(str(x) for x in order)
    return orders

if __name__ == '__main__':
    syms = ['JPM']
    sd = dt.datetime(2008,01,01)
    ed = dt.datetime(2009,12,31)
    orders = testPolicy(syms,sd,ed)
