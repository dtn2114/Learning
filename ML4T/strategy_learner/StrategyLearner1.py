import numpy as np
import datetime as dt
import QLearner as ql
import pandas as pd
import util as ut
import time
import timeit


# Force Index
def calc_force_index(df, n):
    F = pd.Series(df['Adj Close'].diff(n) * df['Volume'].diff(n), name='Force')
    df = df.join(F)
    return df


# Moving Average
def calc_sma(df, n):
    col_name = 'SMA_' + str(n)
    sma = pd.Series(pd.rolling_mean(df['Adj Close'], n), name=col_name)
    df = df.join(sma)
    df['Adj Close / SMA'] = df[col_name] / df['Adj Close']
    return df


# Exponential Moving Average
def calc_ema(df, n):
    EMA = pd.Series(pd.ewma(df['Adj Close'], span=n, min_periods=n - 1), name='EMA')
    df = df.join(EMA)
    return df


# Bollinger Bands
def calc_bb_bands(df, n):
    MA = pd.Series(pd.rolling_mean(df['Adj Close'], n))
    MSD = pd.Series(pd.rolling_std(df['Adj Close'], n))
    b1 = 4 * MSD / MA
    B1 = pd.Series(b1, name='Bollinger_Lower')
    df = df.join(B1)
    b2 = (df['Adj Close'] - MA + 2 * MSD) / (4 * MSD)
    B2 = pd.Series(b2, name='Bollinger_Upper')
    df = df.join(B2)
    return df


# MACD, MACD Signal and MACD difference
def calc_macd(df, n_fast, n_slow):
    ema_fast = pd.Series(pd.ewma(df['Adj Close'], span=n_fast, min_periods=n_slow - 1))
    ema_slow = pd.Series(pd.ewma(df['Adj Close'], span=n_slow, min_periods=n_slow - 1))
    col_name_macd = 'MACD'
    col_name_sign = 'MACDSign'
    col_name_macd_diff = 'MACDDiff'
    macd = pd.Series(ema_fast - ema_slow, name=col_name_macd)
    macd_sign = pd.Series(pd.ewma(macd, span=9, min_periods=4), name=col_name_sign)
    macd_diff = pd.Series(macd - macd_sign, name=col_name_macd_diff)
    df = df.join(macd)
    df = df.join(macd_sign)
    df = df.join(macd_diff)
    return df


def generate_discretize_thresholds(df, steps):
    total_count = df.shape[0]
    step_size = total_count / steps
    columns = ['MACD', 'MACDSign', 'Force', 'Volume', 'Bollinger_Lower', 'Bollinger_Upper', 'EMA', 'Adj Close / SMA']
    threshold_df = pd.DataFrame(index=range(0, steps), columns=columns)

    sorted_by_force = df.sort_values(by='Force', axis=0, ascending=True)
    for i in range(0, steps):
        pointer = (i + 1) * step_size
        if i == steps - 1:
            threshold_df.ix[i, 'Force'] = sorted_by_force['Force'].iloc[-1]
        else:
            threshold_df.ix[i, 'Force'] = sorted_by_force['Force'].iloc[pointer]

    sorted_by_volume = df.sort_values(by='Volume', axis=0, ascending=True)
    for i in range(0, steps):
        pointer = (i + 1) * step_size
        if i == steps - 1:
            threshold_df.ix[i, 'Volume'] = sorted_by_volume['Volume'].iloc[-1]
        else:
            threshold_df.ix[i, 'Volume'] = sorted_by_volume['Volume'].iloc[pointer]

    sorted_by_macd = df.sort_values(by='MACD', axis=0, ascending=True)
    for i in range(0, steps):
        pointer = (i + 1) * step_size
        if i == steps - 1:
            threshold_df.ix[i, 'MACD'] = sorted_by_macd['MACD'].iloc[-1]
        else:
            threshold_df.ix[i, 'MACD'] = sorted_by_macd['MACD'].iloc[pointer]

    sorted_by_macd_sign = df.sort_values(by='MACDSign', axis=0, ascending=True)
    for i in range(0, steps):
        pointer = (i + 1) * step_size
        if i == steps - 1:
            threshold_df.ix[i, 'MACDSign'] = sorted_by_macd_sign['MACDSign'].iloc[-1]
        else:
            threshold_df.ix[i, 'MACDSign'] = sorted_by_macd_sign['MACDSign'].iloc[pointer]

    sorted_by_macd_diff = df.sort_values(by='MACDDiff', axis=0, ascending=True)
    for i in range(0, steps):
        pointer = (i + 1) * step_size
        if i == steps - 1:
            threshold_df.ix[i, 'MACDDiff'] = sorted_by_macd_diff['MACDDiff'].iloc[-1]
        else:
            threshold_df.ix[i, 'MACDDiff'] = sorted_by_macd_diff['MACDDiff'].iloc[pointer]

    sorted_by_bb_lower = df.sort_values(by='Bollinger_Lower', axis=0, ascending=True)
    for i in range(0, steps):
        pointer = (i + 1) * step_size
        if i == steps - 1:
            threshold_df.ix[i, 'Bollinger_Lower'] = sorted_by_bb_lower['Bollinger_Lower'].iloc[-1]
        else:
            threshold_df.ix[i, 'Bollinger_Lower'] = sorted_by_bb_lower['Bollinger_Lower'].iloc[pointer]

    sorted_by_bb_upper = df.sort_values(by='Bollinger_Upper', axis=0, ascending=True)
    for i in range(0, steps):
        pointer = (i + 1) * step_size
        if i == steps - 1:
            threshold_df.ix[i, 'Bollinger_Upper'] = sorted_by_bb_upper['Bollinger_Upper'].iloc[-1]
        else:
            threshold_df.ix[i, 'Bollinger_Upper'] = sorted_by_bb_upper['Bollinger_Upper'].iloc[pointer]

    sorted_by_ema = df.sort_values(by='EMA', axis=0, ascending=True)
    for i in range(0, steps):
        pointer = (i + 1) * step_size
        if i == steps - 1:
            threshold_df.ix[i, 'EMA'] = sorted_by_ema['EMA'].iloc[-1]
        else:
            threshold_df.ix[i, 'EMA'] = sorted_by_ema['EMA'].iloc[pointer]

    sorted_by_sma = df.sort_values(by='Adj Close / SMA', axis=0, ascending=True)
    for i in range(0, steps):
        pointer = (i + 1) * step_size
        if i == steps - 1:
            threshold_df.ix[i, 'Adj Close / SMA'] = sorted_by_sma['Adj Close / SMA'].iloc[-1]
        else:
            threshold_df.ix[i, 'Adj Close / SMA'] = sorted_by_sma['Adj Close / SMA'].iloc[pointer]
    columns = ['MACD', 'MACDSign', 'Force', 'Volume', 'Bollinger_Lower', 'Bollinger_Upper', 'EMA', 'Adj Close / SMA']
    x = {
        'MACD': threshold_df['MACD'].values,
        'MACDSign': threshold_df['MACDSign'].values,
        'Force': threshold_df['Force'].values,
        'Volume': threshold_df['Volume'].values,
        'Bollinger_Lower': threshold_df['Bollinger_Lower'].values,
        'Bollinger_Upper': threshold_df['Bollinger_Upper'].values,
        'EMA': threshold_df['EMA'].values,
        'Adj Close / SMA': threshold_df['Adj Close / SMA'].values,
    }
    return x


class StrategyLearner(object):
    # constructor
    def __init__(self, verbose=False):
        self.verbose = verbose

    def discretize(self, date, current_holding):
        # self.thresholds[]
        values = self.df.loc[date]
        volume_value = values['Volume']
        macd_value = values['MACD']
        force_value = values['Force']
        macd_sign_value = values['MACDSign']
        macd_diff_value = values['MACDDiff']
        bb_lower_value = values['Bollinger_Lower']
        bb_upper_value = values['Bollinger_Upper']
        ema_value = values['EMA']
        adj_sma_value = values['Adj Close / SMA']

        for index, elem in enumerate(self.thresholds['Force']):
            if elem >= force_value:
                force_threshold = index
                break

        for index, elem in enumerate(self.thresholds['MACD']):
            if elem >= macd_value:
                macd_threshold = index
                break

        for index, elem in enumerate(self.thresholds['Bollinger_Lower']):
            if elem >= bb_lower_value:
                bb_lower_threshold = index
                break

        for index, elem in enumerate(self.thresholds['Bollinger_Upper']):
            if elem >= bb_upper_value:
                bb_upper_threshold = index
                break

        for index, elem in enumerate(self.thresholds['Adj Close / SMA']):
            if elem >= adj_sma_value:
                adj_sma_threshold = index
                break

        holding_val = 0
        if current_holding > 0:
            holding_val = 4
        elif current_holding < 0:
            holding_val = 0
        else:
            holding_val = 2
        result = [int(macd_threshold),
                  int(bb_lower_threshold),
                  int(bb_upper_threshold),
                  # int(ema_threshold),
                  # int(force_threshold),
                  int(adj_sma_threshold),
                  int(holding_val)]
        discretized = (result[0] * 10000) + (result[1] * 1000) + (result[2] * 100) + (result[3] * 10) + (result[4])
        return discretized

    # this method should create a QLearner, and train it for trading
    def addEvidence(self, symbol="IBM", \
                    sd=dt.datetime(2008, 1, 1), \
                    ed=dt.datetime(2009, 1, 1), \
                    sv=10000):

        # add your code to do learning here


        # example usage of the old backward compatible util function
        syms = [symbol]
        sd_earlier = sd - dt.timedelta(days=50)
        dates = pd.date_range(sd_earlier, ed)
        prices_all = ut.get_data(syms, dates)  # automatically adds SPY
        prices = prices_all[syms]  # only portfolio symbols
        prices_SPY = prices_all['SPY']  # only SPY, for comparison later
        if self.verbose: print prices

        # example use with new colname
        volume_all = ut.get_data(syms, dates, colname="Volume")  # automatically adds SPY
        volume = volume_all[syms]  # only portfolio symbols

        prices_and_volumes = pd.DataFrame(data=None, index=prices.index, columns=['Adj Close', 'Volume'])
        prices_and_volumes['Adj Close'] = prices[symbol]
        prices_and_volumes['Volume'] = volume[symbol]
        df_with_macd = calc_macd(prices_and_volumes, n_fast=12, n_slow=26)
        df_with_ema = calc_ema(df_with_macd, n=9)
        df_with_bb_bands = calc_bb_bands(df_with_ema, n=20)
        df_with_force = calc_force_index(df_with_bb_bands, n=9)
        df_with_sma = calc_sma(df_with_force, n=9)
        self.df = df_with_sma.loc[df_with_sma.index >= sd]
        self.thresholds = generate_discretize_thresholds(self.df, steps=5)

        self.learner = ql.QLearner(
            num_states=55555,
            num_actions=3,
            alpha=0.3,
            gamma=0.9,
            rar=0.2,
            radr=0.999,
            dyna=0,
            verbose=False
        )
        iterations = 20000
        scores = np.zeros((iterations, 1))
        first_trading_day = self.df.index[0]
        start = time.time()
        iteration = 1
        while (time.time() - start < 28):
            total_reward = 0
            state = self.discretize(date=first_trading_day, current_holding=0)
            action = self.learner.querysetstate(s=state)
            cash_total = sv
            stock_qty = 0
            previous_nav = sv
            new_nav = 0
            for td in self.df.index:
                current_price = self.df['Adj Close'][td]
                if td == first_trading_day:
                    cash_total = sv
                    continue
                # Sell
                if action == 0 and stock_qty >= 0:
                    stock_current_value = (2000 * current_price)
                    cash_total += stock_current_value
                    stock_qty = stock_qty - 2000
                    stock_holdings_total = stock_qty * current_price
                    new_nav = cash_total + stock_holdings_total
                # Buy
                elif action == 2 and stock_qty <= 0:
                    stock_current_value = (2000 * current_price)
                    cash_total -= stock_current_value
                    stock_qty = stock_qty + 2000
                    stock_holdings_total = stock_qty * current_price
                    new_nav = cash_total + stock_holdings_total
                # Hold
                else:
                    stock_holdings_total = current_price * stock_qty
                    new_nav = cash_total + stock_holdings_total

                reward = 0 if previous_nav == 0.0 else (new_nav / previous_nav) - 1.0
                total_reward += reward
                discretized_state = self.discretize(td, stock_qty)
                action = self.learner.query(discretized_state, reward)
                previous_nav = new_nav

            if self.verbose: print  str(iteration) + ':' + str(total_reward)
            scores[iteration - 1, 0] = total_reward
            iteration += 1

            # if self.verbose: print volume

    # this method should use the existing policy and test it against new data
    def testPolicy(self, symbol="IBM", \
                   sd=dt.datetime(2009, 1, 1), \
                   ed=dt.datetime(2010, 1, 1), \
                   sv=10000):
        # here we build a fake set of trades
        # your code should return the same sort of data
        sd_earlier = sd - dt.timedelta(days=50)
        dates = pd.date_range(sd_earlier, ed)
        syms = [symbol]

        prices_all = ut.get_data([symbol], dates)  # automatically adds SPY
        prices = prices_all[syms]  # only portfolio symbols

        # example use with new colname
        volume_all = ut.get_data(syms, dates, colname="Volume")  # automatically adds SPY
        volume = volume_all[syms]  # only portfolio symbols

        prices_and_volumes = pd.DataFrame(data=None, index=prices.index, columns=['Adj Close', 'Volume'])
        prices_and_volumes['Adj Close'] = prices[symbol]
        prices_and_volumes['Volume'] = volume[symbol]
        df_with_macd = calc_macd(prices_and_volumes, n_fast=12, n_slow=26)
        df_with_ema = calc_ema(df_with_macd, n=9)
        df_with_bb_bands = calc_bb_bands(df_with_ema, n=20)
        df_with_force = calc_force_index(df_with_bb_bands, n=12)
        df_with_sma = calc_sma(df_with_force, n=9)
        self.df = df_with_sma.loc[df_with_sma.index >= sd]
        self.thresholds = generate_discretize_thresholds(self.df, steps=5)
        first_trading_day = self.df.index[0]
        state = self.discretize(date=first_trading_day, current_holding=0)
        action = self.learner.querysetstate(s=state)
        portfolio_df = pd.DataFrame(index=self.df.index,
                                    columns=['Adj Close',
                                             'Stock Trade',
                                             'Stock Total',
                                             'Cash Trade',
                                             'Cash Total',
                                             'NAV',
                                             'Daily Return']).fillna(value=0)

        for td in self.df.index:
            current_price = self.df['Adj Close'][td]
            portfolio_df['Adj Close'][td] = current_price
            if td == first_trading_day:
                portfolio_df['Cash Trade'][td] = sv
                portfolio_df['Cash Total'][td] = sv
                portfolio_df['NAV'][td] = sv
                continue

            current_units = portfolio_df['Stock Trade'].sum(axis=0)
            current_cash_bal = portfolio_df['Cash Trade'].sum(axis=0)

            if action == 0 and current_units >= 0:
                incoming_cash = (200 * current_price)
                portfolio_df['Stock Trade'][td] = -2000
                portfolio_df['Stock Total'][td] = current_units - 2000
                portfolio_df['Cash Trade'][td] = incoming_cash
                portfolio_df['Cash Total'][td] = current_cash_bal + incoming_cash
                new_nav = portfolio_df['Cash Total'][td] + (portfolio_df['Stock Total'][td] * current_price)
                portfolio_df['NAV'][td] = new_nav
            elif action == 2 and current_units <= 0:
                outgoing_cash = (200 * current_price)
                portfolio_df['Stock Trade'][td] = (+2000)
                portfolio_df['Stock Total'][td] = current_units + 2000
                portfolio_df['Cash Trade'][td] = - outgoing_cash
                portfolio_df['Cash Total'][td] = current_cash_bal - outgoing_cash
                new_nav = portfolio_df['Cash Total'][td] + (portfolio_df['Stock Total'][td] * current_price)
                portfolio_df['NAV'][td] = new_nav
            else:
                portfolio_df['Stock Trade'][td] = 0
                portfolio_df['Stock Total'][td] = current_units
                portfolio_df['Cash Trade'][td] = 0
                portfolio_df['Cash Total'][td] = current_cash_bal
                new_nav = portfolio_df['Cash Total'][td] + (portfolio_df['Stock Total'][td] * current_price)
                portfolio_df['NAV'][td] = new_nav
            state = self.discretize(date=td, current_holding=portfolio_df['Stock Total'][td])
            action = self.learner.querysetstate(state)

        trades = portfolio_df[['Stock Trade']]
        trades.columns = [symbol]
        if self.verbose: print type(trades)  # it better be a DataFrame!
        if self.verbose:
            print trades

        #print trades
        # if self.verbose: print prices_all
        return trades


if __name__ == "__main__":
    print "One does not simply think up a strategy"
