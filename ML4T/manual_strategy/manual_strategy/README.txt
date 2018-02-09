Overall, you can run testPolicy() from BestPossibleStrategy.py and ManualStrategy.py to get the df_trades data frame. Specific description of each python files is listed below.
There are 4 different files:
1. Indicators.py
This file contains the codes for various technical indicators (SMA, bband, momentum, vol)
sma = sma(price, n)
bbp = bband(price, n, chart=False)
momentum = momentum(price, n, chart=False)
vol = vol(price, n)
Each of the above function will take in price data frame and return the corresponding data frame of that indicator. 
2. BestPossibleStrategy.py
This file has 2 functions: testPolicy and testCode
* df_trades = testPolicy (symbol = "AAPL", sd=dt.datetime(2010,1,1), ed=dt.datetime(2011,12,31), sv = 100000)
This function will return df_trades (A data frame whose values represent trades for each day. Legal values are +1000.0 indicating a BUY of 1000 shares, -1000.0 indicating a SELL of 1000 shares, and 0.0 indicating NOTHING. Values of +2000 and -2000 for trades are also legal so long as net holdings are constrained to -1000, 0, and 1000)
* testCode(symbol = "AAPL", sd=dt.datetime(2010,1,1), ed=dt.datetime(2011,12,31), sv = 100000, chart = False)
This code will run the testPolicy, take the df_trades, compute the portvals by calling compute_portvals() from marketsimcode.py
Then it will run to generate the chart and statistics to compare best possible and benchmark
3. ManualStrategy.py
This file has 2 functions: testPolicy and testCode
* df_trades = testPolicy (symbol = "AAPL", sd=dt.datetime(2010,1,1), ed=dt.datetime(2011,12,31), sv = 100000)
This function will return df_trades (A data frame whose values represent trades for each day. Legal values are +1000.0 indicating a BUY of 1000 shares, -1000.0 indicating a SELL of 1000 shares, and 0.0 indicating NOTHING. Values of +2000 and -2000 for trades are also legal so long as net holdings are constrained to -1000, 0, and 1000)
* testCode(symbol = "AAPL", sd=dt.datetime(2010,1,1), ed=dt.datetime(2011,12,31), sv = 100000, chart = False)
This code will run the testPolicy, take the df_trades, compute the portvals by calling compute_portvals() from marketsimcode.py
Then it will run to generate the chart and statistics to compare manual strategy and benchmark
In the ‘__main__’, user can change the InSample flag from True to False to run the OutofSample period.
4. Marketsimcode.py
This code contain the update compute_portvals that i being called by the manualstrategy.py and bestpossiblestrategy.py
Compute_portvals now take in the df_trades (explained above) then compute portfolio vals. 
