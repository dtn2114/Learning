import pandas as pd
import numpy as np

if __name__ == "__main__":
    raw_data = {'Date': ['2011-01-05', '2011-01-20'], 'Symbol': ['AAPL','AAPL'], \
                    'Order': ['BUY', 'SELL'], 'Shares': ['1500','1500']}
    df = pd.DataFrame(raw_data, columns=['Date','Symbol', 'Order', 'Shares'])
    print df
    df.to_csv('./example.csv')
