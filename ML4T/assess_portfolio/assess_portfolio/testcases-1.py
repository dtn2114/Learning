#s file with 'PYTHONPATH=..:. python testcases-1.py'
#	diff the results with testcases-1.output 
#	discliamer : the output provided is not voted to be correct/perfect... please report differeneces you see. thanks

import pandas as pd
import datetime as dt
import math
import analysis
from util import get_data

def run_test(start_date, end_date, symbols, allocations, start_val, risk_free_rate, sample_freq):

    # Assess the portfolio
    cr, adr, sddr, sr, ev = analysis.assess_portfolio(sd = start_date, ed = end_date,\
        syms = symbols, \
        allocs = allocations,\
        sv = start_val, \
	rfr = risk_free_rate, \
	sf = sample_freq, \
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
    print "End Value:", ev


#wiki test cases
run_test(dt.datetime(2010,1,1), dt.datetime(2010,12,31),['GOOG', 'AAPL', 'GLD', 'XOM'],[0.2, 0.3, 0.4, 0.1],1000000,0.0,252)
run_test(dt.datetime(2010,1,1), dt.datetime(2010, 12, 31),['AXP', 'HPQ', 'IBM', 'HNZ'],[0.0, 0.0, 0.0, 1.0],1000000,0.0,252)
run_test(dt.datetime(2010,6,1), dt.datetime(2010,12,31),['GOOG', 'AAPL', 'GLD', 'XOM'],[0.2, 0.3, 0.4, 0.1],1000000,0.0,252)

#additional test cases
run_test(dt.datetime(2010,9,1), dt.datetime(2010,12,31),['IYR'],[1.0],1000000,0.01,252)
run_test(dt.datetime(2009,7,2), dt.datetime(2010,7,30),['USB','VAR'],[0.3, 0.7],1000000,0.02,252)
run_test(dt.datetime(2008,6,3), dt.datetime(2010,6,29),['HSY','VLO','HOT'],[0.2, 0.4, 0.4],1000000,0.03,252)
run_test(dt.datetime(2007,5,4), dt.datetime(2010,5,28),['VNO','WU','EMC','AMGN'],[0.2, 0.3, 0.4, 0.1],1000000,0.04,252)
run_test(dt.datetime(2006,4,5), dt.datetime(2010,4,26),['ADSK','BXP','IGT','SWY','PH'],[0.2, 0.3, 0.1, 0.2, 0.2],1000000,0.05,252)
run_test(dt.datetime(2005,4,6), dt.datetime(2010,3,25),['ETN','KSS','NYT','GPS','BMC','TEL'],[0.2, 0.1, 0.1, 0.1, 0.4, 0.1],1000000,0.06,252)
run_test(dt.datetime(2003,2,8), dt.datetime(2010,1,23),['HRL','SDS','ACS','IFF','WMB','FFIV','BK','AIV'],[0.2, 0.2, 0.1, 0.1, 0.2, 0, 0.2, 0],1000000,0.08,252)
run_test(dt.datetime(2002,2,9), dt.datetime(2010,10,22),['CCT','JNJ','ERTS','MCO','R','WDC','BBT','JOY','PLL'],[0.2, 0.2, 0.1, 0.1, 0.2, 0, 0, 0.2, 0],1000000,0.09,252)
run_test(dt.datetime(2001,1,10), dt.datetime(2010,11,20),['WWY','OMX','NFX','AVB','EW','JWN','CBS','SH','UNH','CCI'],[0.2, 0.1, 0.1, 0.1, 0.2, 0, 0.1, 0, 0.2, 0],1000000,0.1,252)
