"""unit tests for optimization.py"""

# Updated 9/9/17 6:00 PM PST
import datetime as dt
from time import time
import unittest
import optimization


def str2dt(strng):
    year, month, day = map(int, strng.split('-'))
    return dt.datetime(year, month, day)


class TestOptimization(unittest.TestCase):
    def test_optimization_grade_case_1(self):
        params = dict(
            start_date=str2dt('2010-01-01'),
            end_date=str2dt('2010-12-31'),
            symbols=['GOOG', 'AAPL', 'GLD', 'XOM']
        )
        results = dict(
            allocs=[0.10612267, 0.00777928, 0.54377087, 0.34232718],
            volatility=0.00828691718086,  # BPH: updated from reference solution, Sunday 3 Sep 2017
            min_student_volatility=0.00823918411957
        )
        self._run_test(params, results)

    def test_optimization_grade_case_2(self):
        params = dict(
            start_date=str2dt('2004-01-01'),
            end_date=str2dt('2006-01-01'),
            symbols=['AXP', 'HPQ', 'IBM', 'HNZ']
        )
        results = dict(
            allocs=[0.29856713, 0.03593918, 0.29612935, 0.36936434],
            volatility=0.00706292107796,  # BPH: updated from reference solution, Sunday 3 Sep 2017
            min_student_volatility=0.00706292107796
        )
        self._run_test(params, results)

    def test_optimization_grade_case_3(self):
        params = dict(
            start_date=str2dt('2004-12-01'),
            end_date=str2dt('2006-05-31'),
            symbols=['YHOO', 'XOM', 'GLD', 'HNZ']
        )
        results = dict(
            allocs=[0.05963382, 0.07476148, 0.31764505, 0.54795966],
            volatility=0.00700653270334,  # BPH: updated from reference solution, Sunday 3 Sep 2017
            min_student_volatility=0.00698733578037
        )
        self._run_test(params, results)

    def test_optimization_grade_case_4(self):
        params = dict(
            start_date=str2dt('2005-12-01'),
            end_date=str2dt('2006-05-31'),
            symbols=['YHOO', 'HPQ', 'GLD', 'HNZ']
        )
        results = dict(
            allocs=[0.10913451, 0.19186373, 0.15370123, 0.54530053],
            volatility=0.00789501806472,  # BPH: updated from reference solution, Sunday 3 Sep 2017
            min_student_volatility=0.00789470061851
        )
        self._run_test(params, results)

    def test_optimization_grade_case_5(self):
        params = dict(
            start_date=str2dt('2005-12-01'),
            end_date=str2dt('2007-05-31'),
            symbols=['MSFT', 'HPQ', 'GLD', 'HNZ']
        )
        results = dict(
            allocs=[0.29292607, 0.10633076, 0.14849462, 0.45224855],
            volatility=0.00688155185985,  # BPH: updated from reference solution, Sunday 3 Sep 2017
            min_student_volatility=0.0068774868643
        )
        self._run_test(params, results)

    def test_optimization_grade_case_6(self):
        params = dict(
            start_date=str2dt('2006-05-31'),
            end_date=str2dt('2007-05-31'),
            symbols=['MSFT', 'AAPL', 'GLD', 'HNZ']
        )
        results = dict(
            allocs=[0.20500321, 0.05126107, 0.18217495, 0.56156077],
            volatility=0.00693253248047,  # BPH: updated from reference solution, Sunday 3 Sep 2017
            min_student_volatility=0.00692979997927
        )
        self._run_test(params, results)

    def test_optimization_grade_case_7(self):
        params = dict(
            start_date=str2dt('2011-01-01'),
            end_date=str2dt('2011-12-31'),
            symbols=['AAPL', 'GLD', 'GOOG', 'XOM']
        )
        results = dict(
            allocs=[0.15673037, 0.51724393, 0.12608485, 0.19994085],
            volatility=0.0096198317644,  # BPH: updated from reference solution, Sunday 3 Sep 2017
            min_student_volatility=0.0096170370471
        )
        self._run_test(params, results)

    def test_optimization_grade_case_8(self):
        params = dict(
            start_date=str2dt('2010-06-01'),
            end_date=str2dt('2011-06-01'),
            symbols=['AAPL', 'GLD', 'GOOG']
        )
        results = dict(
            allocs=[0.21737029, 0.66938007, 0.11324964],
            volatility=0.00799161174614,  # BPH: updated from reference solution, Sunday 3 Sep 2017
            min_student_volatility=0.00797292974916
        )

        self._run_test(params, results)

    def test_optimization_reddit_c000ldin_test_1(self):
        # from https://www.reddit.com/r/cs7646_fall2017/comments/6ylil6/did_you_get_the_same_results_as_mine_in_project_2/
        params = {
            'start_date': dt.datetime(2008, 6, 1),
            'end_date': dt.datetime(2009, 6, 1),
            'symbols': ['IBM', 'X', 'GLD'],
        }
        results = {
            'volatility': 0.0155964503875
        }
        self._run_test(params, results)

    def test_optimization_reddit_c000ldin_test_2(self):
        # from https://www.reddit.com/r/cs7646_fall2017/comments/6ylil6/did_you_get_the_same_results_as_mine_in_project_2/
        params = {
            'start_date': dt.datetime(2010, 1, 1),
            'end_date': dt.datetime(2010, 12, 31),
            'symbols': ['GOOG', 'AAPL', 'GLD', 'XOM', 'IBM'],
        }
        results = {
            'volatility': 0.00789304109174
        }
        self._run_test(params, results)

    def test_optimization_reddit_c000ldin_test_3(self):
        # from https://www.reddit.com/r/cs7646_fall2017/comments/6ylil6/did_you_get_the_same_results_as_mine_in_project_2/
        params = {
            'start_date': dt.datetime(2008, 1, 1),
            'end_date': dt.datetime(2009, 1, 1),
            'symbols': ['GOOG', 'AAPL', 'GLD', 'XOM', 'IBM'],
        }
        results = {
            'volatility': 0.0147158024535
        }
        self._run_test(params, results)

    def test_assess_portfolio_missing_data(self):
        params = {
            'start_date': dt.datetime(2007, 1, 1),
            'end_date': dt.datetime(2010, 12, 31),
            'symbols': ['FAKE1', 'FAKE2']
        }

        results = {
            'volatility': 0.00947011321946
        }

        self._run_test(params, results)

    def test_assess_portfolio_community_1(self):
        params = {
            'start_date': dt.datetime(2010, 9, 1),
            'end_date': dt.datetime(2010, 12, 31),
            'symbols': ['IYR']
        }

        results = {
            'volatility': 0.0108856018686
        }

        self._run_test(params, results)

    def test_assess_portfolio_community_2(self):
        params = {
            'start_date': dt.datetime(2009, 7, 2),
            'end_date': dt.datetime(2010, 7, 30),
            'symbols': ['USB', 'VAR']
        }

        results = {
            'volatility': 0.0152323743461
        }

        self._run_test(params, results)

    def test_assess_portfolio_community_3(self):
        params = {
            'start_date': dt.datetime(2008, 6, 3),
            'end_date': dt.datetime(2010, 6, 29),
            'symbols': ['HSY', 'VLO', 'HOT']
        }

        results = {
            'volatility': 0.0194739111493
        }

        self._run_test(params, results)

    def test_assess_portfolio_community_4(self):
        params = {
            'start_date': dt.datetime(2007, 5, 4),
            'end_date': dt.datetime(2010, 5, 28),
            'symbols': ['VNO', 'WU', 'EMC', 'AMGN']
        }

        results = {
            'volatility': 0.0191133197854
        }

        self._run_test(params, results)

    def test_assess_portfolio_community_5(self):
        params = {
            'start_date': dt.datetime(2006, 4, 5),
            'end_date': dt.datetime(2010, 4, 26),
            'symbols': ['ADSK', 'BXP', 'IGT', 'SWY', 'PH']
        }

        results = {
            'volatility': 0.0183482809441
        }

        self._run_test(params, results)

    def test_assess_portfolio_community_6(self):
        params = {
            'start_date': dt.datetime(2005, 4, 6),
            'end_date': dt.datetime(2010, 3, 25),
            'symbols': ['ETN', 'KSS', 'NYT', 'GPS', 'BMC', 'TEL']
        }

        results = {
            'volatility': 0.0160628925927,
        }

        self._run_test(params, results)

    def test_assess_portfolio_community_7(self):
        params = {
            'start_date': dt.datetime(2003, 2, 8),
            'end_date': dt.datetime(2010, 1, 23),
            'symbols': ['HRL', 'SDS', 'ACS', 'IFF', 'WMB', 'FFIV', 'BK', 'AIV']
        }

        results = {
            'volatility': 0.00597684040007
        }

        self._run_test(params, results)

    def test_assess_portfolio_community_8(self):
        params = {
            'start_date': dt.datetime(2002, 2, 9),
            'end_date': dt.datetime(2010, 10, 22),
            'symbols': ['CCT', 'JNJ', 'ERTS', 'MCO', 'R', 'WDC', 'BBT', 'JOY', 'PLL']
        }

        results = {
            'volatility': 0.00816054739105
        }

        self._run_test(params, results)

    def test_assess_portfolio_community_9(self):
        params = {
            'start_date': dt.datetime(2001, 1, 10),
            'end_date': dt.datetime(2010, 11, 20),
            'symbols': ['WWY', 'OMX', 'NFX', 'AVB', 'EW', 'JWN', 'CBS', 'SH', 'UNH', 'CCI']
        }

        results = {
            'volatility': 0.00305975409922

        }

        self._run_test(params, results)

    # =================================================
    #       Main Test Code
    # =================================================

    def _run_test(self, params, results):

        # the margins below are taken from grade_optimization.py
        abs_margins = dict(sum_to_one=0.02, alloc_range=0.02, alloc_match=0.1,
                           sddr_match=0.05)  # absolute margin of error for each component

        start_date = params['start_date']
        end_date = params['end_date']
        symbols = params['symbols']

        start_time = time()

        allocations, \
        cumulative_return, \
        average_daily_return, \
        volatility, \
        sharpe_ratio = optimization.optimize_portfolio(sd=start_date, ed=end_date, syms=symbols, gen_plot=False)

        end_time = time()
        function_runtime = end_time - start_time

        # check runtime < 5 seconds
        # =======================================
        self.assertLessEqual(
            function_runtime,
            5,
            msg="Runtime violation, expected < 5.0s, runtime was {:1.3f} s".format(end_time - start_time)
        )

        # check volatility
        # =======================================
        self.assertLessEqual(
            (volatility / results['volatility']) - 1,
            abs_margins['sddr_match'],
            msg="Sddr too large: {:1.6f} (expected < {:1.6f} + {:1.6f})".format(volatility, results['volatility'],
                                                                                results['volatility'] * abs_margins[
                                                                                    'sddr_match'])
        )

        # check that sddr is not significantly less than expected, this would be the case if the student answers are
        # wrong or you are calculating sddr incorrectly. OR you found a global minimum below what other students have
        # =======================================
        # Note I was able to slightly beat the grade_optimization.py volatility slightly using a nifty trick so I
        # created results "min_student_volatility" as well to capture this.  The testcase below will error if you beat
        # either the latest students volatility, if provided, or the benchmark volatility if there it is not a testcase
        # from the grade_optimization.py
        test_volatility = results['volatility']
        if 'min_student_volatility' in results:
            test_volatility = results['min_student_volatility']
        self.assertGreaterEqual(
            volatility + 1e-8,  # add small delta to account for floating point errors
            test_volatility,
            msg="WARNING: your Volatility {} is less than the expected minimum {}, Verify your code and notify your classmates on Reddit".format(
                volatility, test_volatility)
        )

        # check allocation sum is ~1.0
        # =======================================
        self.assertLessEqual(
            abs(sum(allocations) - 1),
            abs_margins['sum_to_one'],
            msg="sum of allocations: {:1.5f} (expected: 1.0)".format(sum(allocations))
        )

        # check allocations are within tolerance
        # =======================================
        # This loops through each symbol and errors if the allocation is out range
        for symbol, alloc in zip(symbols, allocations):
            self.assertLessEqual(
                alloc,
                1 + abs_margins['alloc_range'],
                msg="{} - allocation out of range: {:1.5f} (expected [0.0, 1.0)".format(symbol, alloc)
            )
            self.assertGreaterEqual(
                alloc,
                -1 * abs_margins['alloc_range'],
                msg="{} - allocation out of range: {:1.5f} (expected [0.0, 1.0)".format(symbol, alloc)
            )
            # print allocations


if __name__ == '__main__':
    unittest.main(verbosity=2)
