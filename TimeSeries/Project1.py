import warnings

import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import ConvergenceWarning
import numpy as np
from sklearn.linear_model import LinearRegression

date_column_start = 5
date_train_start = pd.to_datetime('2010-1-31')
date_train_end = pd.to_datetime('2017-12-31')
date_train_end1 = pd.to_datetime('2019-12-31')
zillow_house_prices = pd.read_csv('data_zillow_house_prices.csv')
prices = zillow_house_prices[zillow_house_prices['RegionName'] == 'Boston, MA'].iloc[0][date_column_start:]
date = pd.to_datetime(zillow_house_prices.columns[date_column_start:])

plt.plot(date, prices)
plt.title("Boston_MA price over time")
plt.show()

prices1 = prices[date.get_loc(date_train_start):date.get_loc(date_train_end) + 1]
date1 = np.arange(len(prices1)).reshape(-1, 1)
dft = sm.tsa.stattools.adfuller(prices1)
print(dft)

model_linear = LinearRegression()
model_linear.fit(date1, prices1)
linear_trend = model_linear.predict(date1)
prices_remove_linear = prices1 - linear_trend
dft1 = sm.tsa.stattools.adfuller(prices_remove_linear)
print(dft1)

X_quad = np.hstack([date1, date1 ** 2])
model_quad = LinearRegression()
model_quad.fit(X_quad, prices1)
quad_trend = model_quad.predict(X_quad)
prices_remove_linear_quad = prices1 - quad_trend
dft2 = sm.tsa.stattools.adfuller(prices_remove_linear_quad)
print(dft2)

X_cubic = np.hstack([date1, date1 ** 2, date1 ** 3])
model_cubic = LinearRegression()
model_cubic.fit(X_cubic, prices1)
cubic_trend = model_cubic.predict(X_cubic)
prices_remove_linear_quad_cubic = prices1 - cubic_trend
dft3 = sm.tsa.stattools.adfuller(prices_remove_linear_quad)
print(dft3)


# sm.graphics.tsa.plot_acf(prices_remove_linear, lags=90)
# plt.show()
# sm.graphics.tsa.plot_pacf(prices_remove_linear)
# plt.show()
# plt.plot(prices_remove_linear)
# plt.show()

def mse(y_true, y_pred):
    return np.mean((np.array(y_true) - np.array(y_pred)) ** 2)


for i in range(5):
    model_ar = sm.tsa.ARIMA(prices_remove_linear.to_list(), order=(i + 1, 0, 0), trend='n').fit()
    prices_predict = model_ar.predict()
    print("mse: ", mse(prices_remove_linear, prices_predict))
    # print(model_ar.summary())

may_2018 = 5
pre_may_2018 = pd.to_datetime('2018-5-31')
date_test_mse = date.get_loc(pre_may_2018)
may_2018_price = prices[date.get_loc(pre_may_2018)]

linear_model = sm.OLS(np.array(prices1), sm.add_constant(date1))
linear_results = linear_model.fit()
linear_may_2018 = linear_results.predict([[1, len(date1) + may_2018 - 1]])[0]
model_ar = sm.tsa.ARIMA(prices_remove_linear.to_list(), order=(2, 0, 0), trend='n').fit()
ar_may_2018 = model_ar.forecast(may_2018)[-1]
mes_may_2018_long = mse(may_2018_price, linear_may_2018 + ar_may_2018)

prices2 = prices[date.get_loc(date_train_start): date.get_loc(pre_may_2018)]
date2 = np.arange(len(prices2))
linear_trend_may_2018 = linear_results.predict(sm.add_constant(date2))
prices_remove_linear_may_2018 = prices2 - linear_trend_may_2018
model_ar_may_2018 = sm.tsa.ARIMA(prices_remove_linear_may_2018.to_list(), order=(2, 0, 0), trend='n').fit()
ar_may_2018_short = model_ar_may_2018.forecast(1)[-1]
mes_may_2018_short = mse(may_2018_price, linear_may_2018 + ar_may_2018_short)
print(mes_may_2018_long, mes_may_2018_short)

test_num = 23
linear_test = linear_results.predict([[1, len(date1) + i] for i in range(test_num)])
ar_test = model_ar.forecast(test_num)
mes_test_long = mse(prices[date.get_loc(date_train_end) + 1: date.get_loc(date_train_end) + test_num + 1], linear_test + ar_test)
total_mse = 0
test_linear = linear_results.predict(sm.add_constant(np.arange(len(prices1) + test_num + 1)))
for i in range(test_num + 1):
    test_price = prices[date.get_loc(date_train_start):date.get_loc(date_train_end) + 1 + i]
    test_prices_remove_linear = test_price - test_linear[:i - test_num - 1]

    test_model_ar1 = sm.tsa.ARIMA(test_prices_remove_linear.to_list(), order=(2, 0, 0), trend='n').fit()
    a = test_model_ar1.forecast(1)
    total_mse += mse(prices[date.get_loc(date_train_end) + 1 + i], test_linear[len(test_price)] + a)

print(mes_test_long, total_mse / (test_num + 1))

interest_rates = pd.read_csv('data_interest_rates.csv', parse_dates=['DATE']).set_index('DATE')
rate = interest_rates.resample('M').mean()['2010':'2019']['MORTGAGE30US']
warnings.simplefilter("ignore", (ConvergenceWarning, UserWarning))

for i in range(4):
    model_ar = sm.tsa.ARIMA(prices_remove_linear.to_list(), rate[:len(prices_remove_linear)], order=(i + 1, 0, 0), trend='n').fit()
    prices_predict = model_ar.predict()
    print(f"mse in-sample: {i+1}", mse(prices_remove_linear, prices_predict))
    model_ar = sm.tsa.ARIMA(prices_remove_linear.to_list(), order=(i + 1, 0, 0), trend='n').fit()
    prices_predict = model_ar.predict()
    print(f"mse in-sample: {i+1}", mse(prices_remove_linear, prices_predict), 'no exog')
    total_mse = 0
    total_mse1 = 0
    warnings.simplefilter("ignore", ConvergenceWarning)
    for x in range(test_num + 1):
        test_price = prices[date.get_loc(date_train_start):date.get_loc(date_train_end) + 1 + x]
        test_prices_remove_linear = test_price - test_linear[:x - test_num - 1]

        test_model_ar = sm.tsa.ARIMA(test_prices_remove_linear.to_list(), rate[:len(prices_remove_linear) + x], order=(i + 1, 0, 0), trend='n').fit()
        test_ar = test_model_ar.forecast(1, exog=rate[len(prices_remove_linear) + x])
        total_mse += mse(prices[date.get_loc(date_train_end) + 1 + x], test_linear[len(test_price)] + test_ar)

        test_model_ar1 = sm.tsa.ARIMA(test_prices_remove_linear.to_list(), order=(i + 1, 0, 0), trend='n').fit()
        test_ar1 = test_model_ar1.forecast(1)
        total_mse1 += mse(prices[date.get_loc(date_train_end) + 1 + x], test_linear[len(test_price)] + test_ar1)

    print(f"mse ot-sample: {i+1}", total_mse / (test_num + 1))
    print(f"mse ot-sample: {i+1}", total_mse1 / (test_num + 1), 'no exog')

# for p in range(4):
#     for q in [1, 5, 10]:
#         model_ar = sm.tsa.ARIMA(prices_remove_linear.to_list(), rate[:len(prices_remove_linear)], order=(p + 1, 0, q), trend='n').fit()
#         prices_predict = model_ar.predict()
#         # print(f"p:{p+1}, q:{q} mse", mse(prices_remove_linear, prices_predict))
#         total_mse = 0
#
#         for x in range(test_num + 1):
#             test_price = prices[date.get_loc(date_train_start):date.get_loc(date_train_end) + 1 + x]
#             test_prices_remove_linear = test_price - test_linear[:x - test_num - 1]
#             test_model_ar = sm.tsa.ARIMA(test_prices_remove_linear.to_list(), rate[:len(prices_remove_linear) + x], order=(p + 1, 0, q), trend='n').fit()
#             test_ar = test_model_ar.forecast(1, exog=rate[len(prices_remove_linear) + x])
#             total_mse += mse(prices[date.get_loc(date_train_end) + 1 + x], test_linear[len(test_price)] + test_ar)
#         print(f"p:{p+1}, q:{q} mse_test", total_mse / (test_num + 1))

prices3 = prices[date.get_loc(date_train_start):date.get_loc(date_train_end1) + 1]
date3 = np.arange(len(prices3))
rate3 = interest_rates.resample('M').mean()['2010':'2021']['MORTGAGE30US']

linear_model3 = sm.OLS(np.array(prices3), sm.add_constant(date3))
linear_results3 = linear_model3.fit()
linear_predict3 = linear_results3.predict()
prices_remove_linear3 = prices3 - linear_predict3


model_arx3 = sm.tsa.ARIMA(prices_remove_linear3.to_list(), rate[:len(prices_remove_linear3)], order=(2, 0, 0), trend='n').fit()
prices_predict3x = model_arx3.predict()
print("mse in-sample:  ", mse(prices_remove_linear3, prices_predict3x))

model_ar3 = sm.tsa.ARIMA(prices_remove_linear3.to_list(), order=(2, 0, 0), trend='n').fit()
prices_predict3 = model_ar3.predict()
print("mse in-sample:  ", mse(prices_remove_linear3, prices_predict3), 'no exog')

total_mse = 0
total_mse1 = 0

linear_test3 = linear_results3.predict(sm.add_constant(np.arange(len(prices3) + test_num + 1)))
for x in range(test_num + 1):
    test_price = prices[date.get_loc(date_train_start):date.get_loc(date_train_end1) + 1 + x]
    test_prices_remove_linear = test_price - linear_test3[:x - test_num - 1]

    test_model_ar = sm.tsa.ARIMA(test_prices_remove_linear.to_list(), rate3[:len(test_prices_remove_linear)], order=(2, 0, 0), trend='n').fit()
    test_ar = test_model_ar.forecast(1, exog=rate3[len(test_prices_remove_linear)])
    total_mse += mse(prices[date.get_loc(date_train_end1) + 1 + x], linear_test3[len(test_price)] + test_ar)

    test_model_ar1 = sm.tsa.ARIMA(test_prices_remove_linear.to_list(), order=(2, 0, 0), trend='n').fit()
    test_ar1 = test_model_ar1.forecast(1)
    total_mse1 += mse(prices[date.get_loc(date_train_end1) + 1 + x], linear_test3[len(test_price)] + test_ar1)
print(f"mse ot-sample:  ", total_mse / (test_num + 1))
print(f"mse ot-sample:  ", total_mse1 / (test_num + 1), 'no exog')
