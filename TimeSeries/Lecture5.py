import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg

CPI = pd.read_csv('CPI.csv', parse_dates=['date'])
# CPI = np.loadtxt('CPI.csv', delimiter=',', skiprows=0)
CPI['YearMonth'] = CPI['date'].dt.to_period('M')
cpi_by_year_month = CPI.groupby('YearMonth').mean().reset_index('YearMonth').dropna()
# stats.probplot(average_by_year_month[['YearMonth', 'CPI']], plot=plt)
# 绘制 'Value_A' 列的线图，X轴使用索引
# plt.figure(figsize=(10, 5))
# cpi_by_year_month.plot(x='YearMonth', y='CPI', kind='line', ax=plt.gca())
# # plt.plot(average_by_year_month['YearMonth'].astype(), average_by_year_month['CPI'], label='CPI Trend')
# plt.title('CPI 的趋势')
# plt.xlabel('日期')
# plt.ylabel('CPI')
# # plt.grid(True)  # 显示网格
# plt.show()
train_end_date = pd.to_datetime('2013-09-01')
cpi_by_year_month['YearMonth'] = cpi_by_year_month.index
train_df = cpi_by_year_month[cpi_by_year_month['date'] < train_end_date]
reg = linear_model.LinearRegression()
reg.fit(train_df[['YearMonth']], train_df['CPI'])
r = abs(reg.coef_[0] * train_df['YearMonth'] + reg.intercept_ - train_df['CPI'])

res = AutoReg(train_df['CPI'], lags=2).fit()
print(res.params)
del train_df["YearMonth"]
del train_df["date"]
sm.graphics.tsa.plot_pacf(train_df.values.squeeze(), lags=30)
plt.show()
sm.graphics.tsa.plot_acf(train_df.values.squeeze(), lags=30)
plt.show()
