import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import statsmodels.api as sm
import pylab as py
# Generate example data
# data = np.array([0.01, 0.1, 0.2, 0.28, 0.80])
# # data = np.round(np.random.normal(0,1,1000),2)
# data = np.round(np.random.uniform(-2,1,1000),2)
# # data = np.round(np.random.exponential(2.5,40),2)
# # data = np.round(np.random.laplace(0, 2**1/2,40),2)
# # data = np.round(np.random.standard_cauchy(100),2)
# sm.qqplot(data, line ='45')
# # sm.qqplot(data, line ='45',dist=stats.uniform)
# py.show()
# # Perform the K-S test
# ks_statistic, p_value = stats.kstest(data, 'uniform')
#
# print('K-S Statistic:', ks_statistic)
# print('p-value:', p_value)
#
# ks_statistic, p_value = stats.kstest(data, 'norm')
#
# print('K-S Statistic:', ks_statistic)
# print('p-value:', p_value)
#
# # Generate some random data
# # data = np.random.normal(loc=0, scale=1, size=100)
#
# # Perform the Lilliefors test
# ks_statistic, p_value = sm.stats.lilliefors(data)
#
# print("Kolmogorov-Smirnov test statistic:", ks_statistic)
# print("p-value:", p_value)
# Create Q-Q plot
# stats.probplot(data, dist="norm", plot=plt)
# plt.plot(data, data, color = 'red', label = 'x=y')
# plt.title('Normal Q-Q plot')
# plt.xlabel('Theoretical quantiles')
# plt.ylabel('Ordered Values')
# plt.grid(True)
# plt.show()

#
# import numpy as np
# import pylab
# import scipy.stats as stats
#
# measurements = np.array([0.28, 0.2, 0.01, 0.80, 0.1])
# stats.probplot(measurements, dist="norm", plot=pylab)
# pylab.show()
# from scipy.special import chdtri
# print(chdtri(10.823064182, 1 - 0.05))
# from scipy.stats import t
# print(t.ppf(1-0.001/2, 120))
# a=np.array([[ 1, 1, 1], [ 1, 2, 4], [ 1, 3, 9], [ 1, 4, 16], [ 1, 5, 25], [ 1, 6, 36], [ 1, 7, 49], [ 1, 8, 64], [ 1, 9, 81], [ 1, 10, 100]])
# y = np.array([1,3,5,8,11,14,18,21,25,28]).reshape(-1,1)
# # print(a.T.dot(y))
# print(np.linalg.inv(a.transpose().dot(a)).dot(a.T.dot(y)))
Wald = 33*(1/42.6-0.03)**2*42.6**2
Likelihood = 2*(33*np.log(1/42.6)-1/42.6*42.6*33-33*np.log(0.03)+0.03*42.6*33)
print(Likelihood,Wald)
# chi-square
print(stats.chi2.sf(Likelihood , 1))
print(stats.chi2.sf(Wald , 1))
print(stats.chi2.sf(4.04 , 1))