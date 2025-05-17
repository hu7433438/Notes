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
Step
1: Understanding
Rank
of
Matrix
Sums and Products
To
determine
the
rank
of
the
sum and product
of
two
rank - 1
matrices, we
need
to
understand
the
properties
of
matrix
addition and multiplication.

Sum
of
Matrices: The
rank
of
the
sum
of
two
matrices is generally
less
than or equal
to
the
sum
of
their
ranks.For
rank - 1
matrices, this
means
the
rank
of
the
sum
can
be
at
most
2.
Product
of
Matrices: The
rank
of
the
product
of
two
matrices is generally
less
than or equal
to
the
minimum
of
their
ranks.For
rank - 1
matrices, this
means
the
rank
of
the
product
can
be
at
most
1.
Step
2: Analyzing
Given
Matrices
Given
matrices:
A = (
    −1
    −3
    ​

    1
    3
    ​
), B = (
    1
    −1
    ​

    −1
    1
    ​
), C = (
    0
    0
    ​

    0
    1
    ​
), D = (
    1
    1
    ​

    1
    1
    ​
)

All
these
matrices
are
rank - 1.

Step
3: Determining
Rank
of
Sums
We
need
to
check
the
rank
of
the
following
sums:

A + A
A + B
A + C
** A + A **:
A + A = (
            −1
            −3
            ​

            1
            3
            ​
        ) + (
            −1
            −3
            ​

            1
            3
            ​
        ) = (
    −2
    −6
    ​

    2
    6
    ​
)

This
matrix is still
rank - 1
because
it is a
scalar
multiple
of
A.

** A + B **:
A + B = (
            −1
            −3
            ​

            1
            3
            ​
        ) + (
            1
            −1
            ​

            −1
            1
            ​
        ) = (
    0
    −4
    ​

    0
    4
    ​
)

This
matrix is rank - 1
because
it is a
scalar
multiple
of(
    0
−1
​

0
1
​
).

** A + C **:
A + C = (
            −1
            −3
            ​

            1
            3
            ​
        ) + (
            0
            0
            ​

            0
            1
            ​
        ) = (
    −1
    −3
    ​

    1
    4
    ​
)

This
matrix is rank - 2
because
it
has
two
linearly
independent
rows.

Step
4: Determining
Rank
of
Products
We
need
to
check
the
rank
of
the
following
products:

AB
AC
BD
** AB **:
AB = (
    −1
    −3
    ​

    1
    3
    ​
)(
    1
−1
​

−1
1
​
)=(
    0
    0
    ​

    0
    0
    ​
)

This
matrix is rank - 0.

** AC **:
AC = (
    −1
    −3
    ​

    1
    3
    ​
)(
    0
0
​

0
1
​
)=(
    0
    0
    ​

    1
    3
    ​
)

This
matrix is rank - 1
because
it is a
scalar
multiple
of(
    0
0
​

1
3
​
).

** BD **:
BD = (
    1
    −1
    ​

    −1
    1
    ​
)(
    1
1
​

1
1
​
)=(
    0
    0
    ​

    0
    0
    ​
)

This
matrix is rank - 0.

Final
Answer
Rank - 2
Combinations: A + C
Rank - 1
Combinations: A + A, A + B, AC
iconSee
Full
Answer
Answered
Sep
22
at
06: 44(Basic
Model)