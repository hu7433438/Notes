#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 6/2/2024 9:35 PM
# @Author  : QingYang H.
# @File    : main.py

import numpy as np
# 13
# a = np.array([[1, 2, 3], [4, 5, 6], [1, 2, 1]])
a = np.array([[3, 0], [1/2, 2]])
print(np.linalg.det(a))
print(np.linalg.det(a.transpose()))
# 7
import scipy.stats as stats
b = stats.norm(1, 2 ** 0.5)
# x_less_than_8 = stats.norm.cdf(8, loc=1, scale=2)
print(b.cdf(1), b.cdf(0.5), b.cdf(2), b.cdf(2) - b.cdf(0.5))
