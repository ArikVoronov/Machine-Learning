from statistics import mean
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import style
import random
style.use('fivethirtyeight')
'''
Simple Linear Reggression
In statistics, simple linear regression is a linear regression model with
a single explanatory variable. That is, it concerns two-dimensional sample
points with one independent variable and one dependent variable and finds a
inear function (a non-vertical straight line) that, as accurately as possible,
predicts the dependent variable values as a function of the independent variables.
The adjective simple refers to the fact that the outcome variable is related to a single predictor.
'''
# Create dataset, potentially linearly correlated
def create_dataset(hm, variance, step = 2, correlation = False):
    val = 1
    ys = []
    for i in range(hm):
        y = val + random.randrange(-variance, variance)
        ys.append(y)
        if correlation and correlation == 'pos':
            val += step
        elif correlation and correlation == 'neg':
            val -= step
    xs = [i for i in range(len(ys))]
    return np.array(xs, dtype = np.float64), np.array(ys, dtype = np.float64)


'''
The linear approximator formula:
y = m*x + n

The formulas for parameters m and n
are derived from the minimization of squared error:
yi = m*xi + n + ei
->
ei = yi - m*xi  - n

m,n = argmin(sum(ei(m,n)**2))

'''
def best_fit_slope(xs,ys):
    m =( ( mean(xs)*mean(ys) - mean(xs*ys) )/
    ( mean(xs)**2 - mean(xs**2) ))
    return m
def best_fit_intercept(xs,ys,m):
    n = mean(ys) - m * mean(xs)
    return n


def squared_error(y1,y2):
    return sum((y2-y1)**2)



# Quality of approximation
def coefficient_of_determination(ys_orig,ys_line):
    y_mean_line = [mean(ys_orig) for _ in ys_orig]
    squared_error_regr = squared_error(ys_orig, ys_line)
    # Residual error is normalized with "error" regarding a simple horizontal mean line (of the original data)
    squared_error_y_mean = squared_error(ys_orig,y_mean_line)
    return ( 1 - squared_error_regr / squared_error_y_mean )


xs, ys = create_dataset(40,80,3, 'pos')
m = best_fit_slope(xs,ys)
n = best_fit_intercept(xs,ys,m)                        

ys_est = m*xs + n
r_squared = coefficient_of_determination(ys, ys_est)

plt.scatter(xs,ys, label = 'raw', color = 'red')
plt.plot(xs,ys_est,label = 'est', color = [0,1,1])
plt.legend(loc = 4)
plt.title(['m = ',m,'n = ',n,'r2 = ', r_squared])
plt.show()
