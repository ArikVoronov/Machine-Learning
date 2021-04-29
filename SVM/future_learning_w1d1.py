import numpy as np

def total_time(x,y, v):
    t = 0
    for i in range(len(y)):
        if i>0:
            dx = x[i]-x[i-1]
        else:
            dx = x[0]
        d = np.sqrt(dx**2 + y[i] **2)
        t += d/v[i]
    return t



x0v = np.arange(0,150)
tv = []
for x0 in x0v:
    y = [30,50]
    x = [x0,100]
    v = [8,3]
    tv.append(total_time(x,y,v))

import matplotlib.pyplot as plt
plt.figure()
plt.plot(x0v,tv)
plt.show()