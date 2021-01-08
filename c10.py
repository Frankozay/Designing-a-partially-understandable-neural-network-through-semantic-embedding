import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy import interpolate
from matplotlib.pyplot import MultipleLocator


def cal(arr, l):
    a = []
    b = []
    l1 = int(len(arr) / l)
    for i in range(l1):
        a.append(np.mean(arr[i * l:i * l + l]))
        # a.append(arr[int((i*l+i*l+l)/2)])
        b.append((i * l) / 10000.0)
    return b, a


def cal2(x, y, l):
    a = []
    b = []
    l1 = int(len(x) / l)  # 个数
    for i in range(l1):
        a.append(x[i * l])
        b.append(np.mean(y[i * l:i * l + l]))
    return a, b


loss_fixed = []
loss_unfixed = []
pr_fixed = []
pr_unfixed = []

for e in tf.train.summary_iterator('./log/plot_fixed/events.out.tfevents.1575723670.unbuntu-HP-Z4-G4-Workstation'):
    for v in e.summary.value:
        if v.tag == 'test_loss':
            loss_fixed.append(v.simple_value)
        if v.tag == 'pr':
            pr_fixed.append(v.simple_value * 100)

for e in tf.train.summary_iterator('./log/plot_unfixed/events.out.tfevents.1575723678.unbuntu-HP-Z4-G4-Workstation'):
    for v in e.summary.value:
        if v.tag == 'test_loss':
            loss_unfixed.append(v.simple_value)
        if v.tag == 'pr':
            pr_unfixed.append(v.simple_value * 100)

test_x = np.array(list(range(0,150000,75)))
test_x = [t/10000.0 for t in test_x]

pfx, pfy = cal2(test_x, pr_fixed, 2)
pufx, pufy = cal2(test_x, pr_unfixed, 2)

fig1 = plt.figure(num=1)
ax1 = fig1.add_subplot(111)

ax1.plot(pfx[::6],pfy[::6],linewidth=1.2,color='firebrick',linestyle="-")
ax1.plot(pufx[::6],pufy[::6],linewidth=1.2,color='darkgoldenrod',linestyle="-")

plt.xlim((0,15))
plt.ylim((20, 100))
# y = range(0, 6, 1)
# plt.yticks(y)
plt.xlabel('iter(1e-4)')
plt.ylabel('pr(%)')
plt.title('densenet-cifar10')
plt.legend()
x_major_locator=MultipleLocator(1)
y_major_locator=MultipleLocator(10)
ax1.yaxis.set_major_locator(y_major_locator)
ax1.xaxis.set_major_locator(x_major_locator)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

plt.show()

