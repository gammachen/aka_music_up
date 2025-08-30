from sympy import *

x = Symbol('x')
# f(x)=-10x**2+100*x+5
y = -10 * x ** 2 + 100 * x  + 5
yprime = y.diff(x)
print(yprime)

# 一阶求导
dict1 = solve([yprime])
print(dict1)

x1 = dict1[x]
print(x1)

print(f'x={x1},最大值={-10 * x1 ** 2 + 100 * x1  + 5}')

'''
100 - 20*x
{x: 5}
5
x=5,最大值=255
'''