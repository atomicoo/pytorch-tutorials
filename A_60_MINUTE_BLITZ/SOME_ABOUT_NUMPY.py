## ==================== ##
##     NUMPY快速教程     ##
## ==================== ##

## 参考资料
#  英文版：https://numpy.org/devdocs/user/quickstart.html
#  中文版：https://www.numpy.org.cn/user/quickstart.html

## 编程环境
#  OS：Win10 专业版
#  IDE：VS Code
#  Python：3.7.7 (Anaconda 5.3.0)
#  NumPy：1.18.5

__author__ = 'Atomicoo'

# %%
import numpy as np
print(np.__version__)


## @@@@@@@@@@
## 基础知识

#############
## 一个例子
# %%
a = np.arange(15).reshape(3, 5)
print(a)
print(a.shape)
print(a.ndim)   # 轴
print(a.dtype.name)
print(a.itemsize)   # 每个元素字节大小
print(a.size)
print(type(a))

#############
## 数组创建
# %%
print(np.array([2,3,4]))
print(np.array([(1.5,2,3), (4,5,6)]))
print(np.array([[1,2], [3,4]], dtype=complex))

# %%
print(np.zeros((3,4)))
print(np.ones((2,3,4), dtype=np.int16))
print(np.empty((2,3)))

# %%
print(np.arange(0, 2, 0.3))
print(np.linspace(0, 2, 9))

# %%
# See also: zeros_like, ones_like, empty_like

#############
## 打印数组
# %%
print(np.arange(12).reshape(4,3))

# %%
# np.set_printoptions(threshold=sys.maxsize)
# sys module should be imported

#############
## 基本操作
# %%
a = np.array([20,30,40,50])
b = np.arange(4)
print(a-b)
print(b**2)
print(np.sin(a))
print(a<35) # 比较

# %%
A = np.array([[1,1],
              [0,1]])
B = np.array([[2,0],
              [3,4]])
print(A*B)  # 逐元素乘
print(A@B)  # 点乘
print(A.dot(B)) # 点乘

# %%
# *=、+=
a = np.ones((2,3), dtype=int)
b = np.random.random((2,3))
a *= 3
print(a)
b += a
print(b)    # 不同类型数组间操作，结果将会向上（更精确）转换
# a += b    # 错误，结果精度不允许向下转换

# %%
a = np.ones(3, dtype=np.int32)
b = np.linspace(0, np.pi, 3)
c = a + b
d = np.exp(c*1j)    # 不同类型数组间操作，结果向上（更精确）转换
print(d)

# %%
a = np.random.random((2,3))
print(a)
print(a.sum())  # 默认对整个数组操作
print(a.min(axis=0))    # 可通过axis指定沿某轴操作
print(a.max(axis=1))

###########
## 通函数
# %%
A = np.arange(6).reshape(2,3)
B = np.array([2.,-1.,4.])
print(A)
print(B)
print(np.exp(A))
print(np.sqrt(B))
print(np.add(A, B)) # 广播机制

# %%
# See also: all, any, argmax, argsort, average, 
# ceil, clip, diff, dot, floor, invert, max, mean, 
# median, min, nonzero, prod, round, sum, transpose, where

###################
## 索引/切片/迭代
# %%
a = np.arange(10)**3
print(a)
print(a[3])
print(a[3:5])
a[:6:2] = 100
print(a)
print(a[::-1])

# %%
def func(x, y):
    return 10*x+y

a = np.fromfunction(func, (3,4), dtype=int)
print(a)
print(a[1,2])
print(a[2,1:])
print(a[-1])    # a[-1] == a[-1,:]
# 注意，缺失的索引认定为全切片

# %%
a = np.array([[[  0,  1,  2],               # a 3D array (two stacked 2D arrays)
               [ 10, 12, 13]],
              [[100,101,102],
               [110,112,113]]])
print(a)
print(a.shape)
print(a[0,...]) # b[0,...] == b[0,:,:]
print(a[...,2]) # b[...,2] == b[:,:,2]
# 注意，省略的索引认定为全切片

# %%
# See also: newaxis
a = np.arange(3)
print(a)
print(a[:, np.newaxis])


## @@@@@@@@@@
## 形状操纵

#############
## 数组变形
# %%
# 多维数组扁平化
a = np.arange(12).reshape((3,4))
print(a.flat)   # 返回迭代器
print(a.flatten())  # 返回列表
print(a.ravel())    # 返回列表
print(a.reshape((12)))  # 返回列表

# %%
# 以下操作不改变原数组
a = np.arange(12).reshape((3,4))
print(a)
print(a.ravel())
print(a.reshape((2,6)))
print(a.T)

# %%
# 以下操作会改变原数组
a = np.arange(12).reshape((3,4))
print(a)
a.resize((2,6))
print(a)

# %%
# np.reshape()
a = np.arange(12).reshape((3,4))
print(a)
print(a.reshape((12)))
print(a.reshape((1,12)))    # a.reshape((12)) != a.reshape((1,12))
print(a.reshape((1,-1)))    # np.reshape()中指定为-1的轴将自动计算其size
print(a.reshape((-1,12)))
print(a.reshape((-1,2,3)))

# %%
a = np.arange(12).reshape((3,4))
print(a)
print(a.reshape((2,6), order='C'))
print(a.reshape((2,6), order='F'))
# order参数指定索引顺序：
# 默认'C'为类C风格，最后一轴变化最快
# 'F'为类Fortran风格，第一轴变化最快

#############
## 数组堆叠
# %%
a = np.arange(4).reshape((2,2))
b = np.floor(10*np.random.random((2,2)))
print(a)
print(b)
print(np.vstack((a,b))) # 垂直堆叠（沿第一轴堆叠）
print(np.hstack((a,b))) # 水平堆叠（沿第二轴堆叠）

# %%
a = np.arange(4).reshape((2,2))
b = np.floor(10*np.random.random((4))).reshape((2,2))
print(a)
print(b)
print(np.column_stack((a,b)))
# a&b第一轴必须有相同的size

# %%
a = np.arange(4).reshape((2,2))
b = np.floor(10*np.random.random((4))).reshape((2,2))
print(a)
print(b)
print(np.row_stack((a,b)))
# a&b第二轴必须有相同的size

# %%
print(np.column_stack is np.hstack) # 输入为1-D数组时两者操作不同，见下。
print(np.row_stack is np.vstack)

# %%
a = np.arange(4)
b = np.floor(10*np.random.random((4)))
print(a)
print(b)
print(np.column_stack((a,b)))
print(np.hstack((a,b)))
# 输入为1-D数组时np.column_stack()与np.hstack()操作不同

# %%
np.r_[1:4,0,np.array([3,2,1])]
# 默认操作类似np.vstack()，但允许使用可选参数指定连接的轴

# %%
np.c_[np.array([1,2,3]), np.array([4,5,6])]
# 默认操作类似np.hstack()，但允许使用可选参数指定连接的轴

# %%
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6]])
print(np.concatenate((a, b), axis=0))
print(np.concatenate((a, b.T), axis=1))
# np.concatenate()允许通过参数axis指定拼接的轴

# %%
# See also: hstack, vstack, column_stack, concatenate, c_, r_

#############
## 数组拆分
# %%
a = np.floor(10*np.random.random((2,12)))
print(a)
for e in np.hsplit(a,3):    # 平均拆分
    print(e)
for e in np.hsplit(a,(3,5)):    # 指定拆分位置
    print(e)

# %%
# See also: vsplit, array_split


## @@@@@@@@@@@@
## 拷贝和视图

###############
## 完全不拷贝
# %%
# 不拷贝，同一数组对象 np.ndarray
a = np.array([[ 0,  1,  2,  3],
              [ 4,  5,  6,  7],
              [ 8,  9, 10, 11]])
b = a
print(b is a)

# %%
def func(x):
    return x

print(id(a))
print(id(func(a)))

#################
## 视图或浅拷贝
# %%
# 浅拷贝，共享相同数据，但为不同数组对象 np.ndarray
a = np.array([[ 0,  1,  2,  3],
              [ 4,  5,  6,  7],
              [ 8,  9, 10, 11]])
b = a.view()
print(b is a)
print(b.base is a)
print(b.flags.owndata)

# %%
c = b.reshape((2,6))
print(a.shape)  # a的形状不变
c[1, 3] = 999
print(a)    # a的数据改变

# %%
# 切片数据会返回一个视图（浅拷贝）
s = a[:, 1:3]
print(s.base is a)
s[:] = 11
print(a)

###########
## 深拷贝
# %%
# 深拷贝，生成一份完整的数组对象及数据的副本
a = np.array([[ 0,  1,  2,  3],
              [ 4,  5,  6,  7],
              [ 8,  9, 10, 11]])
b = a.copy()
print(b is a)
print(b.base is a)

# %%
b[2, 3] = 999
print(a)    # a的数据不变

# %%
c = a[:, 1:3].copy()
del a   # 完全释放a占用的资源
print(c)    # c仍然存在

# %%
