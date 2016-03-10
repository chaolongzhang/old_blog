---
layout: post
category: [python,程序设计,算法]
title: 用Python做数学计算之基础计算
keywords: Python数学计算
tags: [Python数学计算, math]
---

##摘要（Abstract）
本文介绍使用Python来做数学计算，在学习和工作中，经常会遇到一些数学计算的问题。一般人会使用计算器软件，不得不说，计算器太难用了。专业人士可能会使用更强大的工具，如Matlab，但这种重量级工具有时可能并不适用。本文打算使用一个轻量级的工具Python来做计算。准确来说Python并不是一个数学工具，而是一种编程语言。Python提供了很多数学库，利用Python和这些库可以做很多数学计算。

本文不是编程语言的教程，更像是一个工具的使用教程，阅读本文不需要有程序设计基础，当然，需要一点数学基础（比如加减乘除）。本文适合任何想找一个计算工具的人学习和参考。

本文将以实例讲解各种用法。

<!--more-->

##安装Python（Installation）
Python[官方网站](https://www.python.org/)提供下载，完全免费使用。Python目前有Python 2和Python 3两个版本，两个版本有一些语法差别，对于本文来说，几乎没有区别，推荐使用Python 3。在[Download](https://www.python.org/downloads/)界面找到系统对应的版本下载（我的版本是`Mac OS X 64-bit/32-bit installer`），双击直接安装。安装好后，找到程序`IDLE`，启动`IDLE`就可开始写Python程序了。

![Python Shell](/assets/images/2015/python-math-idle.png)

*  提示1：Mac OS和大部分版本Linux系统自带Python运行环境，可以不用安装。当然，也可升级成最新版本。Windows需要自行安装。
*  提示2：也可以安装Sublime Text编辑器，再安装Sublime REPL插件。本人现在使用这种方案，写Python程序非常方便。

![REPL](/assets/images/2015/python-math-sublime.png)

*  提示3：搜狗输入法用户注意，搜狗输入法在`IDLE`下有点小问题（不能退格），切换到英文输入法即可。

##Python 2 和Python 3的注意事项
*  **`print`的语法**。python 3的用法为`print("hello world!")`，python 2的用法为`print "hello world!"`或者`print("hello world!")`。

##基本运算
###加法

```python
>>> 1 +  2		//直接输入，回车直接输出结果
3
>>> sum = 1 + 2	//计算结果保存在sum中
>>> print(sum)		//输出sum
3
>>> a = 1		//变量
>>> b = 2
>>> sum = a + b 	//变量相加
>>> print(sum)
3
```

###减法

```python
>>> a = 1
>>> b = 2
>>> 2 - 1
1
>>> a - b
-1
>>> b - a
1
```

###乘法

```python
>>> 1 * 2
2
>>> 1.5 * 3
4.5
>>> a * b
2
>>> 
```

###除法

**传统的除法**。有编程经验的人一定知道，整型除法的结果也是整型（地板除，舍弃小数部分）。如果有一个数是浮点型，则结果是浮点型（真正的除法，保留小数）。

```python
>>> 1 / 3			//整型除法，商为0，舍弃小数部分
0	
>>> 5 / 2
2	
>>> 1.0 / 3			//浮点数除法，保留小数
0.3333333333333333
>>> 5.0 / 2.0
2.5
```

**真正的除法**。在未来的Python版本中，不管是整型还是浮点型除法，结果总是真实地商（保留小数部分），现阶段可以通过执行`from __future__ import division`指令做到这一点。

```python
>>> from __future__ import division
>>> 1 / 3
0.3333333333333333
>>> 5 / 2
2.5
>>> 1.0 / 3
0.3333333333333333
>>> 5.0 / 2.0
2.5
```

**地板除**。Python 2.2新加了一个操作符`//`，`//`除法不管操作数为何种数值类型，总是舍弃小数部分。

```python
>>> 1 // 3
0
>>> 5 // 2
2
>>> 1.0 // 3
0.0
>>> 5.0 // 2.0
2.0
```

###取余数

```python
>>> 1 % 3
1
>>> 5 % 2
1
>>> 5.0 % 2.0
1.0
```

###幂运算
Python有幂运算符`**`。

```python
>>> 2 ** 3
8
>>> 2.5 ** 5
97.65625
>>> 4 ** -1
0.25
```

代码中分别计算2<sup>3</sup>、2.5<sup>5</sup>、4<sup>-1</sup>。

###复数
**复数的表示**。复数的表示如下：

```python
>>> aComplex = 1 + 2j		//申明一个复数
>>> aComplex
(1+2j)
>>> aComplex.real 			//复数实部
1.0
>>> aComplex.imag 			//复数虚部
2.0
>>> aComplex.conjugate()		//共轭复数
(1-2j)
>>> 
```

**复数的运算**。复数的运算与实数一致。

```python
>>> c = 1 + 2j
>>> d = 2 - 1j
>>> c + d
(3+1j)
>>> c - d
(-1+3j)
>>> c * d
(4+3j)
>>> c / d
1j
>>> c / 2
(0.5+1j)
>>> c * 2
(2+4j)
>>> c ** 2
(-3+4j)
```

##math标准库
Python有一个标准库`math`专门用来做数学运算的。详细介绍可参考Python的[官方文档](https://docs.python.org/3.4/library/math.html)。要使用`math`库，先要`import`这个库。

```python
>>> import math
```

一下的例子假设已经执行了`import math`。

###两个常数

```python
>>> math.pi 			//圆周率pi
3.141592653589793
>>> math.e
2.718281828459045		//自然常数e
```

###数值计算
*  **math.ceil(x)**。向上取整，返回最小的大于或等于x的整数。

```python
>>> math.ceil(2)
2.0
>>> math.ceil(2.2)
3.0
>>> math.ceil(2.9)
3.0
>>> math.ceil(3.0)
3.0
```

*  **math.floor(x)**。向下取整，返回最大的小于或等于x的整数。

```python
>>> math.floor(2)
2.0
>>> math.floor(2.2)
2.0
>>> math.floor(2.9)
2.0
>>> math.floor(3.0)
3.0
```

*  **math.fabs(x)**。取x得绝对值。

```python
>>> math.fabs(1.0)
1.0
>>> math.fabs(-1.0)
1.0
```

*  **math.factorial(x)**。求x的阶乘，x必须为整数，否则出现错误。

```python
>>> math.factorial(5)
120
>>> math.factorial(4)
24
>>> math.factorial(2.1)		//执行错误
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
ValueError: factorial() only accepts integral values
>>> 
```

###幂和对数函数（Power and logarithmic functions）
*  **math.exp(x)**。返回`e ** x`。

```python
>>> math.exp(2)
7.38905609893065
>>> math.e ** 2
7.3890560989306495	//请忽略后面的不一致，计算机浮点数本身的问题
```


*  **math.log(x [,base])**。求以base为底的对数。

```python
>>> math.log(math.e)		//值传一个参数，默认以math.e为底
1.0
>>> math.log(math.e ** 2)
2.0
>>> math.log(8, 2)		//两个参数，2为底
3.0
>>> math.log(100, 10)	//两个参数，10为底s
2.0
```

*  **math.pow(x, y)**。幂运算，计算x<sup>y</sup>，相当于`x ** y`。

```python
>>> math.pow(2, 3)
8.0
>>> 2 ** 3
8
```

*  math.sqrt(x)**。求x的平方根。

```python
>>> math.sqrt(4)
2.0
>>> math.sqrt(9.0)
3.0
```

*  **开根号**。Python的`math`库中只有开平方根，没有立方根和n次方根，不过可以利用`math.pow`或者`**`，只需把根号变成分数。

```python
>>> math.pow(4, 1.0 / 2)		//平方根，相当于math.sqrt(4)
2.0
>>> 4 ** (1.0 / 2)			//平方根，相当于math.sqrt(4)
2.0
>>> 8 ** (1.0 / 3)			//立方根
2.0
>>> 1024 ** (1.0 / 10)			//10次方根
2.0
```

###三角函数（Trigonometric functions）
`math`库中的三角函数使用的弧度（radians）计算，所以用角度(angle)计算前先要转换成弧度。`math`也提供了角度和弧度的转换函数。

| 方法名    | 实现功能 |
| :-----------| :------------|
| math.degrees(x)     | 把弧度x转换成角度 |
| math.radians(x)      | 把角度x转换成弧度 |

```python
>>> math.degrees(math.pi)
180.0
>>> math.radians(180.0)
3.141592653589793
```

*  **math.cos(x)**。不解释。
*  **math.sin(x)**。不解释。
*  **math.tan(x)**。不解释。
*  **math.acos(x)**。反余弦函数。
*  **math.asin(x)**。反正弦函数。
*  **math.atan(x)**。反正切函数。
*  **math.hypot(x，y)**。欧式范数（Euclidean norm）。相当`sqrt(x*x + y*y)`。

```python
>>> math.cos(math.pi)
-1.0
>>> math.sin(math.pi / 2)
1.0
>>> math.tan(math.pi / 4)
0.9999999999999999		//结果为1，计算机浮点数固有缺陷
>>> math.tan(math.pi / 4)
0.9999999999999999
>>> math.acos(-1.0)
3.141592653589793
>>> math.asin(1.0)
1.5707963267948966
>>> math.atan(1)
0.7853981633974483
>>> math.pi / 4
0.7853981633974483
>>> math.hypot(3, 4)
5.0
```

###双曲函数（Hyperbolic functions）
双曲函数和三角函数类似。

*  **math.cosh(x)**。求x的双曲余弦函数。
*  **math.sinh(x)**。求x的双曲正弦函数。
*  **math.tanh(x)**。求x的双曲正切函数。
*  **math.acosh(x)**。求x的反双曲余弦。
*  **math.asinh(x)**。求x的反双曲正弦。
*  **math.atanh(x)**。求x的反双曲正切。

```python
>>> math.cosh(1)
1.5430806348152437
>>> math.sinh(1)
1.1752011936438014
>>> math.tanh(1)
0.7615941559557649
>>> math.acosh(1.5430806348152437)
1.0
>>> math.asinh(1.1752011936438014)
1.0
>>> math.atanh(0.7615941559557649)
0.9999999999999999			//结果为1，计算机浮点数固有缺陷
```

##一个复杂的例子
举一个复杂的例子，计算下面的公式：

![复杂公式](/assets/images/2015/python-math-b.jpg)

```python
>>> - (4 ** (1.0 / 3)) / 4 * math.log(2 ** (1.0 / 3) - 1) - (math.sqrt(3) * (4 ** (1.0 / 3))) / 6 * math.atan(math.sqrt(3) / (1 + 2 * (2 * (1.0 / 3))))
0.24209140862733897
```

##坑
在前面的例子可能已经注意到了，计算机表示浮点数是不准确的，比如`1.0`成了`0.9999999999999999`。这是计算机浮点数表示法的固有缺陷。比如会出现如下结果：

```python
>>> 0.05 + 0.01
0.060000000000000005
>>> 1.0 - 0.12
0.88
>>> 1.0 - 0.32
0.6799999999999999
>>> 1.0 - 0.42
0.5800000000000001
>>> 4.015 * 100
401.49999999999994
>>> 123.3 / 100
1.2329999999999999
```

浮点数的缺陷不仅仅在Python在存在，在任何语言都存在，它是浮点数表示法本身的缺陷。对于不是非常严格的计算，这种缺陷是可以接受的，而对于过于严格的计算，那么应该用整型数计算（整型计算是精确的）。比如在保留两位小数的计算中，可以把原始数乘以100取整数再计算，约定最后两位为小数即可。

##结语
本文介绍了Python标准库的数学计算方法，通过使用Python标准数学库，可以应付一些日常的基本计算（比如我就很少使用计算器了，一般的计算都用Python解决）。当然，数学是如此的丰富，仅有标准库是远远不够的。将来如果有需要的话，我会研究下使用其它的数学库来进行更复杂的计算以及数学图像的绘制。比如[NumPy](http://www.numpy.org/)、[SciPy](http://www.scipy.org/)、[Matplotlib](http://matplotlib.org/)、[SymPy](http://www.sympy.org/)等。

