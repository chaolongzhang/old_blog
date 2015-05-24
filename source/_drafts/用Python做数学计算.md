---
layout: post
category: [程序设计]
title: 用Python做数学计算
keywords: Python数学计算
tags: [Python,数学计算]
---

##摘要（Abstract）
本文介绍使用Python来做数学计算，在学习和工作中，经常会遇到一些数学计算的问题。普通人会使用计算器软件，不得不说，计算器太难用了。专业人士可能会使用更强大的工具，如Matlab，但这种重量级工具有时可能并不适用。本文打算使用一个轻量级的工具Python来做计算。其实Python并不是一个数学工具，而是一种编程语言。Python提供了很多数学库，Python和这些库可以胜任很多数学计算。

本文不是编程语言的教程，更像是一个工具的使用教程，阅读本文不需要有程序设计基础，当然，需要一点数学基础（比如加减乘除）。本文适合任何想找一个计算工具的人学习和参考。

本文将以实例讲解各种用法。

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

```
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


