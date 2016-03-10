---
layout: post
title: "C#调用Python脚本并使用Python的第三方模块"
date: 2015-01-19 21:45:31 +0800
comments: true
categories: [dotnet,python,程序设计]
---

#前言
InronPython是一种在.NET和Mono上实现的Python语言，使用InronPython就可以在.NET环境中调用Python代码，非常方便。

本文主要介绍在C#中调用Python代码，并使用Python安装的第三方模块。

#安装InronPython
要在.NET环境中使用Python，先要安装InronPython（当然也要安装Python），安装很简单，直接下载安装就行。在CodePlex就有下载，下载地址：

*  [http://ironpython.net/](http://ironpython.net/)
*  [https://ironpython.codeplex.com/](https://ironpython.codeplex.com/)

还可以把[Python Tools for Visual Studio](http://pytools.codeplex.com/)也安装了。

<!--more-->

#使用
##添加引用库
在Visual Studio新建一个工程后，添加引用IronPython.dll和Microsoft.Scripting.dll（位于InronPython的安装目录下，如下图）。

![InronPython dll](/assets/images/2015/ironpython1.png)

##C#代码内嵌Python
最简单的使用方式如下：

```c#
var engine = IronPython.Hosting.Python.CreateEngine();
engine.CreateScriptSourceFromString("print 'hello world!'").Execute();
```

##从文件中加载Python代码
一般情况下我们还是要把Python代码单独写在文件中。在工程中新建一个Python文件，如`hello.py`，并设置其属性Copy to Output Directory的值为Copy if newer。在hello.py下编写如下代码：

```python
def say_hello():
    print "hello!"

def get_text():
    return "text from hello.py"

def add(arg1, arg2):
    return arg1 + arg2
```

C#代码如下：

```c#
var engine = IronPython.Hosting.Python.CreateEngine();
var scope = engine.CreateScope();
var source = engine.CreateScriptSourceFromFile("hello.py");
source.Execute(scope);

var say_hello = scope.GetVariable<Func<object>>("say_hello");
say_hello();

var get_text = scope.GetVariable<Func<object>>("get_text");
var text = get_text().ToString();
Console.WriteLine(text);

var add = scope.GetVariable<Func<object, object, object>>("add");
var result1 = add(1, 2);
Console.WriteLine(result1);

var result2 = add("hello ", "world");
Console.WriteLine(result2);
```

更详细的使用方法可参考[文档](http://ironpython.net/documentation/)和[代码例子](http://ironpython.net/documentation/)。

#使用Python安装的第三模块
##问题
到此为止，程序运行得很顺利。可是好景不长，最近用Python写了个程序要使用rsa加密，在Python中安装了rsa模块（下载地址：[https://pypi.python.org/pypi/rsa/3.1.1](https://pypi.python.org/pypi/rsa/3.1.1)）后，直接运行Python代码没问题，可是在C#代码调用时就报异常，异常信息如下：

```
An unhandled exception of type 'IronPython.Runtime.Exceptions.ImportException' occurred in Microsoft.Dynamic.dll
Additional information: No module named rsa
```

没有找到模块，经过一番google，说是要设置`sys.path`，如下：

```python
import sys
sys.path.append(r"c:\python27\lib")
```

照做之后问题依旧。不过想一想，应该是sys.path还没设置对。

##解决
先在python代码加上下面几行：

```python
import sys
sys.path.append(r"c:\python27\lib")
print sys.path
```

运行查看输出，在对比Python环境下的`sys.path`，果然不一样，问题应该就出在`sys.path`上了。

在cmd下分别打开python和IronPython(在IronPython安装目录下的ipy64.exe或ipy.exe)，执行`import sys;print sys.path`，对比输出：

python:

```
['', 'C:\\Python27\\lib\\site-packages\\setuptools-12.0.3-py2.7.egg', 'C:\\Pytho
n27\\lib\\site-packages\\rsa-3.1.1-py2.7.egg', 'C:\\Python27\\lib\\site-packages
\\pyasn1-0.1.7-py2.7.egg', 'C:\\Windows\\SYSTEM32\\python27.zip', 'C:\\Python27\
\DLLs', 'C:\\Python27\\lib', 'C:\\Python27\\lib\\plat-win', 'C:\\Python27\\lib\\
lib-tk', 'C:\\Python27', 'C:\\Python27\\lib\\site-packages']
```

IronPython:

```
['.', 'C:\\Program Files (x86)\\IronPython 2.7\\Lib', 'C:\\Program Files (x86)\\
IronPython 2.7\\DLLs', 'C:\\Program Files (x86)\\IronPython 2.7', 'C:\\Program F
iles (x86)\\IronPython 2.7\\lib\\site-packages']
```

分别在两个环境在执行`import rsa`，Python环境正常，IronPython环境下报ImportError: No module named rsa异常。在IronPython环境下执行如下操作：

```python
sys.path.append('C:\\Python27\\lib\\site-packages\\setuptools-12.0.3-py2.7.egg')
sys.path.append('C:\\Python27\\lib\\site-packages\\rsa-3.1.1-py2.7.egg')
```

再次`import rsa`，不报异常了。

对应的python代码文件加上如下几行：

```python
import sys
sys.path.append('C:\\Python27\\lib')
sys.path.append('C:\\Python27\\lib\\site-packages\\setuptools-12.0.3-py2.7.egg')
sys.path.append('C:\\Python27\\lib\\site-packages\\rsa-3.1.1-py2.7.egg')
import rsa 
```

再次运行，Ok！