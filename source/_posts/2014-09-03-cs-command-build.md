---
layout: post
categories: 程序设计
title: 用命令行编译C#代码
tags: [C#]
---


最近要写一些插件，这些插件代码类似，人太懒，不想重复写代码，所以决定写个脚本来生成C#代码，最好是能直接编译。

\.net下使用csc命令可以编译代码，MSDN文档：http://msdn.microsoft.com/zh-cn/library/78f4aasd.aspx

<!--more-->

要使我的脚本能调用csc命令，还需要设置环境变量，在C:\Windows\Microsoft.NET\Framework可以找到本机安装的所有.NET Framework版本，把要使用的版本的路径加入环境变量就行了。

关于CSC的用法MSDN和网上有很多，一些常用的有：

编译File.cs以产生File.exe:

```
	csc File.c  
```

编译File.cs以产生File.dll(/target也可简写成/t):

```
	csc /target:library File.cs  
```

我的代码引用了两个DLL文件，直接用上面的命令编译出错，需要添加引用，使用命令/reference或/r，如：

```
	csc /target:library /reference:MyDll1.dll /r:MyDll2.dll File.cs  
```