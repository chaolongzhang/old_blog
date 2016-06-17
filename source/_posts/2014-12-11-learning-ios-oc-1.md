---
layout: post
categories: iOS开发入门教程
title: iOS开发入门教程之Objective-C · 面向过程的Objective-C
keywords: iOS
tags: [iOS]
---

上一篇：[楔子](http://zh.5long.me/2014/learning-ios-preface/)

## 前言
Objective-C语言是基于C语言的一种面向对象的扩展语言。Objective-C兼容C语言，也就是说，以前写的C代码在Objective-C平台下也基本可以运行，C语言用的熟练的话，Objective-C也基本了。

本文将以面向过程的形式来使用Objective-C，是读者从C语言过度到Objective-C。

<!--more-->

本系列教程内容如下：

*  [楔子](http://zh.5long.me/2014/learning-ios-preface/)
*  [面向过程的Objective-c](http://zh.5long.me/2014/learning-ios-oc-1/)
*  [面向对象的Objective-c](http://zh.5long.me/2014/learning-ios-oc-2/)
*  [常用的数据类型](http://zh.5long.me/2015/learning-ios-oc-3/)
*  [引用计数](http://zh.5long.me/2015/learning-ios-oc-4/)
*  [协议](http://zh.5long.me/2015/learning-ios-oc-5/)
*  [OS应用](http://zh.5long.me/2015/ios-first-app/)
* [跋](http://zh.5long.me/2015/ios-epilogue/)

## 新建工程
*  启动XCode。
*  菜单`File->New->Project`选择新建一个工程。
*  选择Command Line Tool，如下图：

![Objective-C](/assets/images/2014/oc-01.jpg)

*  点击`Next`，设置工程信息，`Language`选择`Objective-C`，如下图：

![Project](/assets/images/2014/oc-02.jpg)

*  下一步，选择保存工程的文件夹，确定即可创建一个新工程。
*  打开工程下的`main.c`源文件，代码如下

```objective-c
#import <Foundation/Foundation.h>

int main(int argc, const char * argv[])
{
    @autoreleasepool {
        // insert code here...
        NSLog(@"Hello, World!");
    }
    return 0;
}
```
*  运行一下，会有如下输出：

![Run](/assets/images/2014/oc-03.jpg)

## 源文件解析
删除`main`函数的代码，剩余代码如下：

```objective-c
#import <Foundation/Foundation.h>

int main(int argc, const char * argv[])
{
    return 0;
}
```

上面的代码是否有似曾相识的感觉，没错，和C语言基本一样。不过第一行似乎不太一样：

```objective-c
#import <Foundation/Foundation.h>
```

这一句是引用Objective-C的头文件，Objective-C使用`#import`来引用头文件，和C语言的`#include`类似，不过也不完全类似，使用`#import`命令可以确保头文件不会被重复引用，而`#include`无此功能。

`main`函数就不用再解释了，和C语言完全一样。

#Objective-C中的C
##输出
在`main`函数中添加一句代码，如下：

```objective-c
int main(int argc, const char * argv[])
{
    printf("%s\n", "Hello, World!");
    return 0;
}
```

运行，输出Hello, World!。

### 基本数据类型
C语言的基本数据类型在Objective-C下也适用，如`int`、`float`、`double`、`char`、`long`这些类型。C语言的运算在Objective-C上也适用。

### 选择结构
#### if语句
`if`的用法和C语言的用法一样，直接以说明，写一个判断输入的数是否为奇偶数：

```objective-c
#import <Foundation/Foundation.h>
int main(int argc, const char * argv[])
{
    int num = 0;
    scanf("%d", &num);
    if (num % 2 == 0)
    {
        printf("偶数");
    }
    else
    {
        printf("奇数");
    }
    printf("\n");
    return 0;
}
```

#### switch语句
使用`switch`语句重写上个例子：

```objective-c
#import <Foundation/Foundation.h>
int main(int argc, const char * argv[])
{
    int num = 0;
    scanf("%d", &num);
    int mod = num % 2;
    switch (mod)
    {
        case 0:
            printf("偶数");
            break;
        case 1:
            printf("奇数");
            break;
        default:
            break;
    }
    printf("\n");
    return 0;
}
```

#### Boolean变量
Boolean变量是Objective-C的一种基本变量，类型为BOOL，BOOL类型只有两个取值：真(`YES`)和假（`NO`），测试规则如下：
`1 == 1` 结果为`YES`， `1 == 0`结果为`NO`，加入Boolean变量重写上个例子：

```objective-c
#import <Foundation/Foundation.h>
int main(int argc, const char * argv[])
{
    int num = 0;
    scanf("%d", &num);
    BOOL isOdd = (num % 2 == 1);
    if (isOdd)
    {
        printf("奇数");
    }
    else
    {
        printf("偶数");
    }
    return 0;
}
```

### 循环结构
Objective-C的循环结构与C语言类似，也有`for`、`while`，以下以例子分别说明。

#### for循环
使用`for`循环输出0-100奇数代码可以这样写：

```objective-c
#import <Foundation/Foundation.h>
int main(int argc, const char * argv[])
{
    for (int i = 0; i < 100; ++i)
    {
    	if (i % 2 == 1)
    	{
    		printf("%d\t", i);
   	 }
    }
    printf("\n");
    return 0;
}
```

#### while循环
用`while`循环重写上个例子：

```objective-c
#import <Foundation/Foundation.h>
int main(int argc, const char * argv[])
{
    int num = 0;
    while(num <= 100)
    {
    	if (num % 2 == 1)
    	{
    		printf("%d\t", num);
   	 }
   	 ++num;
    }
    printf("\n");
    return 0;
}
```

### 函数
在Objective-C下也可以像C语言一样写函数，以一个例子来说明，这次把判断奇数的功能写在一个函数里，如下图：

```objective-c
#import <Foundation/Foundation.h>

BOOL isOdd(int num);

int main(int argc, const char * argv[])
{
    int num = 0;
    while(num <= 100)
    {
    	if (isOdd(num))
    	{
    		printf("%d\t", num);
   	 }
   	 ++num;
    }
    printf("\n");
    return 0;
}

BOOL isOdd(int num)
{
    BOOL isodd = num % 2 == 1;
    return isodd;
}
```

## 实例
写到这里，我们应该已经发现，我们写的全是C代码，没错，上文已经说过，Objective-C是兼容C语言的。

下面进入本系列实战环节，编写密码生成器，在本文中的实例将生成一个由6个数字组成的密码。代码中使用`arc4random`产生随机数，代码和注释如下：

```objective-c
#import <Foundation/Foundation.h>

#define SOURCE "0123456789"

char* getCharacters();
int random_int(int max);
BOOL generatePasswd(char* passwd, int len);
void show(char *str);

int main(int argc, const char * argv[])
{
    int len = 6;
    char passwd[7] = { 0 };
    if (generatePasswd(passwd, len))
    {
        show(passwd);
    }
    
    return 0;
}

//获取密码可能的取值
char* getCharacters()
{
    char* characters = SOURCE;
    return characters;
}

//产生一个[0,max)的随机数
int random_int(int max)
{
    int num = arc4random() % max;
    return num;
}

//产生密码，密码保存在passwd中，长度为6-50时产生密码并返回YES，否则返回NO
BOOL generatePasswd(char* passwd, int len)
{
    if (len < 6 || 50 < len)
    {
        return NO;
    }
    
    char* source = getCharacters();
    int sLen = (int)strlen(source);
    
    for (int i = 0; i < len; ++i)
    {
        int index = random_int(sLen);
        passwd[i] = source[index];
    }
    passwd[len] = '\0';
    return YES;
}

//显示密码
void show(char *str)
{
    printf("password:%s\n", str);
}
```

## 本文未提到的
*  宏定义
*  break, continue,goto....
*  条件编译
。。。。

## 结语
本文简要的介绍Objective-C面向过程的一些用法，使读者从C过度到Objective-C。

未完继续。。。。

