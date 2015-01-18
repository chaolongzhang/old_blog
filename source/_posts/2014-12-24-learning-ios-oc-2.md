---
layout: post
category: iOS开发入门教程
title: iOS开发入门教程之Objective-C · 面向对象的Objective-C
keywords: iOS
tags: iOS
---

上一篇：[iOS开发入门教程之Objective-C · 面向过程的Objective-C]( http://zh.5long.me/2014/learning-ios-oc-1/)

#前言
在上一篇中介绍了Objective-C的面向过程的编程方法，以帮助读者从C语言过度到Objective-C。本文将介绍Objective-C的面向对象的编程方法，这也是开发iOS应用所使用的编程方法。本文假设读者已经有面向对象的基础。

<!--more-->

#新建一个类
*  按上一篇的方法新建一个工程。
*  菜单`File->New->File`新建一个文件。
*  选择OS X下的Source的Cocoa Class，点击下一步，如图：

![Objective-C](/assets/images/2014/oc-04.jpg)

*  给类命名，如下图：

![Objective-C](/assets/images/2014/oc-05.jpg)

*  保存，OK，类已建成，我们会发现多了两个文件`MyClass.h`和`MyClass.m`。

#类的源文件解析
Objective-C的原文件包括头文件（.h）和实现文件（.m）两种类型，这和C/C++类似。一般在头文件申明类的成员和方法，在实现文件中实现方法。

##头文件
打开MyClass.h，代码如下：

```objective-c
#import <Foundation/Foundation.h>

@interface MyClass : NSObject

@end
```

Objective-C申明一个类是以如下的方式

```objective-c
@interface 类名 : 父类
{
	//成员
}

//函数
@end
```

MyClass是新建的类，NSObject是内置的类，MyClass没有特定的父类，所以指定其父类为NSObject。在大括号中申明类的成员，在打括号外申明函数。如下图，我们给MyClass申明两个整姓数，并添加getter和setter方法，add实现把两个整数相加并输出。

![Objective-C](/assets/images/2014/oc-06.jpg)

```objective-c
#import <Foundation/Foundation.h>

@interface MyClass : NSObject
{
    int num1;
    int num2;
}

- (void)setNum1:(int)num;
- (int)getNum1;
- (void)setNum2:(int)num;
- (int)getNum2;
- (void)add;
@end
```

##实现文件
打开MyClass.m，代码如下：

```objective-c
#import "MyClass.h"

@implementation MyClass

@end
```

代码`#import "MyClass.h"`引入类的头文件，与头文件中类的申明类似，类的实现以如下格式：

```objective-c
#import "MyClass.h"

@implementation 类名
//实现文件
@end
```

补充实现头文件所申明的函数：

```objective-c
#import "MyClass.h"

@implementation MyClass

- (id)init
{
    self = [super init];
    if (self)
    {
        num1 = 0;
        num2 = 0;
    }
    return self;
}

- (void)setNum1:(int)num
{
    num1 = num;
}

- (int)getNum1
{
    return num1;
}

- (void)setNum2:(int)num
{
    num2 = num;
}

- (int)getNum2
{
    return num2;
}

- (void)add
{
    int sum = num1 + num2;
    NSLog(@"%d + %d = %d", num1, num2, sum);
}
@end
```

先忽略`init`函数，其它函数就不用解释了，出现了几个陌生的东西，我们稍后解释。

###构造函数
准确来讲Objective-C没有构造函数，习惯使用 `init***`作为初始化函数（`init`是NSObject的一个初始化函数），我们再来看看`init` 函数：

```objective-c
- (id)init
{
    self = [super init];
    if (self)
    {
        num1 = 0;
        num2 = 0;
    }
    return self;
}
```

`init`函数的返回值类型是`id`，`id`是一个泛类型指针，也就是说它可以任何类型，类似于C语言的`void*`。再看下一句`self=[super init]`，`self`是一个指针，它指向对象本身，类似于C++的`this`指针。`[super init]`中表示调用父类的`init`函数，Objective-C中是以`[]`来调用函数格式为`[对象 函数]`，在Objective-C中也称发消息，也就是想对象发送一个消息，然后对象根据这个消息来执行相应的动作。`if(self)`判断self是否为空，也可写成`if(nil == self)`，`nil`是Objective-C的空指针，根据苹果官方所说，当内存不够时新建一个对象会返回`nil`，所以在进一步初始化之前应检测一下对象是否创建成功。最后返回`self`，也就是返回对象自身。为什么要返回自身？让我们先看下创建一个类的方法：

```objective-c
MyClass m = [[MyClass alloc] init]
```

使用`alloc`在内存中创建一个`MyClass`对象，然后在向这个对象发送`init`消息，完成初始化。为什么要连着些，而不是如下写法呢？

```objective-c
//创建对象的错误方法
MyClass m = [MyClass alloc];
m = [m init];
```

注意，上一段代码是错误的！原因是`alloc`创建一个对象后并不会返回，也就是说`alloc`没有返回值。在创建完后我们必须向这个对象发送一个消息进行初始化，并从这次初始化函数中返回该对象（很抽象，是吧，这和C++/JAVA/C#有很大的区别）。这个初始化函数一般是`init···`，当然，也可以是其它函数，因为`init`也是一个普通函数。

#类的使用
在`main`函数中编写如下代码：

```objective-c
#import <Foundation/Foundation.h>
#import "MyClass.h"

int main(int argc, const char * argv[]) 
{
    MyClass *m = [[MyClass alloc] init];
    
    [m add];
    
    [m setNum1:1];
    [m setNum2:2];
    [m add];
    
    [m setNum1:3];
    [m add];
    
    NSlog(@"%d", [m getNum1]);

    return 0;
}
```

运行一下,输出如下：

```
2014-12-23 21:52:33.770 pro3[62690:1257051] 0 + 0 = 0
2014-12-23 21:52:33.771 pro3[62690:1257051] 1 + 2 = 3
2014-12-23 21:52:33.771 pro3[62690:1257051] 3 + 2 = 5
2014-12-23 21:52:33.771 pro3[62690:1257051] 3
```

先创建一个MyClass的对象m， m的两个成员num1, num2默认为0，所以第一次执行`add`时相加结果为0；然后分别调用`setNum1`和`setNum2`给num1和num2赋值，在次执行`add`时结果为3。使用`[m getNum1]`获得num1的值。

输出信息似乎多出了一些信息（如：2014-12-23 21:44:06.541 pro3[62669:1255233] ），因为我们是使用`NSLog`输出的而不是 `printf`，`NSLog`是Objective-C的一个内置函数，用于输出信息，在输出信息是会附加输出一些信息。

#属性
回顾一下类的实现代码，我们发现大部分代码都是在给num1和num2赋值和取值，如果我们添加更多的成员，就有得写更多的getter和setter，这看起来是一种无价值的重复劳动，根据DRY原则（Don't Repeat Yourself），我们应该想想更酷的方法来解决。感谢Brad Cox与Tom Love（Objective-C语言的创始人），Objective-C已经帮我们很好的解决了这个问题，通过申明属性（@property）来代替繁琐的getter和setter方法。下面使用属性重写MyClass类。

```objective-c
#import <Foundation/Foundation.h>

@interface MyClass : NSObject

@property (nonatomic, assign) int num1;
@property (nonatomic, assign) int num2;

- (void)add;

@end
```

```objective-c
#import "MyClass.h"

@implementation MyClass

@synthesize num1, num2;

- (id)init
{
    self = [super init];
    if (self)
    {
        self.num1 = 0;
        self.num2 = 0;
    }
    return self;
}

- (void)add
{
    int sum = self.num1 + self.num2;
    NSLog(@"%d + %d = %d", self.num1, self.num2, sum);
}
@end
```

`main`函数几乎不用修改，只需把`NSlog(@"%d", [m getNum1]);`改成`NSLog(@"%d", [m num1]);`。数一数代码，确实少了不少代码，而功能却没有减少。

在Objective-C中使用属性申明一个变量，在编译的时候就会自动生成setter和getter。属性的进一步解释将留在后面的引用计数部分。

为了突出属性的优越性，我们还可以重写下`main`函数，以我们熟悉的方式（对象名.成员名）来赋值和取值，代码如下：

```objective-c
#import <Foundation/Foundation.h>
#import "MyClass.h"

int main(int argc, const char * argv[]) {
    
    MyClass *m = [[MyClass alloc] init];
    
    [m add];
    
    //[m setNum1:1];
    m.num1 = 1;
    //[m setNum2:2];
    m.num2 = 2;
    [m add];
    
    //[m setNum1:3];
    [m add];
    
    NSLog(@"%d", m.num2);
    
    return 0;
}
```

#实例
以上对Objective-C的面向对象进行了最初级的介绍，下面来一个实例，继续本教程的密码生成器，本文将把上次的程序已面向对象的方式重写，并增强其功能。代码及注释如下：

类的头文件：

```objective-c
#import <Foundation/Foundation.h>

@interface PasswdGenerater : NSObject
{
    char *characters;   //密码元字符
}

@property (nonatomic, assign) int length;   //密码长度

//申明函数
- (BOOL) generatePasswd:(char*)passwd;
- (int) random_int:(int) max;

@end
```

类的实现文件：

```objective-c
#import "PasswdGenerater.h"

#define SOURCE "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

@implementation PasswdGenerater

- (id)init
{
    self = [super init];
    if (self)
    {
        characters = SOURCE;
        self.length = 6;
    }
    return self;
}

//产生密码，密码保存在passwd中，长度为6-50时产生密码并返回YES，否则返回NO
- (BOOL) generatePasswd:(char*)passwd
{
    if (self.length < 6 || 50 < self.length)
    {
        return NO;
    }
    
    int sLen = (int)strlen(characters);
    
    for (int i = 0; i < self.length; ++i)
    {
        int index = [self random_int:sLen];
        passwd[i] = characters[index];
    }
    passwd[self.length] = '\0';
    return YES;
}

//产生一个[0,max)的随机数
- (int) random_int:(int) max
{
    int num = arc4random() % max;
    return num;
}

@end
```

main函数：

```objective-c
#import <Foundation/Foundation.h>
#import "PasswdGenerater.h"

int main(int argc, const char * argv[])
{
    char passwd[9] = { 0 };
    
    PasswdGenerater *pg = [[PasswdGenerater alloc] init];
    pg.length = 8;
    if ([pg generatePasswd:passwd])
    {
        NSLog(@"%s", passwd);
    }
    return 0;
}
```

运行一下，本次输出如下：

```
2014-12-24 13:42:36.585 prom01[63073:1280247] rOCLm6tG
```

密码强度还行，大功告成。

#本文未提到的
*  封装、继承、多态
*  属性中自定义getter和setter
*  属性的设置，如nonatomic、assign、strong、weak等，这将在引用计数部分讲到
*  多个`init···`初始化函数

#结语
本文简要的介绍Objective-C面向对象的基本用法，使读者能以面向对象的方法使用Objective-C。

未完继续。。。。