---
layout: post
category: iOS开发入门教程
title: iOS开发入门教程之Objective-C · 引用计数
keywords: iOS
tags: iOS
---

上一篇：[iOS开发入门教程之Objective-C · 常用的数据类型](http://zh.5long.me/2015/learning-ios-oc-3/)

#前言
本文将介绍Objective-C的内存管理，内存管理是程序设计中常见的资源管理的一部分。内存管理不是一个轻松的活，当程序抛出`EXC_BAD_ACCESS`异常时，多半是因为内存管理出问题了。Apple在iOS5推出ARC（Automatic Reference Counting， 自动引用计数）后，内存管理稍微轻松了点。

#引用计数
我们已经知道对象的创建了，但是对象创建之后什么时候释放呢？最理想的就是在不再使用这个对象的时候就释放。Objective-C使用了一种引用计数的技术。当一个对象被创建后，它的引用计数是1，可以通过程序增加和减少它的引用计数，当引用计数为0的时候就可以释放该对象了。

<!--more-->

#手动管理引用计数
在手动管理引用计数（非arc）模式下，对象有一个属性`retainCount`记录该对象的引用计数，使用`[obj retainCount]`获得该对象的引用计数。使用retain增加引用计数，release减少引用计数。

##何时增加（retain）？
当把一个对象赋值给一个变量的时候就应该增加其引用计数，如使用如下方式赋值：


```objective-c
NSObject* obj1 = [[NSObject alloc] init];
NSLog(@"obj1 retainCount:%ld", [obj1 retainCount]);
NSObject* obj2 = [obj1 retain];
NSLog(@"obj1 retainCount:%ld, obj2 retainCount:%ld", [obj1 retainCount], [obj2 retainCount]);
```

输出：


```
2015-01-24 16:39:21.653 arc[676:21436] obj1 retainCount:1
2015-01-24 16:39:21.653 arc[676:21436] obj1 retainCount:2, obj2 retainCount:2
```

##何时释放（release）?
当一个变量不在指向改对象时，就应该使用`release`减少引用计数，修改上一段代码，`obj2`将不再指向obj1所指的对象：

```objective-c
//减少引用计数
[obj2 release];
NSLog(@"obj1 retainCount:%ld", [obj1 retainCount]);
obj2 = [[NSObject alloc] init];
NSLog(@"obj1 retainCount:%ld, obj2 retainCount:%ld", [obj1 retainCount], [obj2 retainCount]);
```

输出：

```
2015-01-24 16:53:03.197 arc[702:24199] obj1 retainCount:1
2015-01-24 16:53:03.197 arc[702:24199] obj1 retainCount:1, obj2 retainCount:1
```

在给`obj2`重新赋值之前，我们先指向`[obj2 release]`使所指向的对象引用计数减一，如果不减少引用计数而直接重新赋值，那么原先的对象引用计数还是2，当`obj1`执行减少引用计数之后，对象的引用计数为1，造成该对象无法释放，此时已发生内存泄露了。


#自动引用计数（ARC）
使用手动时对一个变量重新赋值要经过三个步骤：

*  旧对象的引用计数减一
*  把新对象赋给该变量
*  新对象的引用计数加一

对应的代码如下：

```objective-c
[obj release];
obj = newObj;
[obj retain];
```

简单的赋值就要使用三行代码，而且都是一些重复劳动，这些操作可以自动化，苹果可能也意识到了这个问题，在 iOS5后推出了自动引用计数。使用自动引用计数，对于上面的赋值，我们不再需要三行代码了，也只需一行代码搞定：

```objective-c
obj = newObj;
```

不过值得一提的时，ARC并不是垃圾回收机制，ARC是编译级的，也就是说这一行代码在编译时还是会转换成前面三句代码，再编译。运行环境并不知道ARC为何物。

新版的XCode默认就是使用ARC模式了，使用ARC也确实方便了许多。

#再谈属性
在[iOS开发入门教程之Objective-C · 面向对象的Objective-C](http://zh.5long.me/2014/learning-ios-oc-2/)已经介绍了属性。不过并没有进一步解释`assign`，还有`strong`，`weak`，`copy`等。

在ARC模式下，我们申明一个类的属性一般采用如下方式：

```objective-c
@property (nonatomic, strong) NSArray* items;
@property (nonatomic, weak) NSDictionary* map;
@property (nonatomic, copy) NSString* name;
```

##strong
strong与retail类似，赋值时引用计数会加一，如下代码，在类中申明两个属性：

```objective-c
//.h
@property (nonatomic, strong) NSObject* obj1;
@property (nonatomic, strong) NSObject* obj2;

//.m
- (void)fun1
{
    self.obj1 = [[NSObject alloc] init];
    self.obj2 = self.obj1;
    self.obj1 = nil;
    NSLog(@"%@", self.obj2);
}
```

输出(结果可能会不一样)：
```
2015-01-24 17:38:59.336 arc[846:34679] <NSObject: 0x1002021b0>
```

当执行`self.obj2 = self.obj1`后，`obj1`和`obj2`指向同一个对象，该对象的引用计数为2，执行`self.obj1 = nil`后，该对象的引用计数减一，`obj2`还是指向该对象，所以能正常输出。

##weak
weak与retail不同，赋值时只是简单地对对象的引用，不会增加引用计数，当有多个变量指向某一对像并其中一个变量释放该对象后，那么该对象会被释放，其它的变量都被赋值为`nil`。如下代码，


```objective-c
//.h
@property (nonatomic, strong) NSObject* obj1;
@property (nonatomic, weak) NSObject* obj2;

//.m
self.obj1 = [[NSObject alloc] init];
self.obj2 = self.obj1;
self.obj1 = nil;
NSLog(@"%@", self.obj2);
```

输出：


```
2015-01-24 17:49:12.230 arc[898:37498] (null)
```

##assign
直接赋值，对值变量，如：int、Boolean。这类属性使用。

##copy
赋值时传入一份拷贝，将指向新的对象。

#本文未提到的
*  nonatomic
*  只读、只写属性
*  retail修饰属性

#结语
本文介绍了Objective-C的内存管理，包括手动管理和ARC，尽管现在已ARC为主，但了解手动管理也有助于我们写出更强壮的代码，而且有些第三方库还是使用手动管理的方式，所以了解手动管理内存还是有必要地。

未完继续。。。。
