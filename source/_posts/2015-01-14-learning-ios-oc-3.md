---
layout: post
categories: iOS开发入门教程
title: iOS开发入门教程之Objective-C · 常用的数据类型
keywords: iOS
tags: [iOS]
---

上一篇：[iOS开发入门教程之Objective-C · 面向对象的Objective-C]( http://zh.5long.me/2014/learning-ios-oc-2/)

## 前言
在上一篇中介绍了Objective-C的面向对象的编程方法。今天来一点轻松的话题，介绍几个常用的数据类型，分别是：字符串（NSString/ NSMutableString）、数组(NSArray/NSMutableArray)、键值对(NSDictionary/NSMutableDictionary)。这几个数据类型属于Foundation Framework中的一部分，关于Foundation Framework的详细内容可参考[苹果官方文档Foundation Framework](https://developer.apple.com/library/ios/documentation/Cocoa/Reference/Foundation/ObjC_classic/index.html)

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

## 字符串
字符串是程序中最常使用的一种数据类型，在Objective-C下对应的类是`NSString/NSMutableString`。

### NSString
`NSString`是不可变的字符串类型，什么是不可变？也就是说这个对象创建后就无法改变这个对象的值了。比如创建了一个字符串对象s的值为"zh.5long.me"，那么对象s的值就一直是"zh.5long.me"，不能修改。

#### 创建字符串
创建字符串的方式有很多，此处列出一些常用的方法：

*  直接赋值：NSString *s = @"zh.5long.me";
*  init创建一个空的字符串：NSString *s = [[NSString alloc] init];
*  通过一个字符串创建一个新的字符串：NSString *ss = [[NSString alloc] )initWithString:s]、NSString *ss = [NSString stringWithString:s];
*  格式化创建：NSString *ss = [[NSString alloc] )initWithFormat:@"zh.%dlong.%@", 5, @"me"]、NSString *ss = [NSString stringWithFormat:@"zh.%dlong.%@", 5, @"me"];

#### 连接字符串
*  直接连接：NSString *newStr = [ss stringByAppendingString: @"abc"];
*  格式化连接：NSString *newStr = [ss stringByAppendingFormat: @"zh.%dlong.%@", 5, @"me"];

#### 其它方法
*  获取长度：int len = [ss length];
*  字串替换：NSString *newStr = [ss stringByReplacingOccurrencesOfString:@"aa"  withString:@"bb"];

### NSMutableString
NSMutableString是可变的字符串，NSMutableString继承自NSString，所以NSMutableString有NSString有相同的方法。另外它还扩展了修改字符串的方法：

*  appendFormat:格式化添加字符串，如：

```objective-c
NSMutableString *ms = [[NSMutableString alloc] initWithString:@"aa"];
NSLog(@"%@", ms);
[ms appendFormat:@"%@", @"bb"];
NSLog(@"%@", ms);
```

前一个输出为：aa，后一个输出为：aabb。

*  appendString:添加一个字符串，如：

```objective-c
NSMutableString *ms = [[NSMutableString alloc] initWithString:@"aa"];
NSLog(@"%@", ms);
[ms appendFormat:@"bb"];
NSLog(@"%@", ms);
```

前一个输出为：aa，后一个输出为：aabb。

### 可变与不可变
初次接触可变与不可变的概念可能会使读者非常迷惑，比如看下面一段代码：

```objective-c
NSString *str = @"aa";
NSLog(@"%@", str);
str = @"bb";
NSLog(@"%@", str);
```

这段代码前一个输出为：aa，后一个输出为：bb。

这段代码中先把`str`赋值为"aa"，然后再把"bb"赋值给它，这似乎跟前面说NSString是不可变的字符串相矛盾，明明已经改变了`str`的值，怎么又说NSString是不可变的呢？其实这是不矛盾的，Objective-C中申明的对象都是指针，这个指针在指向一个实际的对象，如第一句`NSString *str = @"aa"`，执行这一句时会创建一个字符串对象，str指向这个对象。当执行`str = @"bb"`时，将会创建一个新的对象，其值为"bb"，此时str指向这个新的对象，而前一个对象并未改变，只是str指向的对象不同了。

其它类型关于可变与不可变的概念与此一致，将不再阐述。

## 数组(NSArray/NSMutableArray)
数组是一个对象的集合，并且这些对象保持一定的顺序。如果我们要保存很多字符串，那么就需要使用数组。NSArray和NSMutableArray除不可变和可变外，其它操作一致，相同部分将不再区分。

### 创建并初始化
*  initWithObjects:和arrayWithObjects:，如：

```objective-c
 NSArray *arr1 = [NSArray arrayWithObjects:@"1", @"2", @"3", nil];
    
NSArray *arr2 = [[NSArray alloc] initWithObjects:@"aa", @"bb", @"cc", nil];
```

*  NSArray还可保存不同类型的对象，如：

```objective-c
NSArray *myArray;
NSDate *aDate = [NSDate distantFuture];
NSValue *aValue = [NSNumber numberWithInt:5];
NSString *aString = @"a string";
myArray = [NSArray arrayWithObjects:aDate, aValue, aString, nil];
```

### 取值

*  通过下标取值：`NSString *s = [arr1 objectAtIndex:1]`;
*  for遍历：

```objective-c
NSUInteger len1 = [arr1 count];
for (int i = 0; i < len1; ++i)
    {
        NSString *item = [arr1 objectAtIndex:i];
        NSLog(@"%d:%@", i, item);
    }
```

*  forin遍历，使用for循环遍历需要手动处理结束条件，这就可能会出现两个非常可怕的事情，数组越界和off-by-one(大小差一)，使用forin能很好的处理这些问题，使用方式如下：

```objective-c
for (NSString *item in arr2)
{
        NSLog(@"%@", item);
}
```

### 修改列表（仅限NSMutableArray）
NSSarry是不可修改的列表，所以初始化后就不能再修改，NSMutableArray是可修改的列表，其提供的操作有：

*  向末尾添加一个元素：`[marray addObject:@"bb"]`;
*  插入一个元素：` [marray insertObject:@"cc" atIndex:1]`;
*  删除一个元素：`[marray removeObject:@"aa"]`;

## 键值对(NSDictionary/NSMutableDictionary)
键值对是一种哈希表，保存键值对，通过键就可以快速地找到对应的值，键值对是无序的集合。NSDictionary和NSMutableDictionary除不可变和可变外，其它操作一致，相同部分将不再区分。

### 创建并初始化

*  dictionaryWithObjectsAndKeys: 如：

```objective-c
NSDictionary *dict = [NSDictionary dictionaryWithObjectsAndKeys: @"value1", @"key1", @"value2", @"key2", nil];
NSDictionary *dic1 = [[NSDictionary alloc] initWithObjectsAndKeys:@"v1", @"k1", @"v2", @"k2", @"v3", @"k3", nil];
```

*  dictionaryWithObjects:forKeys:如：

```objective-c
NSDictionary *dic2 = [[NSDictionary alloc] initWithObjects:[NSArray arrayWithObjects:@"v1", @"v2", @"v3", nil] forKeys:[NSArray arrayWithObjects:@"k1", @"k2", @"k3", nil]];
```
### 取值
*  通过键直接取值：` NSString* v1 = [dic1 valueForKey:@"k2"];v1 = dic1[@"k1"]`;
*  forin遍历：

```objective-c
for (NSString *key in dic2)
{
        NSString *value = dic2[key];
        NSLog(@"key:%@  value:%@", key, value);
}
```

### 修改键值对（仅限NSMutableDictionary）

*  添加元素：

```objective-c
NSMutableDictionary *mdic = [[NSMutableDictionary alloc] initWithObjectsAndKeys:@"v1", @"k1", nil];
[mdic setObject:@"v2" forKey:@"k2"];
[mdic setValue:@"v3" forKey:@"k3"];
```

*  修改元素，使用相同的键就可以修改对应的值：
```objective-c
[mdic setObject:@"v11" forKey:@"k1"];
```

*  删除元素：

```objective-c
[mdic removeObjectForKey:@"k1"];
```

*  清空元素：

```objective-c
[mdic removeAllObjects];
```

## 本文未提到的
本文只介绍了这些类型的简单使用，还有很多特性并未介绍，关于这些类型的详细介绍和Foundation Framework其它类型的详细介绍请参考[苹果官方文档Foundation Framework](https://developer.apple.com/library/ios/documentation/Cocoa/Reference/Foundation/ObjC_classic/index.html)

## 结语
本文介绍了Objective-C的三个数据类型：NSString、NSArray、NSDictionary和对应的可变类型。掌握了这些基本类型后，就可举一反三，结合相关文档，就可使用Foundation Framework其它的类型。

未完继续。。。。