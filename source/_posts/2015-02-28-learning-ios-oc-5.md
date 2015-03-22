---
layout: post
category: iOS开发入门教程
title: iOS开发入门教程之Objective-C · 协议（Protocols）
keywords: iOS
tags: iOS
---

上一篇：[iOS开发入门教程之Objective-C · 引用计数](http://zh.5long.me/2015/learning-ios-oc-4/)

##前言
本文将介绍Objective-C的一个重要概念——[协议（Protocols）](https://developer.apple.com/library/ios/documentation/Cocoa/Conceptual/ProgrammingWithObjectiveC/WorkingwithProtocols/WorkingwithProtocols.html#//apple_ref/doc/uid/TP40011210-CH11-SW1)，iOS编程将经常使用到协议。比如在iOS中经常使用到的[UITableView](https://developer.apple.com/library/ios/documentation/UIKit/Reference/UITableView_Class/index.html#//apple_ref/doc/uid/TP40006943)就需要通过协议来设置显示数据和响应事件。所以，掌握协议对iOS编程是非常重要的。Objective-C的协议和其它面向对象语言的接口类似，当然，协议有更多的特性。

<!--more-->

本系列教程内容如下：

*  [楔子](http://zh.5long.me/2014/learning-ios-preface/)
*  [面向过程的Objective-c](http://zh.5long.me/2014/learning-ios-oc-1/)
*  [面向对象的Objective-c](http://zh.5long.me/2014/learning-ios-oc-2/)
*  [常用的数据类型](http://zh.5long.me/2015/learning-ios-oc-3/)
*  [引用计数](http://zh.5long.me/2015/learning-ios-oc-4/)
*  [协议](http://zh.5long.me/2015/learning-ios-oc-5/)
*  iOS应用
* 跋

##协议的申明与实现
###简单的协议
简单的协议于其它语言的接口类似，申明一些方法，实现该协议的类将实现相应的方法，使用`@protocol`申明协议，格式如下:

```objective-c
@protocol ProtocolName
// 方法列表
@end
```

如：

```objective-c
@protocol DataSource

- (int)numberOfItems;
- (NSString*)itemAtIndex:(int)index;

@end
```

<h3 id="SU">简单协议的实现</h3>
可申明一个类，这个类实现某个协议，如：

```objective-c
@interface MyStringDatas : NSObject <DataSource>

@property (nonatomic, strong) NSArray* items;

@end

@implementation MyStringDatas

- (id)initWithItems:(NSArray*)aItems
{
    self = [super init];
    if (self)
    {
        self.items = aItems;
    }
    return self;
}

- (int)numberOfItems
{
    return (int)[self.items count];
}

- (NSString*)itemAtIndex:(int)index
{
    return [self.items objectAtIndex:index];
}

@end
```

申明类是使用`<协议名>`来申明这个类实现某个协议，并在实现该协议的方法。

###协议的继承
协议也可继承其它协议，最常见的就是继承`NSObject`(NSObject被分为NSObject类和NSObject协议，NSObject类实现了NSObject协议)。继承`NSObject`的协议格式如下：

```objective-c
@protocol MyProtocol <NSObject>
...
@end
```

如：

```objective-c
@protocol DataSource <NSObject>

- (int)numberOfItems;
- (NSString*)itemAtIndex:(int)index;

@end
```

任何实现DataSource协议的类也需要实现NSObject协议。

###继承协议的实现
同[简单协议的实现](#SU),类应该继承自`NSObject`，这样也就实现了`NSObject`协议。

###协议的可选与必选方法
协议申明的方法可分为可选与必选方法，这点与接口不同。顾名思义，可选方法就是在实现类可以选择性的决定是否实现该方法，必选方法就必选要实现。Objective-C分别使用`@optional`和`@required`来申明可选于必选方法，格式如下：

```objective-c
@protocol ProtocolName

@required
//必选方法列表

@optional
//可选方法列表

@end
```

默认为必选方法。

修改`DataSource`协议如下：

```objective-c
@protocol DataSource <NSObject>

@required
- (int)numberOfItems;
- (NSString*)itemAtIndex:(int)index;

@optional
- (NSString*)randomItem;

@end
```

###可选于必选协议的实现
同[简单协议的实现](#SU),申明为必选的就必选在类中实现，可选的方法，可以不实现。

###检查可选方法是否被实现
可选方法在类中可以不实现，但如果在运行时调用为实现的方法将会导致运行时异常，那么如何确定可选方法是否被实现呢？Objective-C提供了如下方式检查一个方法是否实现：

```objective-c
if ([dataSource respondsToSelector:@selector(randomItem)])
{
//randomItem方法已实现，可以调用
}
else
{
//randomItem方法未实现，不可调用
}
```

###实现多个协议
和接口类似，同一个类可以实现多个协议。格式如下：

```objective-c
@interface MyClass : NSObject <MyProtocol, AnotherProtocol, YetAnotherProtocol>
...
@end
```

##协议的使用
协议的使用和接口类似，创建一个实现了该协议的对象，就可以调用该协议申明的方法。例子如下：

```objective-c
NSArray* strs = @[@"one", @"two", @"three"];
id<DataSource> dataSource = [[MyStringDatas alloc] initWithItems:strs];  //创建一个实现DataSource协议的对象
int len = [dataSource numberOfItems];   //调用协议的方法
for (int i = 0; i < len; ++i)
{
    NSString* item = [dataSource itemAtIndex:i];
    NSLog(@"item %d: %@", i, item);
    if ([dataSource respondsToSelector:@selector(randomItem)])  //检查是否实现randomItem
    {
        //此处不会执行
        NSString* randomItem = [dataSource randomItem];
        NSLog(@"random item %d: %@", i, randomItem);
    }
}
```

##结语
本文介绍了Objective-C的协议，iOS编程大量使用了协议，所以掌握协议是很有必要地。

参考：[Working with Protocols](https://developer.apple.com/library/ios/documentation/Cocoa/Conceptual/ProgrammingWithObjectiveC/WorkingwithProtocols/WorkingwithProtocols.html#//apple_ref/doc/uid/TP40011210-CH11-SW1)

未完继续。。。。


