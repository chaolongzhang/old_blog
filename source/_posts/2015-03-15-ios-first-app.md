---
layout: post
category: [iOS,iOS开发入门教程]
title: iOS开发入门教程之第一个APP
keywords: iOS
tags: iOS
---

上一篇：[ iOS开发入门教程之Objective-C · 协议（Protocols](http://zh.5long.me/2015/learning-ios-oc-5/)

##前言
经过本系列前面几篇文章的学习，相信读者已经可以使用Objective-C语言写一些简单地程序了。Objective-C的语言基础也已经基本讲完了，通过大量的代码练习和阅读相关书籍，相信很快就能掌握iOS开发的相关技能。考虑到教程的完整性，本文通过一个小例子来开始我们的iOS应用开发，该例子应用在[iOS开发入门教程 · 楔子](http://zh.5long.me/2014/learning-ios-preface/)一文中已经提到——密码生成器。

正如你所想的，一篇文章是肯定无法覆盖iOS开发的方方面面，本文也不打算这么做，本文只是通过一个例子来开始读者的iOS开发之路，同时本人强烈推荐苹果官方的iOS开发入门教程：

*  [马上着手开发 iOS 应用程序 ](https://developer.apple.com/library/ios/referencelibrary/GettingStarted/RoadMapiOSCh/index.html)
*  [Start Developing iOS Apps Today](https://developer.apple.com/library/ios/referencelibrary/GettingStarted/RoadMapiOS/index.html)

更完整地iOS开发教程请参考[iOS开发入门教程 · 楔子](http://zh.5long.me/2014/learning-ios-preface/)中列出的资源：
>*  《Objective-C开发经典教程》
*  《Objective-C基础教程》
*  《Objective-C编程》
*  《Objective-C程序设计》
*  《iOS 7应用开发入门经典》
*  《iOS编程》
*  《精通iOS开发》
*  [斯坦福大学公开课：iPad和iPhone应用开发(iOS5)](http://v.163.com/special/opencourse/ipadandiphoneapplication.html)
*  [斯坦福大学公开课：iOS 7应用开发](http://v.163.com/special/opencourse/ios7.html)

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

##新建工程
*  启动XCode。
*  菜单`File->New->Project`选择新建一个工程。
*  选择iOS->Application->Single View Application，如下图：

![iOS](/assets/images/2015/ios-01.jpg)

*  Product Name设置为PasswdGenerater，其它的默认即可，如下图：

![iOS](/assets/images/2015/ios-02.jpg)

*  确认之后，就建立了一个单视图应用，工程文件如下：

![iOS](/assets/images/2015/ios-03.jpg)

##PasswdGenerater基础版
###制作界面
打开Main.storyboard，拖出如下界面，并设置约束（使用自动布局，自动布局参考[ iOS开发入门教程外篇之自动布局Auto Layout](http://zh.5long.me/2015/ios-ui-autolayout/)。

![iOS](/assets/images/2015/ios-04.jpg)

中间红色的`UILabel`用于显示生成的密码，下面的按钮控制触发生成密码的事件，右上角的`info`按钮暂时未使用。

为UILabel设置`IBOutlet`，如下：


```objective-c
@property (weak, nonatomic) IBOutlet UILabel *passwdLabel;
```

为按钮设置`IBAction`，如下：

```objective-c
- (IBAction)switchPressed:(UIButton *)sender
{
    //生成密码
}
```

###密码生成类
参考[iOS开发入门教程之Objective-C · 面向对象的Objective-C]()中的实例，新建一个密码生成器类，代码如下：

```objective-c
//PasswdGenerater.h
@interface PasswdGenerater : NSObject
@property (nonatomic, assign) int passwdLength;
- (NSString*)generatePasswd;
@end

//PasswdGenerater.m
#define SOURCE_NUMBER       @"0123456789"
#define SOURCE_LOWERCASE    @"abcdefghijklmnopqrstuvwxyz"
#define SOURCE_UPPERCASE    @"ABCDEFGHIJKLMNOPQRSTUVWXYZ"
#define SOURCE_MARK         @"`~!@#$%^&*(),./;'[]\<>?:\"{}|"
@interface PasswdGenerater ()
@property (nonatomic, copy) NSString* charachters;
@end

@implementation PasswdGenerater
- (id)init
{
    self = [super init];
    if (self)
    {
        self.charachters = [NSString stringWithFormat:@"%@%@%@%@", SOURCE_NUMBER, SOURCE_LOWERCASE, SOURCE_UPPERCASE, SOURCE_MARK];
        self.passwdLength = 12;
    }
    return self;
}
- (NSString*)generatePasswd
{
    NSMutableString* passwd = [[NSMutableString alloc] init];
    for (int i = 0; i < self.passwdLength; ++i)
    {
        int index = [self random_int:(int)self.charachters.length];
        [passwd appendFormat:@"%c", [self.charachters characterAtIndex:index]];
    }
    return passwd;
}
//产生一个[0,max)的随机数
- (int) random_int:(int) max
{
    int num = arc4random() % max;
    return num;
}
@end
```

###实现功能
当点击按钮时要能生成密码，就需要实现相应地事件。制作界面时，我们已经设置了按钮的`IBAction`，当点击这个按钮时，会执行`switchPressed`方法，在`switchPressed`实现相应地功能即可，代码如下：

```objective-c
- (IBAction)switchPressed:(UIButton *)sender
{
    if (nil == self.passwdGenerater)
    {
        self.passwdGenerater = [[PasswdGenerater alloc] init];
    }
    NSString* passwd = [self.passwdGenerater generatePasswd];
    self.passwdLabel.text = passwd;
}
```

###运行
至此，一个简易的密码生成器已经完成，运行后点击Generate按钮即可生成密码，效果图如下：

![iOS](/assets/images/2015/ios-05.png)

##结语
本文通过一个简单地例子让读者初涉iOS开发，PasswdGenerater可以进一步开发，已实现如下功能：

*  设置密码长度
*  设置密码的元字符，在数字、大/小写字母、特殊符号任意组合
*  使用开始/结束连续生成密码，结束后显示最后生成的密码

要想进一步学习iOS开发还需读相关书籍和代码训练。

未完待续。。。。
