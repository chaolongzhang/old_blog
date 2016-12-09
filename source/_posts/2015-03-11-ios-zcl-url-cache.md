---
layout: post
categories: 开源
title: iOS网络资源缓存ZCLURLCache·上篇
keywords: URLCache
tags: iOS
---

## 前言
iOS网络资源缓存ZCLURLCache分为上下两篇，主要介绍本人开发的一个开源iOS版网络资源缓存库ZCLURLCache，代码已托管在[GitHub](https://github.com/chaolongzhang/ZCLURLCache.git)，欢迎下载测试，若发现错误，请留言告知，不甚感激。

本文介绍本人开发ZCLURLCache的历史原因、ZCLURLCache架构、ZCLURLCache使用方法。若你只打算使用这个组件而不深入了解它，那么看这篇文章就已经足够。

下篇将介绍ZCLURLCache的设计细节，敬请关注。

## 从一个故事说起
去年给某司做了个iOS APP，其中各种辛酸在此就不多说，将来有机会再说，这里只聊一聊流量的问题。话说经过日夜奋战，终于到了现场测试的阶段了，一说现场测试，还有点小激动，可以免费参观某景区 :D

<!--more-->

在测试前一段时间发现了几个Bug，测着测着就发现这网络资源怎么就加载不出来了，该不会是网速太慢吧，不会啊，我用的可是4G网络。4G！不对，查下话费，已欠费！4G伤不起。一百大洋就这样没了。

![smiley](/assets/images/face/smiley_003.png)

有了这个惨痛的教训，就该想想怎么解决，最好是把以后类似的问题也解决了。既然下载网络资源太费流量，那就应该把资源缓存起来，有了方向后，就取网上找相关资料，有用的资源有：

*  [NSCache](https://developer.apple.com/library/ios/documentation/Cocoa/Reference/NSCache_Class/index.html)
*  [NSUrlCache](https://developer.apple.com/library/ios/documentation/Cocoa/Reference/Foundation/Classes/NSURLCache_Class/)
*  [SDWebImage](https://github.com/rs/SDWebImage)

使用SDWebImage解决图片缓存，OK，流量问题基本解决（当然不仅仅是图片的问题），最好写代码的方式就是不写代码。

![smiley](/assets/images/face/smiley_011.png)

## 进一步思考
在开发移动应用时应该尽量少用数据流量，其原因有三：

1.  服务商太坑，流量往往很贵。
2.  数据流量的速度可能会很慢，尤其是在偏僻的地方或人多的地方。
3.  由于网络不好的原因，APP看起来会很卡，用户体验不好。

大量使用用户流量的结果只有一个，卸载应用。

这样看来缓存网络资源是很有必要的，使用SDWebImage缓存图片是一个比较好的方案，但它也有不足：

1.  正如它的名字，它只能缓存图片，像网页、声音、视频等就无能为力了。
2.  SDWebImage较复杂，当然使用起来还是挺方便的，所以这个不算缺点。
3.  比较重量级，有些是我们不需要的功能，但会使应用体积变大。

如果针对每一个资源使用一个缓存的组件的话，会使应用的体积进一步变大。有一个通用的组件也许会更实用，在iOS平台下可以使用`NSData`来存储网络资源，在使用的时候再把`NSData`还原成具体的文件。

## ZCLURLCache问世
基于以上的思考，本人决定写一个通用的网络资源缓存组件——ZCLURLCache。ZCLURLCache是一个轻量级的网络资源缓存组件，具有如下特点：

1.  使用`NSData`保存数据，可以转换成`NSData`的数据都可以存储，如图片、html、声音、视频等文件。
2.  通过一个URL请求数据时，若已有缓存，则直接使用缓存数据。
3.  使用异步方式下载数据，不会阻塞主线程。
4.  使用一致的`block`返回数据，使用缓存还是下载数据是透明的。
5.  使用MD5编码URL，确保文件不会重复。

## ZCLURLCache的流程
当通过某个URL获取数据时，组件的流程如下：

1.  检查URL对应的数据是否在内存中有缓存，若有，直接取出数据返回，否则进入第二步。
2.  检查URL对应的数据是否在磁盘上又缓存，若有，从磁盘上读取数据，把数据放入内存缓存，并返回数据。
3.  联网下载URL对应的数据，下载完后分别缓存在内存和磁盘，返回数据。

## ZCLURLCache架构
ZCLURLCache分为五个类：

1.  ZCLCacheManager，缓存管理类，单例，也是唯一一个需要在客户端代码中使用的类。
2.  ZCLMemoryCache，内存缓存类，在内存中缓存数据并提供相关方法。
3.  ZCLDiskCache，磁盘缓存类，在磁盘上缓存数据并提供相关方法。
4.  ZCLUrlDownloader，数据下载类。
5.  ZCLUtil，工具类，MD5编码。

每个类的具体实现将在下一篇介绍。

## ZCLURLCache的使用
###获取ZCLURLCache
代码已托管在GitHub，其中包括一个图片缓存的例子，地址<https://github.com/chaolongzhang/ZCLURLCache.git>

### 获取URL数据
ZCLURLCache的使用非常简单，步骤如下：

1.  拷贝上述五个类文件到项目工程下（在文件`zclcachecore`）。
2.  在要使用的类中引用头文件`#import "ZCLCacheManager.h"`。
3.  调用`[ZCLCacheManager shareCacheManager]`获得缓存管理类的单例。
4.  调用`- (void)asynDataWithUrl:(NSString*)url completed:(void (^)(NSData*))completed`获得数据，数据从`block`返回，如在`UITableView`中可以这样使用：

```objective-c
UIImageView* imageView = (UIImageView*)[cell viewWithTag:101];
imageView.image = [UIImage imageNamed:@"meplace.png"];
[self.cacheManager asynDataWithUrl:url completed:^(NSData *data)
{
    UIImage* image = [UIImage imageWithData:data];
    imageView.image = image;
}];
```

### 清空缓存
缓存管理类（ZCLCacheManager）提供两个方法分别清空内存缓存和磁盘缓存：

1.  `cleanMemoryCache`，清空内存缓存。
2.  `cleanDiskCache`，清空磁盘缓存。

调用前需要调用`[ZCLCacheManager shareCacheManager]`获得缓存管理类的单例，如：

```objective-c
ZCLCacheManager* cacheManager = [ZCLCacheManager shareCacheManager];
[cacheManager cleanMemoryCache];
[cacheManager cleanDiskCache];
```

## 结语
`ZCLURLCache`的开发和测试耗时一个晚上和一个上午（晚上7:00-9:00，上午9:30-11:30），还要加上前期查找资料的时间。

目前`ZCLURLCache`还不完善，下一步将在以下几个方面改进：

*  使用一个下载队列，在没有缓存是把URL加入下载队列，而不是直接下载，已防止同一时间有过多的URL在下载。
*  自动释放一些缓存，在不需要手动清空缓存的情况下保证内存不会过多的使用。
*  缓存有效机制，URL数据更新是能及时更新缓存数据。
*  下载时返回进度信息。

写的代码难免出错，欢迎下载测试，若发现错误，还请告知，不甚感激。

未完待续。。。。

