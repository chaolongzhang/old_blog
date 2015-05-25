---
layout: post
category: [iOS,开源]
title: iOS网络资源缓存ZCLURLCache·下篇
keywords: URLCache
tags: iOS
styles: [data-table]
---

##前言
本文接着上一篇（ [iOS网络资源缓存ZCLURLCache·上篇](http://zh.5long.me/2015/ios-zcl-url-cache/)）继续介绍本人开发的一个开源iOS版网络资源缓存库ZCLURLCache，代码已托管在[GitHub](https://github.com/zchlong/ZCLURLCache.git)，欢迎下载测试，若发现错误，请留言告知，不甚感激。

在上一篇中主要介绍开发ZCLURLCache的历史原因、ZCLURLCache架构、ZCLURLCache使用方法。本文将介绍ZCLURLCache的设计细节。

##ZCLURLCache架构
在上一篇中已经介绍ZCLURLCache分为五个类：

>1.  ZCLCacheManager，缓存管理类，单例，也是唯一一个需要在客户端代码中使用的类。
>2.  ZCLMemoryCache，内存缓存类，在内存中缓存数据并提供相关方法。
>3.  ZCLDiskCache，磁盘缓存类，在磁盘上缓存数据并提供相关方法。
>4.  ZCLUrlDownloader，数据下载类。
>5.  ZCLUtil，工具类，MD5编码。

<!--more-->

##预备知识
在介绍ZCLURLCache之前，先介绍几个ZCLURLCache使用到的东西。

*  ***NSCache***。[NSCache](https://developer.apple.com/library/ios/documentation/Cocoa/Reference/NSCache_Class/index.html)可在内存中临时缓存对象，类似于[NSDictionary](https://developer.apple.com/library/ios/documentation/Cocoa/Reference/Foundation/Classes/NSDictionary_Class/index.html#//apple_ref/occ/cl/NSDictionary)，使用方法也类似，ZCLURLCache使用NSCache作为内存缓存容器。
*  ***NSFileManager***。[NSFileManager](https://developer.apple.com/library/ios/documentation/Cocoa/Reference/Foundation/Classes/NSFileManager_Class/)是一个文件管理类，实现了文件的基本操作。有一点需要提醒，选择保存数据的文件夹要慎重，本人就有因数据保存的位置不当而导致APP被拒的经历。ZCLURLCache使用NSFileManager实现磁盘缓存，缓存文件保存在`Library/Caches/me.5long.zh.cache`。
*   ***NSMutableURLRequest***。[NSMutableURLRequest](https://developer.apple.com/library/ios/documentation/Cocoa/Reference/Foundation/Classes/NSMutableURLRequest_Class/)就不多说了。: D
*  ***单例模式***。单例模式不用过多介绍，听过设计模式的人都知道。使用单例模式是因为整个应用要共享一个缓存（不然缓存就不起作用了）。在ZCLURLCache源代码中多次出现类似于如下的代码：

```objective-c
+ (ZCLCacheManager*)shareCacheManager
{
    static ZCLCacheManager* instance = nil;
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^
    {
        instance = [[ZCLCacheManager alloc] init];
    });

    return instance;
}
```

没错，在Objective-C下实现一个单例模式就这么简单。

##ZCLURLCache的实现
###ZCLCacheManager
ZCLCacheManager是缓存管理类，使用单例模式，也是唯一一个需要在客户端代码中使用的类。ZCLCacheManager有三个成员：

| 成员名    | 对象         | 应用    |
| :-----------| :------------| :---------|
| memoryCache | ZCLMemoryCache | 内存缓存 |
| diskCache        | ZCLDiskCache        | 磁盘缓存 |
| urlDownload    | ZCLUrlDownloader | 下载URL资源 |

ZCLCacheManager的方法有。

| 方法名    | 实现功能 |
| :-----------| :------------|
| + (ZCLCacheManager*)shareCacheManager                                                                        | 获取ZCLCacheManager实例 |
| - (void)asynDataWithUrl:(NSString*)url completed:(void (^)(NSData* data))completed | 获取URL对应的数据                 |
| - (void)cleanMemoryCache                                                                                                         | 清空内存缓存                             |
| - (void)cleanDiskCache                                                                                                                | 清空磁盘缓存                             |

这里重点介绍`- (void)asynDataWithUrl:(NSString*)url completed:(void (^)(NSData* data))completed`，当客户端代码需要获取URL对应的数据时，需要调用此函数，函数将会执行如下流程：

1.  在memoryCache查找是否有对应的数据，若有，则取出数据，调用`completed`返回数据，否则执行下一步；
2.  在diskCache查找是否有对应的数据，若有，则从磁盘取出数据，并把数据添加到memoryCache，调用`completed`返回数据，否则执行下一步；
3.  从网络上下载URL对应的数据，把数据添加到memoryCache中并保存到磁盘，调用`completed`返回数据。

###ZCLMemoryCache
ZCLMemoryCache是内存缓存类，在内存中缓存数据并提供相关方法。ZCLMenoryCache的成员只有一个——memoryCache，类型为NSCache。这个类主要是对NSCache的封装，其方法有：

| 方法名    | 实现功能 |
| :-----------| :------------|
| + (ZCLMemoryCache*)shareMemoryCache                                            | 获取ZCLMemoryCache实例                  	|
| - (BOOL) urlInMemoryCache:(NSString*)url                                             | 测试内存中是否缓存有url对应的数据   	|
| - (NSData*) dataWithUrl:(NSString*)url                                                     | 从内存缓存中取出url对应的数据      	|
| - (void)addData:(NSData*)data withUrl:(NSString*)url                           | 向内存缓存中添加一个数据          		|
| - (void)cleanMemoryCache                                                                          | 清空内存缓存                                    	|


###ZCLDiskCache
ZCLDiskCache是磁盘缓存类，在磁盘上缓存数据并提供相关方法。ZCLDiskCache与ZCLMemoryCache类似，唯一的区别是ZCLDiskCache缓存在磁盘上。

ZCLDiskCache的方法有：

| 方法名    | 实现功能 |
| :-----------| :------------|
| + (ZCLDiskCache*)shareDiskCache			| 获取ZCLDiskCache实例			|
| - (BOOL) urlInDiskCache:(NSString*)url 			| 测试磁盘中是否缓存有url对应的数据	|
| - (NSData*) dataWithUrl:(NSString*)url 			| 从磁盘缓存中取出url对应的数据		|
| - (void)addData:(NSData*)data withUrl:(NSString*)url 	| 向磁盘缓存中添加一个数据			|
| - (void)cleanDiskCache					| 清空磁盘缓存					|

ZCLDiskCache主要是对磁盘文件的操作，使用`NSFileManager`操作文件。

###ZCLUrlDownloader
ZCLUrlDownloader是数据下载类，只有一个功能——下载URL对应的数据。ZCLUrlDownloader的方法有：

| 方法名    | 实现功能 |
| :-----------| :------------|
| + (ZCLUrlDownloader*)shareUrlDownloader						| 获取ZCLUrlDownloader实例|
| - (void)downloadUrl:(NSString*)url completed:(void (^)(NSData*, NSError*)) completed	| 下载URL对应的数据 		|

下载数据时先创建一个`NSMutableURLRequest`对象，然后使用`[NSURLConnection sendAsynchronousRequest:request queue:[NSOperationQueue mainQueue] completionHandler:^(NSURLResponse *response, NSData *data, NSError *connectionError)`发起异步请求，请求完成后通过调用block返回数据和错误（若无错误则为nil）。

###ZCLUtil
ZCLUtil是一个工具类，目前只有一个功能——MD5编码。ZCLUtil的方法有：

| 方法名    | 实现功能 |
| :-----------| :------------|
| (NSString*)md5_32:(NSString *)string  | 对string进行MD5编码	|

使用MD5编码主要有一些两个原因：

1.  URL不适宜直接作为文件名，如'/'会被当作目录分隔符，而且URL长度可能会很长。
2.  MD5编码可确保URL与文件名一一对应，即惟一性，且文件名长度固定。

##后记
ZCLURLCache的诞生起源于一次坑爹的测试，如今已可作为基础组件使用。如果你阅读了源代码（GitHub上可[下载](https://github.com/zchlong/ZCLURLCache.git)），你会发现ZCLURLCache的代码非常简单，几乎没有多余的代码。KISS原则（Keep It Simple, Stupid）倡导在设计中应当注重简约的原则，这也是我推荐的一种实践方式。

ZCLURLCache还存在很多不足和缺陷（上一篇已经提到过），同时也包含很多错误（bugs），欢迎下载测试，若发现错误，还请告知，不甚感激。
