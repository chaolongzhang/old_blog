---
layout: post
categories: 开源
date: 2015-09-07 00:00:30
title:  源码阅读:SDWebImage
keywords: SDWebImage源码阅读
tags: iOS
---

## 前言
SDWebImage是一个开源的iOS第三方库，主要用于下载并缓存网络图片，在我的博文[iOS网络资源缓存ZCLURLCache·上篇](http://zh.5long.me/2015/ios-zcl-url-cache/)提到过SDWebImage。它提供了`UIImageView`、`MKAnnotationView`、`UIButton`的categories（分类），支持网络图片的加载与缓存，其中`UIImageView`应该是使用最广泛的。

*  源码：<https://github.com/rs/SDWebImage>
*  目前版本：3.7

本文从源码的角度讨论SDWebImage的下载和缓存的实现。

<!--more-->

## 流程
在SDWebImage的使用例子中，给`UIImageView`设置图片的代码是：

```objective-c
[cell.imageView sd_setImageWithURL:[NSURL URLWithString:[_objects objectAtIndex:indexPath.row]]
                      placeholderImage:[UIImage imageNamed:@"placeholder"] options:indexPath.row == 0 ? SDWebImageRefreshCached : 0];
```

可以看出，SDWebImage的使用非常简单，只用这一行代码，就可实现网络图片的加载与缓存。在这行代码的背后，会执行下列操作：

1.  `UIImageView+WebCache`categories会从管理类`SDWebImageManager`找图片，并刷新`UIImageView`。
2.  `SDWebImageManager`向从缓存类`SDImageCache`找URL对应的图片缓存，如果没找到，起用`SDWebImageDownloader`下载图片。
3.  缓存类`SDImageCache`会先在内存`NSCache`中找图片，如果内存中没有，就在磁盘上找，在磁盘上找到了，把图片放入内存。
4.  `SDWebImageDownloader`会创建一个`SDWebImageDownloaderOperation`操作队列下载图片，下载完后缓存在内存和磁盘上（可选）。
5.  `SDWebImageDownloaderOperation`操作队列使用`NSURLConnection`在后台发起请求，下载图片，反馈进度和下载结果。

源码中有几个关键的类，分别是：

1.  `SDImageCache`，缓存类，在内存和磁盘上缓存图片，并对图片编码。
2.  `SDWebImageDownloader`，下载管理类，下载图片。
3.  `SDWebImageDownloaderOperation`，下载操作队列，继承自`NSOperation`，在后台发起HTTP请求并下载图片。
4.  `SDWebImageManager`，管理类，协调缓存类与下载类。
5.  categories（分类），扩展视图。

## SDImageCache
`SDImageCache`实现了图片的缓存机制，使用`NSCache`作为内存缓存，默认以`com.hackemist.SDWebImageCache.default`为磁盘的缓存命名空间，程序运行后，可以在应用程序的文件夹`Library/Caches/default/com.hackemist.SDWebImageCache.default`下看到一些缓存文件。当然，也可以指定其它命名空间初始化：

```objective-c
- (id)initWithNamespace:(NSString *)ns diskCacheDirectory:(NSString *)directory{
    if ((self = [super init]))
    {
        NSString *fullNamespace = [@"com.hackemist.SDWebImageCache." stringByAppendingString:ns];

        // initialise PNG signature data
        kPNGSignatureData = [NSData dataWithBytes:kPNGSignatureBytes length:8];

        // Create IO serial queue
        _ioQueue = dispatch_queue_create("com.hackemist.SDWebImageCache", DISPATCH_QUEUE_SERIAL);

        // Init default values
        _maxCacheAge = kDefaultCacheMaxCacheAge;

        // Init the memory cache
        _memCache = [[AutoPurgeCache alloc] init];
        _memCache.name = fullNamespace;

        // Init the disk cache
        if (directory != nil) {
            _diskCachePath = [directory stringByAppendingPathComponent:fullNamespace];
        } else {
            NSString *path = [self makeDiskCachePath:ns];
            _diskCachePath = path;
        }

        // Set decompression to YES
        _shouldDecompressImages = YES;

        // memory cache enabled
        _shouldCacheImagesInMemory = YES;

        // Disable iCloud
        _shouldDisableiCloud = YES;

        dispatch_sync(_ioQueue, ^{
            _fileManager = [NSFileManager new];
        });

#if TARGET_OS_IPHONE
        // Subscribe to app events
        [[NSNotificationCenter defaultCenter] addObserver:self
                                                 selector:@selector(clearMemory)
                                                     name:UIApplicationDidReceiveMemoryWarningNotification
                                                   object:nil];

        [[NSNotificationCenter defaultCenter] addObserver:self
                                                 selector:@selector(cleanDisk)
                                                     name:UIApplicationWillTerminateNotification
                                                   object:nil];

        [[NSNotificationCenter defaultCenter] addObserver:self
                                                 selector:@selector(backgroundCleanDisk)
                                                     name:UIApplicationDidEnterBackgroundNotification
                                                   object:nil];
#endif
    }

    return self;
}
```

`SDImageCache`会在系统发出低内存警告时释放内存，并且在程序进入`UIApplicationWillTerminateNotification`时，清理磁盘缓存，清理磁盘的机制是：

1.  删除过期的图片，默认7天过期，可以通过`maxCacheAge`修改过期天数。
2.  如果缓存的数据大小超过设置的最大缓存`maxCacheSize`，则继续删除缓存，优先删除最老的图片，可以通过修改`maxCacheSize`来改变最大缓存大小。

### 缓存中取图片

```objective-c
- (NSOperation *)queryDiskCacheForKey:(NSString *)key done:(SDWebImageQueryCompletedBlock)doneBlock
{
    if (!doneBlock){
        return nil;
    }

    if (!key){
        doneBlock(nil, SDImageCacheTypeNone);
        return nil;
    }

    // First check the in-memory cache...
    UIImage *image = [self imageFromMemoryCacheForKey:key];
    if (image){
        doneBlock(image, SDImageCacheTypeMemory);
        return nil;
    }

    NSOperation *operation = [NSOperation new];
    dispatch_async(self.ioQueue, ^{
        if (operation.isCancelled){
            return;
        }

        @autoreleasepool {
            UIImage *diskImage = [self diskImageForKey:key];
            if (diskImage && self.shouldCacheImagesInMemory){
                NSUInteger cost = SDCacheCostForImage(diskImage);
                [self.memCache setObject:diskImage forKey:key cost:cost];
            }

            dispatch_async(dispatch_get_main_queue(), ^{
                doneBlock(diskImage, SDImageCacheTypeDisk);
            });
        }
    });

    return operation;
}
```

传人的Block定义是：

```objective-c
typedef void(^SDWebImageQueryCompletedBlock)(UIImage *image, SDImageCacheType cacheType);
```

先从内存中取图片，内存中没有的时候再从磁盘中取，通过Block返回取到的图片和获取图片的方式，`SDImageCacheType`的定义如下：

```objective-c
typedef NS_ENUM(NSInteger, SDImageCacheType)
{
    /**
     * The image wasn't available the SDWebImage caches, but was downloaded from the web.
     */
    SDImageCacheTypeNone,
    /**
     * The image was obtained from the disk cache.
     */
    SDImageCacheTypeDisk,
    /**
     * The image was obtained from the memory cache.
     */
    SDImageCacheTypeMemory
};
```

当然，也可能磁盘也没有缓存，此时`doneBlock`中的`diskImage`的值时`nil`，处理方式`doneBlock`将在`SDWebImageManager`讲到。

### 把图片存入缓存

```objective-c
- (void)storeImage:(UIImage *)image recalculateFromImage:(BOOL)recalculate imageData:(NSData *)imageData forKey:(NSString *)key toDisk:(BOOL)toDisk{
    if (!image || !key){
        return;
    }
    // if memory cache is enabled
    if (self.shouldCacheImagesInMemory) {
        NSUInteger cost = SDCacheCostForImage(image);
        [self.memCache setObject:image forKey:key cost:cost];
    }

    if (toDisk) {
        dispatch_async(self.ioQueue, ^ {
            NSData *data = imageData;

            if (image && (recalculate || !data))
            {
#if TARGET_OS_IPHONE
                // We need to determine if the image is a PNG or a JPEG
                // PNGs are easier to detect because they have a unique signature (http://www.w3.org/TR/PNG-Structure.html)
                // The first eight bytes of a PNG file always contain the following (decimal) values:
                // 137 80 78 71 13 10 26 10

                // If the imageData is nil (i.e. if trying to save a UIImage directly or the image was transformed on download)
                // and the image has an alpha channel, we will consider it PNG to avoid losing the transparency
                int alphaInfo = CGImageGetAlphaInfo(image.CGImage);
                BOOL hasAlpha = !(alphaInfo == kCGImageAlphaNone ||
                                  alphaInfo == kCGImageAlphaNoneSkipFirst ||
                                  alphaInfo == kCGImageAlphaNoneSkipLast);
                BOOL imageIsPng = hasAlpha;

                // But if we have an image data, we will look at the preffix
                if ([imageData length] >= [kPNGSignatureData length]){
                    imageIsPng = ImageDataHasPNGPreffix(imageData);
                }

                if (imageIsPng){
                    data = UIImagePNGRepresentation(image);
                }
                else{
                    data = UIImageJPEGRepresentation(image, (CGFloat)1.0);
                }
#else
                data = [NSBitmapImageRep representationOfImageRepsInArray:image.representations usingType: NSJPEGFileType properties:nil];
#endif
            }

            if (data){
                if (![_fileManager fileExistsAtPath:_diskCachePath]){
                    [_fileManager createDirectoryAtPath:_diskCachePath withIntermediateDirectories:YES attributes:nil error:NULL];
                }

                // get cache Path for image key
                NSString *cachePathForKey = [self defaultCachePathForKey:key];
                // transform to NSUrl
                NSURL *fileURL = [NSURL fileURLWithPath:cachePathForKey];

                [_fileManager createFileAtPath:cachePathForKey contents:data attributes:nil];

                // disable iCloud backup
                if (self.shouldDisableiCloud){
                    [fileURL setResourceValue:[NSNumber numberWithBool:YES] forKey:NSURLIsExcludedFromBackupKey error:nil];
                }
            }
        });
    }
}
```

如果需要存入内存，则先存入内存，`toDisk`标识是否存入磁盘，为`Yes`的时候，即要存入磁盘，需要先对图片编码，再存入磁盘。

## SDWebImageDownloader
`SDWebImageDownloader`是下载管理类，是一个单例类，图片的下载在一个`NSOperationQueue`队列中完成。

```objective-c
@property (strong, nonatomic) NSOperationQueue *downloadQueue;
```

默认情况下，队列最多并发数为6，可以通过修改`maxConcurrentOperationCount`来改变最多并发数。

```objective-c
- (id)init{
    if ((self = [super init])){
        ···
        _downloadQueue.maxConcurrentOperationCount = 6;

        ···
    }
    return self;
}

- (void)setMaxConcurrentDownloads:(NSInteger)maxConcurrentDownloads {
    _downloadQueue.maxConcurrentOperationCount = maxConcurrentDownloads;
}
```

下载图片的消息是：

```objective-c
- (id <SDWebImageOperation>)downloadImageWithURL:(NSURL *)url
                                         options:(SDWebImageDownloaderOptions)options
                                        progress:(SDWebImageDownloaderProgressBlock)progressBlock
                                       completed:(SDWebImageDownloaderCompletedBlock)completedBlock;
```

该方法会创建一个`SDWebImageDownloaderOperation`操作队列来执行下载操作，传入的两个Block用于网络下载的回调，`progressBlock`为下载进度回调，`completedBlock`为下载完成回调，回调信息存储在`URLCallbacks`中，为保证只有一个线程操作`URLCallbacks`，`SDWebImageDownloader`把这些操作放入了一个barrierQueue队列中。

```objective-c
_barrierQueue = dispatch_queue_create("com.hackemist.SDWebImageDownloaderBarrierQueue", DISPATCH_QUEUE_CONCURRENT);

- (void)addProgressCallback:(SDWebImageDownloaderProgressBlock)progressBlock andCompletedBlock:(SDWebImageDownloaderCompletedBlock)completedBlock forURL:(NSURL *)url createCallback:(SDWebImageNoParamsBlock)createCallback
{
    ···
    dispatch_barrier_sync(self.barrierQueue, ^{
        BOOL first = NO;
        if (!self.URLCallbacks[url]){
            self.URLCallbacks[url] = [NSMutableArray new];
            first = YES;
        }

        // Handle single download of simultaneous download request for the same URL
        NSMutableArray *callbacksForURL = self.URLCallbacks[url];
        NSMutableDictionary *callbacks = [NSMutableDictionary new];
        if (progressBlock) callbacks[kProgressCallbackKey] = [progressBlock copy];
        if (completedBlock) callbacks[kCompletedCallbackKey] = [completedBlock copy];
        [callbacksForURL addObject:callbacks];
        self.URLCallbacks[url] = callbacksForURL;

        if (first){
            createCallback();
        }
    });
}
```

`SDWebImageDownloader`还提供了两种下载任务调度方式（先进先出和后进先出）：

```objective-c
typedef NS_ENUM(NSInteger, SDWebImageDownloaderExecutionOrder)
{
    /**
     * Default value. All download operations will execute in queue style (first-in-first-out).
     */
    SDWebImageDownloaderFIFOExecutionOrder,

    /**
     * All download operations will execute in stack style (last-in-first-out).
     */
    SDWebImageDownloaderLIFOExecutionOrder
};
```

通过修改`executionOrder`可改变下载方式：

```objective-c
@property (assign, nonatomic) SDWebImageDownloaderExecutionOrder executionOrder;

- (id <SDWebImageOperation>)downloadImageWithURL:(NSURL *)url options:(SDWebImageDownloaderOptions)options progress:(SDWebImageDownloaderProgressBlock)progressBlock completed:(SDWebImageDownloaderCompletedBlock)completedBlock{
        ···
        [wself.downloadQueue addOperation:operation];
        if (wself.executionOrder == SDWebImageDownloaderLIFOExecutionOrder)
        {
            // Emulate LIFO execution order by systematically adding new operations as last operation's dependency
            [wself.lastAddedOperation addDependency:operation];
            wself.lastAddedOperation = operation;
        }
    }];
    ···
}
```

##SDWebImageDownloaderOperation
`SDWebImageDownloaderOperation`是下载操作队列，继承自`NSOperation`，并采用了`SDWebImageOperation`协议，该协议只有一个`cancel`方法。只暴露了一个方法：

```objective-c
- (id)initWithRequest:(NSURLRequest *)request
              options:(SDWebImageDownloaderOptions)options
             progress:(SDWebImageDownloaderProgressBlock)progressBlock
            completed:(SDWebImageDownloaderCompletedBlock)completedBlock
            cancelled:(SDWebImageNoParamsBlock)cancelBlock;
```

该方法的`progressBlock`与`completedBlock`Block与`SDWebImageDownloader`下载管理类对应。

`SDWebImageDownloaderOperation`使用`start`和`done`来控制状态，而不是使用`main`。图片的下载使用`NSURLConnection`，在协议中接收数据并回调Block通知下载进度和下载完成。

##SDWebImageManager
 `SDWebImageManager`是一个单例管理类，负责协调图片缓存和图片下载，是对 `SDImageCache`和`SDWebImageDownloader`的封装。

```objective-c
@property (strong, nonatomic, readwrite) SDImageCache *imageCache;
@property (strong, nonatomic, readwrite) SDWebImageDownloader *imageDownloader;
```

 在一般的使用中，我们并不直接使用`SDImageCache`和`SDWebImageDownloader`，而是使用 `SDWebImageManager`。`SDWebImageManager`的核心方法是：

```objective-c
- (id <SDWebImageOperation>)downloadImageWithURL:(NSURL *)url
                                         options:(SDWebImageOptions)options
                                        progress:(SDWebImageDownloaderProgressBlock)progressBlock
                                       completed:(SDWebImageCompletionWithFinishedBlock)completedBlock
```

代码有点长，这里就不贴出来了，代码控制了图片的缓存和下载的整个流程，同样两个Block与前面的也是一一对应。

##Categories（分类）
在`Categories`目录下实现了多个分类，实现方式都是一致的。使用最多的是`UIImageView+WebCache`，针对`UIImageView`扩展了一些方法。在使用的时候调用的方法是：

```objective-c
- (void)sd_setImageWithURL:(NSURL *)url placeholderImage:(UIImage *)placeholder options:(SDWebImageOptions)options progress:(SDWebImageDownloaderProgressBlock)progressBlock completed:(SDWebImageCompletionBlock)completedBlock
 ```

 该方法依赖于`SDWebImageManager`，从`SDWebImageManager`管理类中获取到图片并刷新显示，至于图片是从缓存中得到的还是从网络上下载的对`UIImageView`是透明的。

 其它的categories就不多说了，当然还可以创建自己的分类。

## 结语
本文是我读SDWebImage的源代码的一点理解，主要集中在图片的下载和缓存，不包括WebP、GIF和图片编码的讨论。涉及到得技术有：

*  ***NSCache***。[NSCache](https://developer.apple.com/library/ios/documentation/Cocoa/Reference/NSCache_Class/index.html)可在内存中临时缓存对象，类似于[NSDictionary](https://developer.apple.com/library/ios/documentation/Cocoa/Reference/Foundation/Classes/NSDictionary_Class/index.html#//apple_ref/occ/cl/NSDictionary)
*  ***NSOperationQueue和NSOperation***。并发编程，更多请参考[并发编程：API 及挑战][1]。
* ***NSMutableURLRequest和NSURLConnection***。网络请求与下载。
*  ***GCD***。如`dispatch_main_sync_safe`、`dispatch_async`、`dispatch_barrier_sync`，更多请参考[并发编程：API 及挑战][1]

本文讨论并不完整，更多的东西还靠以后慢慢发掘。

[1]: http://objccn.io/issue-2-1/
