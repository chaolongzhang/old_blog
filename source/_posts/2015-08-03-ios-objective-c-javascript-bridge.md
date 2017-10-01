---
layout: post
categories: iOS
date: 2015-08-03 00:00:30
title: iOS开发之Objective-C(Swift)与JavaScript交互·WebViewJavascriptBridge使用篇
keywords: Objective-C(Swift)与JavaScript交互，WebViewJavascriptBridge
tags: iOS
---

## 前言
在iOS原生应用程序加载网页来实现部分界面或功能已不是什么稀奇的事了。很多应用都使用了HTML+CSS+Javascript+Native APP的方式来开发，如Fackbook、微信和支付宝等。采用这种开发模式具有明显的好处：

-  跨平台，iOS、Android都可使用，write once run anywhere。
-  方便实现复杂的界面，使用前台技术可实现很炫的界面。
-  升级方便，修改程序后不用审核即可更新用户程序。

当然，这也面临着一个问题，那就是怎么实现原生程序和Web页面交互，也即Objective-C(Swift)与JavaScript交互。iOS SDK中的`UIWebView`和`WKWebView`都可以方便的实现Objective-C(Swift)与JavaScript交互。同时，GitHub上的[WebViewJavascriptBridge][1]使得交互变得更为轻松。本文将介绍WebViewJavascriptBridge的使用方法。

<!--more-->

## 准备

在GitHub上下载[WebViewJavascriptBridge][1]，里面包含WebViewJavascriptBridge源代码和使用例子。使用前把文件夹WebViewJavascriptBridge拖入项目即可。

WebViewJavascriptBridge可应用于`UIWebView`和`WKWebView`。[WKWebView][2]是iOS 8引入的一个新的框架——WebKit.

## 使用

使用WebViewJavascriptBridge一般会有以下流程：

1.  初始化
2.  注册可调用的函数
3.  发送消息、调用函数

当然，还必须加载一个网页。

初始化的时候需要提供一个函数/Block，当发送消息给另一个语言时（如Objective-C调用[bridge send:(id)data]向JavaScript发送消息），会执行这个函数/Block。

同时，只有注册的函数才能供另一个语言调用。

### Objective-C
导入头文件#import "WebViewJavascriptBridge.h"

申明属性 @property WebViewJavascriptBridge* bridge;

如果使用`WKWebView`，则使用如下方式：

导入头文件#import "WKWebViewJavascriptBridge.h"

申明属性 @property WKWebViewJavascriptBridge* bridge;

**使用UIWebView/WKWebView初始化**

```objective-c
//webView可以是UIWebView或WKWebView
self.bridge = [WebViewJavascriptBridge bridgeForWebView:webView handler:^(id data, WVJBResponseCallback responseCallback)
{
    NSLog(@"Received message from javascript: %@", data);
    responseCallback(@"Right back atcha");
}];
```

在JavaScript中的bridge调用send方法后，会执行handler的block。


**注册JavaScript可调用的函数**

```objective-c
[self.bridge registerHandler:@"testObjcCallback" handler:^(id data, WVJBResponseCallback responseCallback)
{
        NSLog(@"testObjcCallback called: %@", data);
        responseCallback(@"Response from testObjcCallback");
}];
```
在JavaScript中的bridge调用`testObjcCallback`方法后，会执行该handler的block。

**调用JavaScript方法**

```objective-c
//无回调直接调用
[self.bridge callHandler:@"testJavascriptHandler" data:@{ @"foo":@"before ready" }];

//有回调调用
id data = @{ @"greetingFromObjC": @"Hi there, JS!" };
[self.bridge callHandler:@"testJavascriptHandler" data:data responseCallback:^(id response)
 {
    NSLog(@"testJavascriptHandler responded: %@", response);
}];
```

当使用有回调调用方法时，会把`block`传入JavalockScript方法，JavaScript代码可执行该`block`（JavaScript的调用方法见下方）。

**向JavaScript发送消息**

```objective-c
[self.bridge send:@"A string sent from ObjC to JS" responseCallback:^(id response)
{
    NSLog(@"sendMessage got response: %@", response);
}];
```

发送消息与调用方法类似，相当调用一个特殊函数（该函数在JavaScript代码中初始化时设置，见下方）。

**加载网页**

```objective-c
//加载本地网页
NSString* htmlPath = [[NSBundle mainBundle] pathForResource:@"web" ofType:@"html"];
NSString* appHtml = [NSString stringWithContentsOfFile:htmlPath encoding:NSUTF8StringEncoding error:nil];
NSURL *baseURL = [NSURL fileURLWithPath:htmlPath];
[self.webview loadHTMLString:appHtml baseURL:baseURL];

//加载远程网页
NSURL *url = [NSURL URLWithString:urlStr];
NSURLRequest *request = [NSURLRequest requestWithURL:url];
[self.webview loadRequest:request];
```

### JavaScript

**添加监听**

```javascript
function connectWebViewJavascriptBridge(callback)
    {
        if (window.WebViewJavascriptBridge)
        {
            callback(WebViewJavascriptBridge)
        }
        else
        {
            document.addEventListener('WebViewJavascriptBridgeReady', function(event)
                                    {
                                      callback(WebViewJavascriptBridge)
                                    }, false)
        }
    }

connectWebViewJavascriptBridge(init);
```


**初始化**

初始化要在添加监听之后进行。初始化时需要传入一个函数，当Objective-C代码发送消息调用`[bridge send:(id)data]`或`[bridge send:(id)data responseCallback:(WVJBResponseCallback)responseCallback]`时会执行此函数。

```javascript
function recieved(message, responseCallback)
{
    log('JS got a message', message)
    var data =
    {
        'Javascript Responds': 'Wee!'
    }
    log('JS responding with', data)
    responseCallback(data)
}

function init(bridge)
{
        bridge.init(recieved);
}
```

**注册Objective-C可调用的函数**

```javascript
function testJavascript(data, responseCallback)
{
    log('ObjC called testJavascriptHandler with', data)
    var responseData =
    {
        'Javascript Says': 'Right back atcha!'
    }
    log('JS responding with', responseData)
    responseCallback(responseData)
}

bridge.registerHandler('testJavascriptHandler', testJavascript);
```

注册后在Objective-C代码中就可调用该函数（调用方法见上述）。

**调用Objective-C方法**

```javascript
function callObjC()
{
    gbridge.callHandler('testObjcCallback',
                       {
                            'foo': 'bar'
                       },
                       function(response)
                       {
                            log('JS got response', response)
                       });
}
```

Objective-C会执行以名称`testObjcCallback`注册的block。

**发送消息**
与调用函数类似，会调用Objective-C中初始化时传入的block。

```javascript
function sendMessage()
    {
        var data = 'Hello from JS button'
        gbridge.send(data, function(responseData)
                    {
                        log('JS got response', responseData)
                    })
    }
```

## 结语
可以说WebViewJavascriptBridge的使用是如此简单，以至不需要太多的笔墨。当然WebViewJavascriptBridge也是很强大的，它能结合WebApp和Native APP并充分发挥各自的优势。


[1]: https://github.com/marcuswestin/WebViewJavascriptBridge
[2]: https://developer.apple.com/library/prerelease/ios/documentation/WebKit/Reference/WKWebView_Ref/index.html
