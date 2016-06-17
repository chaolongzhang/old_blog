---
layout: post
categories: iOS开发入门教程
title: iOS开发入门教程 · 楔子
keywords: iOS
tags: [iOS]
---

## 前言
鉴于同学们学习iOS开发的热情，本人决定写一个iOS开发入门教程系列。本教程假定读者已经学过C语言，并且有一定的面向对象的基础。若读者还不具备这两个条件，请自行学习，C语言就不推荐教材了，学习面向对象的话推荐学习Python，有一本书叫《像计算机科学家一样思考Python》写等很好，推荐学习。

## 内容
本系列教程包括两部分：Objective-c 和 iOS基础开发。

本教程旨在帮助读者入门，所涉及到的东西并不全面，读者还需配合其他书籍或教程来学习，读者可以从以下资源中来选取部分资料来学习（本文只列举了部分资料，更多资料请自行查找）：

<!--more-->

*  《Objective-C开发经典教程》
*  《Objective-C基础教程》
*  《Objective-C编程》
*  《Objective-C程序设计》
*  《iOS 7应用开发入门经典》
*  《iOS编程》
*  《精通iOS开发》
*  [斯坦福大学公开课：iPad和iPhone应用开发(iOS5)](http://v.163.com/special/opencourse/ipadandiphoneapplication.html)
*  [斯坦福大学公开课：iOS 7应用开发](http://v.163.com/special/opencourse/ios7.html)
*  [苹果官方马上着手开发 iOS 应用程序(Start Developing iOS Apps Today)](https://developer.apple.com/library/ios/referencelibrary/GettingStarted/RoadMapiOSCh/index.html)

本系列教程内容如下：

*  [楔子](http://zh.5long.me/2014/learning-ios-preface/)
*  [面向过程的Objective-c](http://zh.5long.me/2014/learning-ios-oc-1/)
*  [面向对象的Objective-c](http://zh.5long.me/2014/learning-ios-oc-2/)
*  [常用的数据类型](http://zh.5long.me/2015/learning-ios-oc-3/)
*  [引用计数](http://zh.5long.me/2015/learning-ios-oc-4/)
*  [协议](http://zh.5long.me/2015/learning-ios-oc-5/)
*  [OS应用](http://zh.5long.me/2015/ios-first-app/)
* [跋](http://zh.5long.me/2015/ios-epilogue/)

本教程将以做一个密码生成器的iOS应用来贯穿全部内容，该应用提供一个非常简单的功能：按一定的规则生成密码。

## 环境搭建
### 主流方案
开发iOS应用需要使用XCode，XCode只能运行在苹果操作系统上，所以，需要一台装有苹果操作系统的电脑。

在苹果操作系统下搭建开发环境只需下载XCode，安装即可。

### 替代方案
由于苹果操作系统一般只能运行在苹果电脑上，若暂时还没有苹果电脑，可以有以下替代方案：

*  装黑苹果，难度大，而且不一定装的上，只对爱折腾电脑的人推荐
*  虚拟机，不一定能装上
*  使用[GNUstep](http://www.gnustep.org/windows/installer.html)，只能编译Objective-c，不能开发iOS应用，参考[windows下 Codeblocks 搭建 Objective-c 开发环境](http://blog.csdn.net/hjj1006340261/article/details/11488327)

学习Objective-c语言时推荐使用GNUsetep，但学习iOS应用开发时就必须在Mac OS下。

更新中。。。。
