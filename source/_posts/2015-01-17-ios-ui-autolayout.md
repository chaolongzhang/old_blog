---
layout: post
category: iOS开发
title: iOS开发入门教程外篇之自动布局Auto Layout
keywords: Auto Layout
tags: iOS
---

#前言
随着iOS设备屏幕尺寸的增加，iOS开发者也面临一个Android开发者的问题：屏幕适配。以前只有两种屏幕是还可以用代码去适配，但随着屏幕尺寸的增多，势必会造成代码的堆积，必须寻找其它解决方案，加上苹果的强硬态度，我们终将使用Auto Layout，不管是手写代码还是Interface Builder，使用自动布局都会带来一些好处，当然，前提是能很好地理解Autolayout，否则将会带来意想不到的麻烦。

<!--more-->

#苹果官方指南（Auto Layout Guide）
苹果官方关于Auto Layout的介绍：[Auto Layout Guide](https://developer.apple.com/library/ios/documentation/UserExperience/Conceptual/AutolayoutPG/Introduction/Introduction.html)，看完之后对Auto Layout就有大概的了解了。其中包括：基本概念、在Interface Builder中使用、在代码中使用、处理冲突以及Visual Format Language。

#在Interface Builder使用Auto Layout
在Interface Builder下使用Auto Layout有两篇经典的教程：

*  [Beginning Auto Layout Tutorial in iOS 7: Part 1](http://www.raywenderlich.com/50317/beginning-auto-layout-tutorial-in-ios-7-part-1)
*  [Beginning Auto Layout Tutorial in iOS 7: Part 2](http://www.raywenderlich.com/50319/beginning-auto-layout-tutorial-in-ios-7-part-2)

其中第一篇有中文翻译版：[开始iOS 7中自动布局教程(一)](http://www.cocoachina.com/industry/20131203/7462.html)

看完这两篇基本就入门了，加上自己的琢磨，就可以在Interface Builder下使用Auto Layout了。

#在代码中使用Auto Layout
用代码写UI或者运行时要改变界面布局就得在代码中使用Auto Layout，其主要就是添加Constraints。主要用到 [NSLayoutConstraint](https://developer.apple.com/library/ios/documentation/AppKit/Reference/NSLayoutConstraint_Class/index.html#//apple_ref/occ/cl/NSLayoutConstraint)的两个方法：`constraintsWithVisualFormat:options:metrics:views:`和`constraintWithItem:attribute:relatedBy:toItem:attribute:multiplier:constant:`。为什么会有两个方法，苹果解释说前一个使用visual format language，是一种可视化较好的表达方式，但是不能处理视图间固定的比例（如：imageView.width = 2 * imageView.height），所以就加了后者。

关于Auto Layout在代码中的用法请参考苹果官方文档：[Working with Auto Layout Programmatically](https://developer.apple.com/library/ios/documentation/UserExperience/Conceptual/AutolayoutPG/AutoLayoutinCode/AutoLayoutinCode.html#//apple_ref/doc/uid/TP40010853-CH11-SW1)

#Visual Format Language(简称VFL)
说它是一种语言有点夸张，其实就是一种标记方式。Visual Format Language的句法官方也有说明，参考[Visual Format Language](https://developer.apple.com/library/ios/documentation/UserExperience/Conceptual/AutolayoutPG/VisualFormatLanguage/VisualFormatLanguage.html#//apple_ref/doc/uid/TP40010853-CH3-SW1)，就是把在Interface Builder的操作转化成字符串。比如：`[button]-[textfield]`就表示视图button和视图textfield的水平间距保存标准距离（8），如下图：

![standardSpace](/assets/images/2015/standardSpace.png)

#Auto Layout的扩展
Auto Layout的使用并不是那么的浪漫，管理布局的约束有时候也会很复杂。当然，也不必悲观，在[GitHub](https://github.com/)上已有不少Auto Layout的扩展了，如：[Masonry](https://github.com/Masonry/Masonry)、[Snappy](https://github.com/Masonry/Snap)、[PureLayout](https://github.com/smileyborg/PureLayout)、[Cartography](https://github.com/robb/Cartography)等，这些能帮你很多忙。

Masonry是一个轻量级的布局框架，代码及例子托管在GitHub上[https://github.com/Masonry/Masonry](https://github.com/Masonry/Masonry)，Masonry把Auto Layout冗长的代码变动简短，并且可读性更高，当然，错误也更少，值得一试。

其它三个框架还未研究，暂不讨论。

#结语
本文简要的介绍了Auto Layout及它的扩展，详细的教程及说明请打开链接查看。