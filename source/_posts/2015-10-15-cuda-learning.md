---
layout: post
category: [CUDA]
title: 关于CUDA的一些学习资料
keywords: CUDA,GPGPU,并行程序
tags: [CUDA,GPGPU]
---

##前言
最近看了一些关于CUDA方面的资料，并粗略地浏览了两本关于CUDA和GPGPU的书（[《GPGPU编程技术：从GLSL、CUDA到OpenCL》](http://book.douban.com/subject/6538230/)和[《CUDA并行程序设计：GPU编程指南》](http://book.douban.com/subject/25813577/)），对于CUDA目前本人也是处于正在入门的阶段。在此汇编CUDA的学习资料，方便以后的学习。


##关于GPGPU
所谓GPGPU，就是把强大的图形处理器应用于日益复杂的计算工作。究其本质，图形处理单元（Graphices Processor Unit, GPU）是为高速图形处理而设计的，它具有天然的并行性，程序并行运行自然要比串行快很多。

<!--more-->

早期的GPGPU编程比较痛苦，经典的GPGPU需要使用通用的图形API，比如OpenGL，而图形API并不是为了通用程序而设计的，所以需要使用大量的技巧，让通用计算的数据“看起来像幅图”。除了难以掌握外，更重要的是不能根据硬件特性获取硬件应有的性能，于是就产生了CUDA。

##关于CUDA
CUDA全名计算统一设备架构（Compute Unified Device Architecture, CUDA），东家是英伟达（NVIDIA）。CUDA是专门为通用计算而设计的（Tesla卡甚至连图形输出都没有，专为计算而设计），CUDA采用一种简单的数据并行模型，再结合编程模型，从而无需操纵复杂的图形基元，CUDA使得GPU看起来和别的可编程设备一样。

##CUDA书籍
*  [《GPGPU编程技术:从GLSL、CUDA到OpenCL》](http://book.douban.com/subject/6538230/)
*  [《CUDA并行程序设计-GPU编程指南》](http://book.douban.com/subject/25813577/)
*  [《CUDA Programming》](http://book.douban.com/subject/11681504/)
*  [《GPU高性能编程CUDA实战》](http://book.douban.com/subject/5917267/)
*  [《CUDA by Example》](http://book.douban.com/subject/4754651/)
*  [《CUDA专家手册:GPU编程权威指南》](http://book.douban.com/subject/25976234/)
*  [《The CUDA Handbook》](http://book.douban.com/subject/7952510/)
*  [《CUDA Application Design and Development》](http://book.douban.com/subject/6999088/)
*  [《GPU精粹2·高性能图形芯片和通用计算编程技巧》](http://book.douban.com/subject/2144796/)
*  [《高性能CUDA应用设计与开发》](http://book.douban.com/subject/20445361/)
*  [《GPU高性能运算之CUDA》](http://book.douban.com/subject/4102289/)
*  [《CUDA范例精解》](http://book.douban.com/subject/5338392/)
*  [《基于GPU的多尺度离散模拟并行计算》](http://book.douban.com/subject/3544392/)

更多书籍查看豆列[GPGPU&CUDA](http://www.douban.com/doulist/42577001/)。

##CUDA Online
*  英伟达官方培训网站：[英文](https://developer.nvidia.com/cuda-training), [中文](https://cudazone.nvidia.cn/cuda-training/)
*  NVIDIA DEVELOPER ZONE：<https://developer.nvidia.com/>
*  GPU技术会议（GTC）资料：<http://on-demand-gtc.gputechconf.com/gtcnew/on-demand-gtc.php>
*  GTC：<http://www.gputechconf.com/>
* GTC中国：<http://www.nvidia.cn/object/gpu-technology-conference-cn.html>
* CUDA zone：<https://developer.nvidia.com/cuda-zone>
*  CUDA zone 中国：<https://cudazone.nvidia.cn/>
*  Learning Cuda By Shane Cook：<http://www.cudadeveloper.com/learningcuda.php>


##在线课程
*  ECE-498AL：<https://courses.engr.illinois.edu/ece498al>
*  Stanford CS193G：[ iTunes U](https://itunes.apple.com/cn/itunes-u/programming-massively-parallel/id384233322?mt=10)、<https://code.google.com/p/stanford-cs193g-sp2010/>
*  Wisconsin ME964：<http://sbel.wisc.edu/Courses/ME964/2011/>
*  EE171并行计算机架构：<http://www.nvidia.com/object/cudau_ucdavis>

##其它
CUDA作为并行计算架构，通常会和其它并行架构一起配合使用。如OpenMP、OpenACC、MPI，OpenACC是NVIDIA明确支持的。

*  OpenMP
*  OpenACC:[官网](http://www.openacc.org/)、[NVIDIA OpenAcc指令](http://www.nvidia.cn/object/openacc-gpu-directives-cn.html)、[NVIDIA OPENACC TOOLKIT](https://developer.nvidia.com/openacc)
*  MPI
*  [ZeroMQ](http://zeromq.org/)
*  [OpenCL](https://developer.nvidia.com/opencl)

##结语
本文主要整理收集了CUDA的相关书籍和资料，以供学习时方便查阅，整理可能不完整。

##参考文献
1.  库克. CUDA并行程序设计[M]. 机械工业出版社, 2014.
2.  仇德元. GPGPU编程技术. 机械工业出版社, 2011.



