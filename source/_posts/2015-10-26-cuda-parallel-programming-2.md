---
layout: post
categories: CUDA
title: 【CUDA并行程序设计系列（2）】CUDA简介及CUDA初步编程
keywords: CUDA,GPGPU,并行程序
tags: [CUDA,GPGPU]
---

## 前言

CUDA并行程序设计系列是本人在学习CUDA时整理的资料，内容大都来源于对《CUDA并行程序设计：GPU编程指南》、《GPU高性能编程CUDA实战》和[CUDA Toolkit Documentation](http://docs.nvidia.com/cuda/index.html)的整理。通过本系列整体介绍CUDA并行程序设计。内容包括GPU简介、CUDA简介、环境搭建、线程模型、内存、原子操作、同步、流和多GPU架构等。

本系列目录：

*  [【CUDA并行程序设计系列（1）】GPU技术简介](http://zh.5long.me/2015/cuda-parallel-programming-1/)
*  [【CUDA并行程序设计系列（2）】CUDA简介及CUDA初步编程](http://zh.5long.me/2015/cuda-parallel-programming-2/)
*  [【CUDA并行程序设计系列（3）】CUDA线程模型](http://zh.5long.me/2015/cuda-parallel-programming-3/)
*  [【CUDA并行程序设计系列（4）】CUDA内存](http://zh.5long.me/2015/cuda-parallel-programming-4/)
*  [【CUDA并行程序设计系列（5）】CUDA原子操作与同步](http://zh.5long.me/2015/cuda-parallel-programming-4/)
*  [【CUDA并行程序设计系列（6）】CUDA流与多GPU](http://zh.5long.me/2015/cuda-parallel-programming-5/)
*  [关于CUDA的一些学习资料](http://zh.5long.me/2015/cuda-learning/)

本文对CUDA进行简单介绍，并通过实例代码演示怎么编写在GPU上运行的代码，最后写一段代码来查询本机GPU的设备参数。

<!--more-->

## CUDA C简介
CUDA C是NVIDIA公司设计的一种编程语言，用于在GPU上得编写通用计算程序，目前也叫“ CUDA C and C++”，或者“CUDA C/C++”。CUDA C是C语言的扩展，所以使用起来和C语言类似。当然，CUDA现在已经不局限于C语言了，在NVIDIA ZONE的[LANGUAGE SOLUTIONS](https://developer.nvidia.com/language-solutions)就明确支持多种语言和开发环境，如：C++、Python、Java、.Net、OpenACC、OpenCL等，操作系统也支持Linux、Mac OS、Windows。当然，前提是电脑至少配备一个支持CUDA的GPU，在[NVIDIA官方](https://developer.nvidia.com/cuda-gpus)可以查看自己电脑的显卡是否支持CUDA。

## GPU计算能力
在查看某个显卡是否支持CUDA时，还会看到一个计算能力（Compute Capability）的参数。正如不同的CPU有着不同的功能和指令集，对于支持CUDA的GPU也同样如此。NVIDIA将GPU支持的各种功能统称为计算能力。硬件的计算能力是固定的。不同计算能力具有一定差别，如，在计算能力1.0版本不支持全局内存上的原子操作。更高计算能力的GPU是低计算能力的超级，所以计算能力2.0支持的功能在3.0也全部支持。可以看到，目前GPU最高的计算能力已经达到5.3（Tegra X1）。

## 环境搭建
环境搭建比较简单，在[NVIDIA开发官网](https://developer.nvidia.com/cuda-downloads)选择对应的平台下载最新版（目前是7.5）并安装就行了。在[CUDA Toolkit Online Documentation](http://docs.nvidia.com/cuda/index.html)有详细的安装教程。

## Hello,World!
新建一个文件“hello_world.cu"

```c
int main( void )
{
    printf( "Hello, World!\n" );
    return 0;
}
```

除了后缀”.cu”表示CUDA文件，这段代码甚至不需要任何解释，下面编译并运行：

```
nvcc hello_world.cu
./a.out
```

输出：

```
Hello, World!
```

我们使用nvcc命令编译，这将使用CUDA C编译器来编译这段代码。

修改一下这段代码：

```c
__global__  void kernel( void )
 {
}

int main( void )
{
    kernel<<<1,1>>>();
    printf( "Hello, World!\n" );
    return 0;
}
```

编译运行结果还是一样的，可以看到CUDA C为标准C增加了`__global__`修饰符，这个修饰符告诉编译器，函数应该编译为在设备（Device）而不是主机(Host)上运行，CUDA把GPU称为设备（Device），把CPU称为主机（Host），而在GPU设备上执行的函数称为核函数（Kernel），在设备上运行的函数需要增加`__global__`或`__device__`修饰符。

调用核函数增加了<<<1,1>>>修饰符，其它没有任何变化。当然，这段代码GPU并没有做什么实际的工作，下面让GPU实际做点事情。

## GPU上的加法
修改上面的代码：

```c
__global__  void add( int a, int b, int *c )
{
    *c = a + b;
}

int main( void )
{
    int c;
    int *dev_c;
    cudaMalloc( &dev_c, sizeof(int) ) ;

    add<<<1,1>>>( 2, 7, dev_c );

    cudaMemcpy( &c, dev_c, sizeof(int),cudaMemcpyDeviceToHost );
    printf( "2 + 7 = %d\n", c );
    cudaFree( dev_c );

    return 0;
}
```

主机调用核函数`add`做加法运算，函数`add`将在GPU上运行。需要注意的是，主机和设备在物理上处于不同的位置，使用了不同的内存，核函数不能直接使用主机上存储的数据，同样主机也不能直接使用设备上存储的数据，数据需要在主机和设备之间传输。

在设备上分配内存使用`cudaMalloc()`，该函数类似`malloc()`。在执行完`add`函数后，计算结果保存在设备的内存上，还需要使用`cudaMemcpy()`传输到主机内存上，参数`cudaMemcpyDeviceToHost`表示从设备传输到主机。显然，`cudaMemcpyHostToDevice`表示从主机传输到设备，`cudaMemcpyDeviceToDevice`表示从设备传输到另一个设备。

至此，一个完整地CUDA C代码已经实现，下面来写一段代码查询显卡的设备参数。

## 查询设备
在进行CUDA并行编程之前，对自己PC机的GPU设备性能及相关信息的了解是很有必要的，下面写一段代码来查询设备参数信息。

查询设备时会用的几个函数：

1.  `cudaGetDeviceCount()`，获得CUDA设备的数量，一台PC可能会有多个CUDA设备，可以通过这个函数查询。
2.  `cudaGetDeviceProperties()`，通过设备编号查询设备属性，设备编号从0开始。设备属性保存在`cudaDeviceProp`结构体中，具体结构可查看[cudaDeviceProp Struct Reference](http://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html)。

完整代码如下：

```c
int main(void)
{
      cudaDeviceProp prop;
      int count;

      cudaGetDeviceCount(&count);
      printf("cuda device count: %d\n", count);

      for (int i = 0; i < count; ++i)
      {
            cudaGetDeviceProperties(&prop, i);

            printf ("    ---  General Information for device %d ------\n", i);
            printf ("Name: %s\n", prop.name);
            printf ( "Compute capability: %d.%d\n", prop.major, prop.minor );
            printf ( "Clock rate: %d \n", prop.clockRate );
            printf( "Device copy overlap: ");
            if (prop.deviceOverlap)
            {
                  printf ( "Enabled\n");
            }
            else
            {
                  printf ( "Disabled\n" );
            }
            printf ( "Kernel execiton timeout: " );
            if (prop.kernelExecTimeoutEnabled )
            {
                  printf ( "Enabled\n" );
            }
            else
            {
                  printf ( "Disabled\n" );
            }
            printf ("integrated:");
            if (prop.integrated)
            {
                  printf("true\n");
            }
            else
            {
                  printf("false\n");
            }
            printf ( "--- Memory Information for device %d ----\n", i);
            printf ( "Total global mem: %ld\n", prop.totalGlobalMem );
            printf ( "Total constant Mem: %ld\n", prop.totalConstMem );
            printf ("Max mem pitch: %ld\n", prop.memPitch );
            printf ( "Texture Alignment: %ld\n", prop.textureAlignment );
            printf ( "  --- MP Information for device %d ---\n", i );
            printf ( "Multiprocessor count: %d\n", prop.multiProcessorCount );
            printf ( "Shared mem per mp: %ld\n", prop.sharedMemPerBlock );
            printf ("Registers per mp: %d\n", prop.regsPerBlock );
            printf ("Threads in warp: %d\n", prop.warpSize );
            printf ("Max threads per block: %d\n", prop.maxThreadsPerBlock );
            printf ("Max thread dimensions: ( %d %d %d )\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2] );
            printf ("Max grid dimensions: ( %d %d %d )", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2] );
            printf ("\n");
      }

      return 0;
}
```

运行这段代码，就可以知道自己PC机配备GPU的具体信息，这对以后写代码是很重要的。

## 参考文献

*  库克. CUDA并行程序设计. 机械工业出版社, 2014.
* 桑德斯. GPU高性能编程CUDA实战. 机械工业出版社, 2011.
*  [CUDA C Programming Guide](http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)
*  [CUDA Toolkit Documentation](http://docs.nvidia.com/cuda/index.html)