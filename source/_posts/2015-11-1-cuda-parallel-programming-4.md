---
layout: post
category: [CUDA]
title: 【CUDA并行程序设计系列（4）】CUDA内存
keywords: CUDA,GPGPU,并行程序
tags: [CUDA,GPGPU]
---

##前言

CUDA并行程序设计系列是本人在学习CUDA时整理的资料，内容大都来源于对《CUDA并行程序设计：GPU编程指南》、《GPU高性能编程CUDA实战》和[CUDA Toolkit Documentation](http://docs.nvidia.com/cuda/index.html)的整理。通过本系列整体介绍CUDA并行程序设计。内容包括GPU简介、CUDA简介、环境搭建、线程模型、内存、原子操作、同步、流和多GPU架构等。

本系列目录：

*  [【CUDA并行程序设计系列（1）】GPU技术简介](http://zh.5long.me/2015/cuda-parallel-programming-1/)
*  [【CUDA并行程序设计系列（2）】CUDA简介及CUDA初步编程](http://zh.5long.me/2015/cuda-parallel-programming-2/)
*  [【CUDA并行程序设计系列（3）】CUDA线程模型](http://zh.5long.me/2015/cuda-parallel-programming-3/)
*  [【CUDA并行程序设计系列（4）】CUDA内存](http://zh.5long.me/2015/cuda-parallel-programming-4/)
*  [【CUDA并行程序设计系列（5）】CUDA原子操作与同步](http://zh.5long.me/2015/cuda-parallel-programming-4/)
*  [【CUDA并行程序设计系列（6）】CUDA流与多GPU](http://zh.5long.me/2015/cuda-parallel-programming-5/)
*  [关于CUDA的一些学习资料](http://zh.5long.me/2015/cuda-learning/)

本章将介绍CUDA的内存结构，通过实例展示寄存器和共享内存的使用。

##CUDA内存结构
GPU的内存结构和CPU类似，但也存在一些区别，GPU的内存中可读写的有：寄存器(registers)、Local memory、共享内存（shared memory）和全局内存（global memory），只读的有：常量内存（constant memory）和纹理内存（texture memory）。

<!--more-->

CUDA Toolkit Document给出的的内存结构如下图所示：

![CUDA内存结构](/assets/images/2015/cuda-4-1.png)

每个线程都有独立的寄存器和Local memory，同一个block的所有线程共享一个共享内存，全局内存、常量内存和纹理内存是所有线程都可访问的。全局内存、常量内存和纹理内存对程序的优化有特殊作用。

##寄存器
与CPU不同，GPU的每个SM（流多处理器）有成千上万个寄存器，在[GPU技术简介](http://zh.5long.me/2015/cuda-parallel-programming-1/)中已经提到，SM类似于CPU的核，每个SM拥有多个SP（流处理器），所有的工作都是在SP上处理的，GPU的每个SM可能有8~192个SP，这就意味着，SM可同时运行这些数目的线程。

寄存器是每个线程私有的，并且GPU没有使用寄存器重命名机制，而是致力于为每一个线程都分配真实的寄存器，CUDA上下文切换机制非常高效，几乎是零开销。当然，这些细节对程序员是完全透明的。

和CPU一样，访问寄存器的速度是非常快的，所以应尽量优先使用寄存器。无论是CPU还是GPU，通过寄存器的优化方式都会使程序的执行速度得到很大提高。

举一个例子：

```c
for (int i = 0; i < size; ++i)
{
      sum += array[i];
}
```

`sum`如果存于内存中，则需要做size次读/写内存的操作，而如果把`sum`设置为局部变量，把最终结果写回内存，编译器会将其放入寄存器中，这样只需1次内存写操作，将大大节约运行时间。

##Local memory
Local memory和寄存器类似，也是线程私有的，访问速度比寄存器稍微慢一点。事实上，是由编译器在寄存器全部使用完的时候自动分配的。在优化程序的时候可以考虑减少block的线程数量以使每个线程有更多的寄存器可使用，这样可减少Local memory的使用，从而加快运行速度。

##共享内存
共享内存允许同一个block中的线程读写这一段内存，但线程无法看到也无法修改其它block的共享内存。共享内存缓冲区驻留在物理GPU上，所以访问速度也是很快的。事实上，共享内存的速度几乎在所有的GPU中都一致（而全局内存在低端显卡的速度只有高端显卡的1/10），因此，在任何显卡中，除了使用寄存器，还要更有效地使用共享内存。

共享内存的存在就可使运行线程块中的多个线程之间相互通信。共享内存的一个应用场景是线程块中多个线程需要共同操作某一数据。考虑一个矢量点积运算的例子：

（x<sub>1</sub>, x<sub>2</sub>, x<sub>3</sub>, x<sub>4</sub> ) * (y<sub>1</sub>, y<sub>2</sub>, y<sub>3</sub>, y<sub>4</sub>) = x<sub>1</sub>y<sub>1</sub> + x<sub>2</sub>y<sub>2</sub> + x<sub>3</sub>y<sub>3</sub> + x<sub>4</sub>y<sub>4</sub>


和矢量加法一样，矢量点积也可以在GPU上并行计算，每个线程将两个相应的元素相乘，然后移到下两个元素，线程每次增加的索引为总线程的数量，下面是实现这一步的代码：

```c
const int N = 33 * 1024;
const int threadsPerBlock = 256;

__global__ void dot( float *a, float *b, float *c )
{
    __shared__ float cache[threadsPerBlock];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIndex = threadIdx.x;

    float   temp = 0;
    while (tid < N)
    {
        temp += a[tid] * b[tid];
        tid += blockDim.x * gridDim.x;
    }
    cache[cacheIndex] = temp;
}
```

CUDA C使用`__shared__`修饰符申明共享内存的变量。在每个线程中分别计算相应元素的乘积之和，并保存在共享内存变量`cache`对应的索引中，可以看出，如果只有一个block，那么所有线程结束后，对`cache`求和就是最终结果。当然，实际会有很多个block，所以需要对所有block中的cache求和，由于共享内存在block之间是不能访问的，所以需要在各个block中分部求和，并把部分和保存在数组中，最后在CPU上求和。block中分部求和代码如下：

```c
__global__ void dot( float *a, float *b, float *c ) {
    __shared__ float cache[threadsPerBlock];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIndex = threadIdx.x;

    float   temp = 0;
    while (tid < N) {
        temp += a[tid] * b[tid];
        tid += blockDim.x * gridDim.x;
    }

    cache[cacheIndex] = temp;

    //同步
    __syncthreads();

   //分部求和
    int i = blockDim.x/2;
    while (i != 0) {
        if (cacheIndex < i)
            cache[cacheIndex] += cache[cacheIndex + i];
        __syncthreads();
        i /= 2;
    }

    if (cacheIndex == 0)
        c[blockIdx.x] = cache[0];
}
```

`__syncthreads()`是线程同步函数，调用这个函数确保在线程块中所有的线程都执行完`__syncthreads()`之前的代码，在执行后面的代码，当然，这会损失一定性能。

当执行`__syncthreads()`之后的代码，我们就能确定`cache`已经计算好了，下面只需要对`cache`求和就可以了，最简单的就是用一个`for`循环计算。但是，这相当只有一个线程在起作用，线程块其它线程都在做无用功，

使用规约运行是一个更好地选择，即每个线程将`cache`中的两个值相加起来，然后结果保存会`cache`中，规约的思想如下图所示。

![规约算法图示](/assets/images/2015/cuda-4-2.png)

按这种方法，每次将会使数据减少一半，只需执行log2(threadsPerBlock)个步骤后，就能得到`cache`中所有值的总和。

最后使用如下代码将结果保存在`c`中：

```c
if (cacheIndex == 0)
        c[blockIdx.x] = cache[0];
```

这是因为只有一个值需要写入，用一个线程来操作就行了，如果不加`if`，那么每个线程都将执行一次写内存操作，浪费大量的运行时间。

最后，只需要在CPU上把`c`中的值求和就得到了最终结果。下面给出完整代码：

```c
#include <stdio.h>

const int N = 33 * 1024;
const int threadsPerBlock = 256;
const int blocksPerGrid = (N + threadsPerBlock -1) / threadsPerBlock;

__global__ void dot( float *a, float *b, float *c )
{
    __shared__ float cache[threadsPerBlock];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIndex = threadIdx.x;

    float   temp = 0;
    while (tid < N)
    {
        temp += a[tid] * b[tid];
        tid += blockDim.x * gridDim.x;
    }

    cache[cacheIndex] = temp;

    //同步
    __syncthreads();

    //规约求和
    int i = blockDim.x/2;
    while (i != 0)
     {
        if (cacheIndex < i)
        {
            cache[cacheIndex] += cache[cacheIndex + i];
        }

        __syncthreads();
        i /= 2;
    }

    if (cacheIndex == 0)
    {
        c[blockIdx.x] = cache[0];
    }
}

int  main(int argc, char const *argv[])
{
    float   *a, *b, *partial_c;
    float   *dev_a, *dev_b, *dev_partial_c;

    a = (float*)malloc( N*sizeof(float) );
    b = (float*)malloc( N*sizeof(float) );
    partial_c = (float*)malloc( blocksPerGrid*sizeof(float));

    cudaMalloc(&dev_a, N*sizeof(float));
    cudaMalloc(&dev_b, N*sizeof(float));
    cudaMalloc(&dev_partial_c, blocksPerGrid*sizeof(float));

    for (int i=0; i < N; ++i)
    {
        a[i] = i;
        b[i] = i * 2;
    }

    cudaMemcpy(dev_a, a, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, N*sizeof(float), cudaMemcpyHostToDevice);

    dot<<<blocksPerGrid,threadsPerBlock>>>( dev_a, dev_b, dev_partial_c );

    cudaMemcpy(partial_c, dev_partial_c, blocksPerGrid*sizeof(float), cudaMemcpyDeviceToHost);

    int c = 0;
    for (int i=0; i < blocksPerGrid; ++i)
    {
        c += partial_c[i];
    }

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_partial_c);

    free(a);
    free(b);
    free(partial_c);

    return 0;
}
```

##常量内存
常量内存，通过它的名字就可以猜到它是只读内存。常量内存其实只是全局内存的一种虚拟地址形式，并没有特殊保留的常量内存块。内存的大小为64KB。常量内存可以在编译时申明为常量内存，使用修饰符`__constant__`申明，也可以在运行时通过主机端定义为只读内存。常量只是从GPU内存的角度而言的，CPU在运行时可以通过调用`cudaCopyToSymbol`来改变常量内存中的内容。

##全局内存
GPU的全局内存之所以是全局内存，主要是因为GPU与CPU都可以对它进行写操作，任何设备都可以通过PCI-E总线对其进行访问。在多GPU系统同，GPU之间可以不通过CPU直接将数据从一块GPU卡传输到另一块GPU卡上。在调用核函数之前，使用`cudaMemcpy`函数就是把CPU上的数据传输到GPU的全局内存上。

##纹理内存
和常量内存一样，纹理内存也是一种只读内存，在特定的访问模式中，纹理内存能够提升程序的性能并减少内存流量。纹理内存最初是为图形处理程序而设计，不过同样也可以用于通用计算。由于纹理内存的使用非常特殊，有时使用纹理内存是费力不讨好的事情。因此，对于纹理内存，只有在应用程序真正需要的时候才对其进行了解。主要应该掌握全局内存、共享内存和寄存器的使用。

##参考文献

*  库克. CUDA并行程序设计. 机械工业出版社, 2014.
* 桑德斯. GPU高性能编程CUDA实战. 机械工业出版社, 2011.
*  [CUDA C Programming Guide](http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)
*  [CUDA Toolkit Documentation](http://docs.nvidia.com/cuda/index.html)
*  R\. Couturier, Ed., Designing Scientific Applications on GPUs, CRC Press, 2013.


