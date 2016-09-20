---
title: 使用Python写CUDA程序
date: 2016-09-20 21:40:37
categories: CUDA
keywords: CUDA,GPGPU,并行程序
tags: [CUDA,GPGPU]
---


使用Python写CUDA程序有两种方式：
* [Numba][1]
* [PyCUDA][2]

[numbapro](https://docs.continuum.io/numbapro/)现在已经不推荐使用了，功能被拆分并分别被集成到[accelerate][3]和Numba了。

## 例子
### numba

Numba通过及时编译机制（JIT）优化Python代码，Numba可以针对本机的硬件环境进行优化，同时支持CPU和GPU的优化，并且可以和Numpy集成，使Python代码可以在GPU上运行，只需在函数上方加上相关的指令标记，如下所示：

<!--more-->

```python
import numpy as np 
from timeit import default_timer as timer
from numba import vectorize

@vectorize(["float32(float32, float32)"], target='cuda')
def vectorAdd(a, b):
    return a + b

def main():
    N = 320000000

    A = np.ones(N, dtype=np.float32 )
    B = np.ones(N, dtype=np.float32 )
    C = np.zeros(N, dtype=np.float32 )

    start = timer()
    C = vectorAdd(A, B)
    vectorAdd_time = timer() - start

    print("c[:5] = " + str(C[:5]))
    print("c[-5:] = " + str(C[-5:]))

    print("vectorAdd took %f seconds " % vectorAdd_time)

if __name__ == '__main__':
    main()
```

### PyCUDA

PyCUDA的内核函数（kernel）其实就是使用C/C++编写的，通过动态编译为GPU微码，Python代码与GPU代码进行交互，如下所示：

```python
import pycuda.autoinit
import pycuda.driver as drv
import numpy as np
from timeit import default_timer as timer

from pycuda.compiler import SourceModule
mod = SourceModule("""
__global__ void func(float *a, float *b, size_t N)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N)
  {
    return;
  }
  float temp_a = a[i];
  float temp_b = b[i];
  a[i] = (temp_a * 10 + 2 ) * ((temp_b + 2) * 10 - 5 ) * 5;
  // a[i] = a[i] + b[i];
}
""")

func = mod.get_function("func")   

def test(N):
    # N = 1024 * 1024 * 90   # float: 4M = 1024 * 1024

    print("N = %d" % N)

    N = np.int32(N)
    
    a = np.random.randn(N).astype(np.float32)
    b = np.random.randn(N).astype(np.float32)   
    # copy a to aa
    aa = np.empty_like(a)
    aa[:] = a
    # GPU run
    nTheads = 256
    nBlocks = int( ( N + nTheads - 1 ) / nTheads )
    start = timer()
    func(
            drv.InOut(a), drv.In(b), N,
            block=( nTheads, 1, 1 ), grid=( nBlocks, 1 ) )
    run_time = timer() - start  
    print("gpu run time %f seconds " % run_time)    
    # cpu run
    start = timer()
    aa = (aa * 10 + 2 ) * ((b + 2) * 10 - 5 ) * 5
    run_time = timer() - start  

    print("cpu run time %f seconds " % run_time)  

    # check result
    r = a - aa
    print( min(r), max(r) )

def main():
  for n in range(1, 10):
    N = 1024 * 1024 * (n * 10)
    print("------------%d---------------" % n)
    test(N)

if __name__ == '__main__':
    main()
```

## 对比

numba使用一些指令标记某些函数进行加速（也可以使用Python编写内核函数），这一点类似于OpenACC，而PyCUDA需要自己写kernel，在运行时进行编译，底层是基于C/C++实现的。通过测试，这两种方式的加速比基本差不多。但是，numba更像是一个黑盒，不知道内部到底做了什么，而PyCUDA就显得很直观。因此，这两种方式具有不同的应用：
* 如果只是为了加速自己的算法而不关心CUDA编程，那么直接使用numba会更好。
* 如果为了学习、研究CUDA编程或者实验某一个算法在CUDA下的可行性，那么使用PyCUDA。
* 如果写的程序将来要移植到C/C++，那么就一定要使用PyCUDA了，因为使用PyCUDA写的kernel本身就是用CUDA C/C++写的。

## 参考文献

1. Numba. <http://numba.pydata.org/>
2. PyCUDA. <https://mathema.tician.de/software/pycuda/>

[1]: http://numba.pydata.org/
[2]: https://mathema.tician.de/software/pycuda/
[3]: https://docs.continuum.io/accelerate/


