---
layout: post
category: [dotNet,程序设计]
title:  混合语言编程：C#使用原生的Directx和OpenGL
keywords: 混合语言编程
tags: [C#,VC++]
---

本文要说的是混合C#和C/C++语言编程，在C#的Winform和WPF下使用原生的Direct和OpenGL进行绘图。

由于项目需要做一些图形展示，所以就想到了使用Directx和OpenGL来绘图，但项目准备使用C#来开发（大家比较熟悉C#），在网上看了相关的资料，有一些第三方的控件可用，试用了下，一运行就占了几百M的内存，而且

<!--more-->

也不知道是否稳定，教程也少，还不如直接使用原生的。在网上看的Directx和OpenGL的教程基本上都是C/C++的，找了很久也就找到相关介绍，只能自己研究下。

我以前做过C#和C++混合语言编程相关的东西，在C++实现一些C#不好实现的功能，C#动态调用DLL文件，所以也想到了用C++的代码来控制Winform控件的绘画，这样就可实现用Direct和OpenGL在Winform的控件上绘画了。

由于我对Direct和OpenGL都不熟悉，没有这方面的编程经验，只能去瞎折腾，下面分别说说最近在Directx和OpenGL怎么试验的。

Directx:

之前没学过Directx，拿了同学的代码来看，也是雾里看花啊，不过有一点启示了我，在初始化的时候，要传入一个句柄去创建设备（CreateDevice），通常都是传入窗口的设备，我想如果传入一个控件的句柄，那所有的绘画都将在这个控件上实现，因为控件也是继承自Window的。而Winform的控件在底层的实现应该和WIN32是一样的。这样的话只要把Winform的控件的句柄传入C++代码进行初始化，那么绘画的结果将显示在这个控件上。结果一试，还真行。关键代码如下：

```C++
extern "C" _declspec(dllexport) HRESULT InitD3D( HWND hWnd );

extern "C" _declspec(dllexport) VOID Render();
```

在InitD3D传入控件的句柄进行初始化，C#再调用Render进行绘画，以下是C#代码：

```C#
HWND handle = this.button1.Handle;
InitD3D(handle);
private void Draw()
  {
         for (; ; )
            {
                Render();
            }
}
 ```

 效果图：
  ![Directx](/assets/images/2014/20130911123438.png)

  OpenGL:

查看了OpenGL的相关教程(推荐http://www.yakergong.net/nehe/)，OpenGL是通过RC来执行的，创建RC时就必须指定一个DC，还要设置DC的像素格式。每个线程有且只能拥有一个RC。

如果在初始化OpenGL的绘画环境时传入一个Winform的控件句柄，再通过这个句柄取到HDC，就可使用这个HDC来创建RC，这样OpenGL的绘画环境就准备好了，并且这个RC关联到Winform的控件上。

在给制前，先为当前线程选择RC（之前通过HDC创建的），再进行绘制，这样绘制的结果将显示在这个Winform控件上。

关键代码如下：

```C++
extern "C" _declspec(dllexport) void Init( HWND hWnd);

extern "C" _declspec(dllexport) void Render();

void Init(HWND hWnd)
{
  	int PixelFormat;
  	int bits = 16;
  	hDC = GetDC(hWnd);
  	static  PIXELFORMATDESCRIPTOR pfd=           // pfd Tells Windows How We Want Things To Be
  	{
     	sizeof(PIXELFORMATDESCRIPTOR),              // Size Of This Pixel Format Descriptor
     	1,                                                         　　 // Version Number
     	PFD_DRAW_TO_WINDOW |                        // Format Must Support Window
     	PFD_SUPPORT_OPENGL |                          // Format Must Support OpenGL
     	PFD_DOUBLEBUFFER,                               // Must Support Double Buffering
     	PFD_TYPE_RGBA,                                     // Request An RGBA Format
     	bits,                                                        // Select Our Color Depth
     	0, 0, 0, 0, 0, 0,                                       // Color Bits Ignored
     	0,                                                         // No Alpha Buffer
     	0,                                                         // Shift Bit Ignored
     	0,                                                        // No Accumulation Buffer
     	0, 0, 0, 0,                                           // Accumulation Bits Ignored
     	16,                                                    // 16Bit Z-Buffer (Depth Buffer) 
     	0,                                                     // No Stencil Buffer
     	0,                                                     // No Auxiliary Buffer
     	PFD_MAIN_PLANE,                         // Main Drawing Layer
     	0,                                                  // Reserved
     	0, 0, 0                                          // Layer Masks Ignored
  	};
  	PixelFormat=ChoosePixelFormat(hDC,&pfd);
  	SetPixelFormat(hDC,PixelFormat,&pfd);
  	hRC=wglCreateContext(hDC);
}

void Render()
{
  	wglMakeCurrent(hDC, hRC);
  	draw();
  	SwapBuffers(hDC);
  	wglMakeCurrent(NULL, NULL);
}
```

```C#
Init(this.pictureBox1.Handle);

//开始绘制
var timer = new System.Timers.Timer();
timer.Interval = 100;
	timer.Elapsed += (tsender, te) =>
               	{
                   	Render();
               	};
timer.AutoReset = true;
timer.Enabled = true;
```

  ![OpenGL](/assets/images/2014/20130911125045.png)

  WPF:WPF的控件没有句柄，但是可以换个思路，把绘画代码封装在Winform控件上，在WPF使用自定义的Winform控件。
  ![WPF](/assets/images/2014/20130911130045.png)
  


