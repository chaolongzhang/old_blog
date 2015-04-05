---
layout: post
category: [软件开发,软件工艺]
title: 敏捷技能修炼之小舵板之二：分离构造和使用
keywords: 敏捷修炼
tags: [敏捷开发]
---

上一篇：[敏捷技能修炼之小舵板之一：意图导向编程 http://zh.5long.me/2014/Agile-Development-Programming-by-Intention/](http://zh.5long.me/2014/Agile-Development-Programming-by-Intention/)

#前言
软件开发从面向过程向面向对象发展的时候，给开发带来了不少好处，但同时也引出一个新的问题：实例化。比较常用的创建实例的做法是"哪里用到，就在哪里创建"，但这也经常有负面效应。《敏捷技能修炼-敏捷软件开发与设计的最佳实践》一书中提出分离使用实例和构造实例。

#对象隐藏
在一个老系统上添加新功能时，往往是非常复杂的，复杂之处不在于实现功能的复杂，而在于集成的复杂性，老系统耦合度高时更是如此。我们都知道面向对象中隐藏实现是非常重要的。但是，隐藏对象的类型也同样重要。也就是说，我们应该隐藏对象的具体类型，面向接口编程，使调用代码变得更简单，系统更易于维护。

<!--more-->

#分离构造和使用的实例
分离构造和使用就是要把对象的构造是使用分开，我们看下面这段代码：

```java
public class BusinessObject
{
	public void actioMethod()
	{
		//other things
		Service myServiceObject = new Service()
		myServiceObject.doService()
		//other thing
	}
}
```

这段代码中BusinessObject使用了Service的一个实例来实现某个功能，BusinessObject类是Service类的创建者，同时也是Service类的使用者。

使用new就往往意味着无法使用重写（override），也就是说，使用`` new Service()``创建的对象就是Service的一个实例，不会是它的子类对象或其他类。

假设某一天要修改代码，Service不再是一个类，而是一个接口了，有一个它的实现类是Service_Imp1，那么上面代码应该做如下修改：

```java
public class BusinessObject
{
	public void actioMethod()
	{
		//other things
		Service myServiceObject = new Service_Imp1()  //此处有修改
		myServiceObject.doService()
		//other thing
	}
}
```

如果有很多地方都使用new来创建实例，我们就不得不一一修改，代码已经产生了耦合性。

对比上面两段代码，可以看出，只在创建实例的地方不同，如果把创建实例的方法分离出来，那情况就不一样了。我们使用下面一个类创建Service实例

```java
public class ServiceFactory
{
	public static Service getSevice()
	{
		return new Service()
	}
}
```

那么上面第一段代码就可以改成如下代码

```java
public class BusinessObject
{
	public void actioMethod()
	{
		//other things
		Service myServiceObject = ServiceFactory.getSevice();
		myServiceObject.doService()
		//other thing
	}
}
```

Service改成接口是，我们只需修改ServiceFactory类

```java
public class ServiceFactory
{
	public static Service getSevice()
	{
		return new Service_Imp1()
	}
}
```

不管有多少地方使用到实例，我们只需修改一处。更重要的是，假设还有一个Service的实现类Service_Imp2，系统根据不同条件来决定使用哪个类，那么ServiceFactory类就可做如下修改，而不用修改客户端代码

```java
public class ServiceFactory
{
	public static Service getSevice()
	{
		if (选择条件)
		{
			return new Service_Imp1()
		}
		else
		{
			return new Service_Imp2()
		}
	}
}
```

#实际考虑
这样看，隐藏确实给了我们很多好处，但使用一个单独的类来构造实例有时似乎过于奢侈，毕竟我写的类很多都不会有子类或一开始就使用接口，我们应该用最少的类来完成功能。书中推荐的另一种方法是我比较喜欢的，示例代码如下：

```java
public class BusinessObject
{
	public void actioMethod()
	{
		//other things
		Service myServiceObject = Service.getInstance();
		myServiceObject.doService()
		//other thing
	}
}
class Service
{
	private Service() { }
	public static  Service getInstance()
	{
		return new Service();
	}
	public  void doService() { }
}
```

把构造方法放在Service类中，我们不需要额外写一个类来构造实例，同时也能得到ServiceFactory类来构造实例的好处。

假如代码有需要修改成Service变成一个接口，两个实现类：Service_Imp1和Service_Imp2这种情况，可以轻而易举的重新设计构造方法：

```java
abstract class Service
{
	private Service() { }
	public static  Service getInstance()
	{
		return ServiceFactory.getSevice();
	}
	abstract  void doService() { }
}
public class ServiceFactory
{
	public static Service getSevice()
	{
		if (选择条件)
		{
			return new Service_Imp1()
		}
		else
		{
			return new Service_Imp2()
		}
	}
}
```

#分离构造和使用与单例模式
单例模式是我们比较常用的一种设计模式，使用分离构造和使用就能额外的得到一个好处：类可方便的改成单例模式。还以Service类为例

```java
public class Service
{
	private Service() { }
	public static  Service getInstance()
	{
		return new Service();
	}
	public  void doService() { }
}
```

假设系统修改了，Service要使用单例模式，整个系统的代码只需做如下修改：

```java
public class Service
{
	private static Service instance;
	private Service() { }
	public static  Service getInstance()
	{
		if (null == instance)
		{
			instance = new Service();
		}
		return instance;
	}
	public  void doService() { }
}
```

#结语
以上就是关于分离构造和使用的介绍，后续还会继续更新第三个小舵板。
