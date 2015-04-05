---
layout: post
category: [程序设计,算法]
title: Python实现k-means算法
tags: [Python]
---

k-means算法描述如下：

—输入：期望得到的簇的数目k，n个对象的数据D
—输出：k个簇的集合

<!--more-->

—方法：
（1）选择k个对象作为初始的簇的质心

（2）repeat

（3）计算对象与各个簇的质心的距离，将对象划分到距离其最近的簇

（4）重新计算每个新簇的均值

（5）Until簇的质心不再变化

Python代码：

—假设：给定如下要进行聚类的对象：
 {2，4，10，12，3，20，30，11，25}，k = 2


```python
# coding=utf-8  
''''' 
计算两点之间的距离 
'''  
def distance(n1, n2):  
   	d = n1 - n2  
   	if d < 0:  
       	d = -d  
   	return d  
      
''''' 
计算一串数的中点 
'''  
def center(nums):  
   	sum = 0  
   	for item in nums:  
       	sum = sum + item  
   	return sum / len(nums)  
  
''''' 
k-means算法 
'''  
def main():  
   	n = 2  
   	nums = [2, 4, 10, 12, 3, 20, 30, 11, 25]  
      
   	c1 = nums[0]            #类1的中心点，初始为第一个数  
   	c2 = nums[1]            #类2的中心点，初始为第二个数  
   	new_c1 = c1-1           #一次聚类后类1新的中心点  
   	new_c2 = c2 - 1         #一次聚类后类2新的中心点  
   	g1 = []                 #类1  
   	g2 = []                 #类2  
   	isFirst = True  
   	step = 1  
  
   	#k-means核心算法  
   	while not c1 == new_c1 or not c2 == new_c2:  
       	print "----------------------------------------------"  
       	print  "step %d"  % step   
       	step = step + 1  
  
       	#不是第一次运行，使用新的中心点  
       	if not isFirst:  
           	c1 = new_c1  
           	c2 = new_c2  
       	isFirst = False  
  
       	#通过距离划分族  
       	for item in nums :  
           	d1 = distance(item, new_c1)  
           	d2 = distance(item, new_c2)  
           	if d1 <= d2 :  
               	g1.append(item)  
           	else:  
               	g2.append(item)  
  
       	#计算新类的中点  
       	new_c1 = center(g1)  
       	new_c2 = center(g2)  
  
       	g1.sort()  
       	g2.sort()  
       	print  "group 1 is",  g1  
       	print  "group 2 is",  g2  
  
       	#清空g1, g2  
       	del g1[:]  
       	del g2[:]  
       	print "----------------------------------------------"  
  
#开始运行  
main()  
```

初学数据挖掘，很多原理都不懂，写个算法练练手。