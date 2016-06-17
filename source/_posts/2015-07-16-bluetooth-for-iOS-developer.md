---
layout: post
categories: iOS
title: iOS开发之蓝牙4.0 BLE开发
keywords: iOS开发,蓝牙4.0,BLE
tags: [iOS, 蓝牙4.0]
---

## 前言
物联网时代的到来，智能硬件层出不穷。蓝牙低功耗技术（BLE，Bluetooth Low Energy）使得蓝牙4.0的应用越来越广泛，比如小米手环就是使用蓝牙4.0来传输数据。

在蓝牙4.0之前，iOS的蓝牙功能作为私有，开发者只能使用蓝牙开发联机游戏而不能和第三方蓝牙设备通信，当时要开发一个用iPhone通过蓝牙来控制硬件的APP是件非常复杂的事情。好在蓝牙4.0之后，苹果开放了蓝牙4.0开发接口。而TI公司也推出了两款蓝牙芯片：CC2540和CC2541。之后就涌现了一大批基于蓝牙4.0的硬件设备和APP。如[Stick-N-Find](http://v.youku.com/v_show/id_XNTMzMDM4MTI0_rss.html)和耐克数字运动手环 NIKE+。随后中国也出现了一大批类似设备，小米、华为都出了手环。

本人差不多也是在那时接触蓝牙4.0，做一个蓝牙通信的APP。在当时资料还不是那么丰富的情况下，整个开发过程也是个探索过程。由于最近又需要做一个蓝牙通信的APP，于是整理了之前写的代码，重新写了一份蓝牙通信的代码（之前写的他太乱了 : D）。

本文介绍iOS下开发蓝牙4.0通信程序的流程以及使用到相关的库。

<!--more-->

## 相关资料

*  iOS SDK蓝牙4.0相关的组件为[CoreBluetooth](https://developer.apple.com/library/ios/documentation/CoreBluetooth/Reference/CoreBluetooth_Framework/index.html)，官方文档<https://developer.apple.com/library/ios/documentation/CoreBluetooth/Reference/CoreBluetooth_Framework/index.html>
*  蓝牙4.0苹果官方开发指南：<https://developer.apple.com/library/ios/documentation/NetworkingInternetWeb/Conceptual/CoreBluetooth_concepts/AboutCoreBluetooth/Introduction.html>


## 关键类
蓝牙4.0相关的组件为CoreBluetooth，所以要使用之前先引用'CoreBluetooth/CoreBluetooth.h'，其中有两个重要的类：

*  'CBCentralManager'：蓝牙的控制中心，通过它来获知蓝牙状态、扫描周边蓝牙设备、发起连接等。
*  'CBPeripheral'：和周边蓝牙设备对应的一个对象，每一个蓝牙设备对应一个CBPeripheral对象。通过它乡对应的蓝牙设备发送数据、读取数据、获取RSSI值等。

'CBCentralManager'和'CBPeripheral'通过协议来交互，对应的两个协议为：

*  'CBCentralManagerDelegate'：CBCentralManager'的协议。
*  'CBPeripheralDelegate'：CBPeripheral'的协议。

比如当'CBCentralManager'发起扫描周边设备时，当发现了一个设备后就会调用CBCentralManagerDelegate'的'- (void)centralManager:(CBCentralManager*)central didDiscoverPeripheral:(CBPeripheral*)peripheral advertisementData:(NSDictionary*)advertisementData RSSI:(NSNumber*)RSSI'，通过实现该方法就可做相应地处理。

关于协议，请参考[iOS开发入门教程之Objective-C · 协议（Protocols）](http://zh.5long.me/2015/learning-ios-oc-5/)。

以下列出这两个协议常用的方法：

**CBCentralManagerDelegate**

```objective-c
@required
/*
本机设备（iPhone/iPad）蓝牙状态改变
*/
- (void)centralManagerDidUpdateState:(CBCentralManager *)central;

@optional
/*
扫描到了一个蓝牙设备peripheral
*/
- (void)centralManager:(CBCentralManager *)central didDiscoverPeripheral:(CBPeripheral *)peripheral advertisementData:(NSDictionary *)advertisementData RSSI:(NSNumber *)RSSI;

/*
成功连接蓝牙设备peripheral
*/
- (void)centralManager:(CBCentralManager *)central didConnectPeripheral:(CBPeripheral *)peripheral;

/*
连接蓝牙设备peripheral失败
*/
- (void)centralManager:(CBCentralManager *)central didFailToConnectPeripheral:(CBPeripheral *)peripheral error:(NSError *)error;

/*
蓝牙设备peripheral已断开
*/
- (void)centralManager:(CBCentralManager *)central didDisconnectPeripheral:(CBPeripheral *)peripheral error:(NSError *)error;
```

**CBPeripheralDelegate**

```objective-c
@optional
/*
蓝牙设备peripheral的RSS值改变
*/
- (void)peripheralDidUpdateRSSI:(CBPeripheral *)peripheral error:(NSError *)error NS_DEPRECATED(NA, NA, 5_0, 8_0);

/*
从peripheral成功读取到services
*/
- (void)peripheral:(CBPeripheral *)peripheral didDiscoverServices:(NSError *)error;

/*
从peripheral的service成功读取到characteristics
*/
- (void)peripheral:(CBPeripheral *)peripheral didDiscoverCharacteristicsForService:(CBService *)service error:(NSError *)error;

/*
从蓝牙设备peripheral接收到数据
*/
- (void)peripheral:(CBPeripheral *)peripheral didUpdateValueForCharacteristic:(CBCharacteristic *)characteristic error:(NSError *)error;

/*
乡蓝牙设备peripheral成功写入数据，写数据时需要以CBCharacteristicWriteWithResponse才会调用
*/
 - (void)peripheral:(CBPeripheral *)peripheral didWriteValueForCharacteristic:(CBCharacteristic *)characteristic error:(NSError *)error;
```

## 蓝牙通信方式
蓝牙通信方式为`service+characteristic`，这和网络通信的IP+端口类似，`service`可理解为IP，`characteristic`可理解为端口号。`service`和`characteristic`都是一个数字编号，一般用16进制表示，如`0xFFE0`、`0xFFE1`。

需要注意的是发送方和接收方必须是在相同的`service+characteristic`下才能正常发送和接收数据，如发送方在`service`为`0xFFE0`，`characteristic`为`0xFFE1`下发送信息，接收方也应该在`service`为`0xFFE0`，`characteristic`为`0xFFE1`下接收信息，否则是接收不到数据的。

一个蓝牙设备可以有多个`service`，一个`service`也可以有多个`characteristic`。不同的`service`和`characteristic`组合可形成多个通信通道。

还有一点需要提醒的是，由于苹果的限制，**iOS SDK并没有提供获取蓝牙设备MAC地址的方法，所以一般还是不要用MAC，使用`CBPeripheral`的`UUID`和`name`代替**。


## 蓝牙通信开发流程
在此就已经了解了iOS开发蓝牙通信所需要的库了，下面进入开发流程。蓝牙通信流程一般是：

1.  初始化
1.  扫描周边的蓝牙设备。
2.  连接某一设备。
3.  发送/接收数据。
4.  断开连接

当然，所有这些操作都默认iPhone/iPad已开启蓝牙功能：D

### 初始化
初始化一个`CBCentralManager`。

```objective-c
self.centralManager = [[CBCentralManager alloc] initWithDelegate:self queue:nil];
```

### 扫描周边的蓝牙设备
调用`CBCentralManager`的`scanForPeripheralsWithServices`就能扫描周边的设备。

```objective-c
[self.centralManager scanForPeripheralsWithServices:nil options:nil];
```

当扫描到一个蓝牙设备时，会调用`CBCentralManagerDelegate`的`centralManager:(CBCentralManager *) didDiscoverPeripheral:(CBPeripheral *) advertisementData:(NSDictionary *) RSSI:(NSNumber *)`方法，通过实现该协议方法，就可做相应地处理。

```objective-c
- (void)centralManager:(CBCentralManager*)central didDiscoverPeripheral:(CBPeripheral*)peripheral advertisementData:(NSDictionary*)advertisementData RSSI:(NSNumber*)RSSI
{
    NSLog(@"find new device:%@", peripheral.name);
}
```

### 连接某一设备
通过调用`CBCentralManager`的`connectPeripheral: options:`方法就可向蓝牙设备发起连接，连接成功或失败会分别调用`CBCentralManagerDelegate`的`centralManager:(CBCentralManager *) didConnectPeripheral:(CBPeripheral *)`和`centralManager:(CBCentralManager *) didFailToConnectPeripheral:(CBPeripheral *) error:(NSError *)`方法。

如果连接成功，还应该读取`peripheral`的`service`和`characteristic`。

```objective-c
[self.centralManager connectPeripheral:peripheral options:nil];

- (void)centralManager:(CBCentralManager*)central didConnectPeripheral:(CBPeripheral*)peripheral
{
    peripheral.delegate = self;
    [peripheral discoverServices:nil];
}

- (void)peripheral:(CBPeripheral *)peripheral didDiscoverServices:(NSError *)error
{
    [peripheral discoverCharacteristics:nil forService:service];
}
```

### 发送数据
发送数据只需要在指定的`service`和`characteristic`组合下发送即可，如果是以`CBCharacteristicWriteWithResponse`模式发送，发送完后还会调用`CBPeripheralDelegate`的`peripheral:(CBPeripheral *) didWriteValueForCharacteristic:(CBCharacteristic *) error:(NSError *)`，实现该协议方法可判断发送是否成功。以`CBCharacteristicWriteWithoutResponse`模式则不会有回调。

```objective-c
[peripheral writeValue:data forCharacteristic:characteristic type:CBCharacteristicWriteWithResponse];

- (void)peripheral:(CBPeripheral *)peripheral didWriteValueForCharacteristic:(CBCharacteristic *)characteristic error:(NSError *)error
{
    if (error)
    {
        NSLog(@"%@", error);
    }
}
```

### 接收数据
当接收到数据后，`CBCentralManager`会调用协议`CBCentralManagerDelegate`的`peripheral:(CBPeripheral *) didUpdateValueForCharacteristic:(CBCharacteristic *) error:(NSError *)error`方法，实现该方法就可获取接收到得数据。

```objective-c
- (void)peripheral:(CBPeripheral *)peripheral didUpdateValueForCharacteristic:(CBCharacteristic *)characteristic error:(NSError *)error
{
    NSString *msg=[[NSString alloc] initWithData:characteristic.value  encoding:NSUTF8StringEncoding];
    NSLog(@"message:%@", msg);
}
```

### 断开连接
通过调用`CBCentralManager`的`cancelPeripheralConnection:`即可断开蓝牙连接，成功断开后，会调用`CBCentralManagerDelegate`的`- (void)centralManager:(CBCentralManager *) didDisconnectPeripheral:(CBPeripheral *) error:(NSError *)`方法。

```objective-c
[self.centralManager cancelPeripheralConnection:peripheral];

- (void)centralManager:(CBCentralManager *)central didDisconnectPeripheral:(CBPeripheral *)peripheral error:(NSError *)error
{
}
```

## 结语
本文介绍了开发iOS下蓝牙通信APP的必要知识及开发流程，给出了一个iOS蓝牙通信的框架。关于各个函数的详细说明，还需要查阅官方文档。结合官方文档，相信读者可以开发出一个蓝牙通信的模型来。


