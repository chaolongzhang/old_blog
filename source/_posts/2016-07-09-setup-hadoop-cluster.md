---
title: 搭建Hadoop集群
date: 2016-07-09 12:01:55
categories: 云计算
tags: [Hadoop]
---

## 前言

本文是我在一次搭建Hadoop环境的记录，记录了在搭建Hadoop集群时的规划、软件版本、系统环境配置、Zookeeper和Hadoop的安装以及运行。

**注意：本文需要有一定的Linux基础和虚拟机使用基础。**

## 规划
在虚拟机中安装3个Ubuntu 14.04搭建Hadoop集群，3台主机网络配置如下：

|  主机   | hostname |    IP           | Netmask |  Gateway      |
| :----: | :------: | :-------------: | :-----: |  :----------: |
| master | master   | 192.168.138.130 | 24      | 192.168.138.2 |
| slave1 | slave1   | 192.168.138.131 | 24      | 192.168.138.2 |
| slave2 | slave2   | 192.168.138.132 | 24      | 192.168.138.2 |

软件配置如下：

- JDK: Oracle JDK 7
- zookeeper: 3.4.6
- Hadoop: 1.1.2

<!--more-->

目录配置：

|       名称       |          目录               |
| :-------------: | :------------------------: |
| zookeeper安装目录 | /opt/hadoop/zookeeper      |
| zookeeper数据目录 | /opt/hadoop/zookeeper_data |
| Hadoop安装目录    | /opt/hadoop/hadoop         |
| Hadoop tmp 目录  | /usr/hadoop/tmp            |
| Hadoop data目录  | /usr/hadoop/data           |

用户配置：

* Group: hadoop
* User: hadoop

以下操作如无特殊说明，全部在hadoop用户下执行，如果hadoop用户无sudo权限，需要修改/etc/sudoers文件，为hadoop用户添加权限：

![设置用户权限](/assets/images/2016/hadoop-1.png)

如果hadoop用户没有权限执行操作，可用切换至root用户操作，操作完后需要修改文件或目录的权限或所有者：

```shell
chmod
chown -R
```

## 虚拟机安装

(略) vmware/virtualbox都可以。

## 安装Ubuntu
（略）安装语言选择英语，安装的时候设置hadoop用户，安装完成后执行如下操作：

- 更换源
- 更新`sudo apt-get update & sudo apt-get upgrade`

## 环境准备
### 安装JDK

- 添加ppa

```shell
sudo add-apt-repository ppa:webupd8team/java
sudo apt-get update
```

- 安装oracle-java-installera

```shell
sudo apt-get install oracle-java7-installer
```

- 设置系统默认jdk

```shell
sudo update-java-alternatives -s java-7-oracle
```

-  设置JAVA_HOME

先查看JAVA安装路径：`sudo update-alternatives --config java`，输出如下：

![查看JAVA环境](/assets/images/2016/hadoop-2.png)

修改`/etc/environment`文件，添加 `JAVA_HOME="/usr/lib/jvm/java-7-oracle”`

其中`“/usr/lib/jvm/java-7-oracle”`是根据前面的输出结果来确定的。

最后执行如下命令，使设置生效：

```shell
source /etc/environment
```

关于在Ubuntu下安装JDK可参考[How To Install Java on Ubuntu with Apt-Get](https://www.digitalocean.com/community/tutorials/how-to-install-java-on-ubuntu-with-apt-get)

### 安装SSH

```shell
sudo apt-get install openssh-server
```

## 安装Zookeeper

- 拷贝zookeeper-3.4.6.tar.gz到`/opt/hadoop`目录下，然后执行如下命令：

```shell
tar -zxvf zookeeper-3.4.6.tar.gz
mv zookeeper-3.4.6 zookeeper
```

- 修改配置文件/etc/profile

```shell
echo '# zookeeper configuration ' >> /etc/profile
echo 'export ZOOKEEPER_HOME=/opt/hadoop/zookeeper' >> /etc/profile
echo 'export PATH=$ZOOKEEPER_HOME/bin:$PATH' >> /etc/profile
echo 'export CLASSPATH=$ZOOKEEPER_HOME/lib' >> /etc/profile
```

- 是配置文件生效

```shell
source /etc/profile
```

- 创建zookeeper数据目录

```shell
mkdir /opt/hadoop/zookeeper_data
```

- 创建zookeeper配置文件，在`zookeeper/conf`目录下执行命令：

```shell
cp zoo_sample.cfg zoo.cfg
```

- 修改zoo.cfg文件

```shell
dataDir=/opt/hadoop/zookeeper_data

# zookeeper cluster mathines
server.1=192.168.138.130:2888:3888
server.2=192.168.138.131:2888:3888
server.3=192.168.138.132:2888:3888
```

## 安装Hadoop
### 安装master

- 拷贝hadoop-1.1.2.tar.gz到`/opt/hadoop`目录下，然后执行如下命令:

```shell
tar -zxvf hadoop-1.1.2.tar.gz
mv hadoop-1.1.2 hadoop
```

- 测试，Hadoop 解压后即可使用。输入如下命令来检查 Hadoop 是否可用，成功则会显示 Hadoop 版本信息。

```shell
bin/hadoop version
```

- 修改/etc/profile，添加

```shell
# hadoop
export HADOOP_INSTALL=/opt/hadoop/hadoop
export PATH=$PATH:$HADOOP_INSTALL/bin
```

- 修改hadoop/conf/core-site.xml文件如下：

```xml
<configuration>
    <property>
        <name>hadoop.tmp.dir</name>
                <value>/usr/hadoop/tmp</value>
    </property>
        <property>
        <name>fs.default.name</name>
        <value>hdfs://master:9001</value>
    </property>
</configuration>
```

- 修改hadoop/conf/hdfs-site.xml文件如下：

```xml
<configuration>
    <property>
        <name>dfs.data.dir</name>
        <value>/usr/hadoop/data</value>
    </property>
    <property>
        <name>dfs.replication</name>
        <value>2</value>
    </property>
</configuration>
```

`dfs.replication=2`表明有两台datanode，数据目录dfs.data.dir设置为`/usr/hadoop/data`，因此需要创建`/usr/hadoop`目录，命令如下：

```shell
sudo mkdir /usr/hadoop
sudo chown -R hadoop:hadoop
```

- 修改hadoop/conf/mapred-site.xml文件如下：

```xml
<configuration>
    <property>
        <name>mapred.job.tracker</name>
        <value>master:9002</value>
    </property>
</configuration>
```

- 修改masters文件内容如下：

```
master
```

- 修改slaves文件如下：

```
slave1
slave2
```

- 修改hosts

```
192.168.138.130 master
192.168.138.131 slave1
192.168.138.132 slave2
```

### 克隆虚拟机slaves

完整克隆master到slave1和slave2

### 修改网络配置

- 设置master、slave1和slave2的IP地址、子网掩码、网关，配置内容见规划。
- 修改主机名/etc/hostname，配置内容见规划。

## 创建zookeeper所需的ID文件

- master机器上执行（其中/opt/hadoop/zookeeper_data是zookeepers数据目录，根据前面的设置修改）：

```shell
echo '1' >> /opt/hadoop/zookeeper_data/myid
```

- slave1 机器上执行：

```shell
echo ‘2' >> /opt/hadoop/zookeeper_data/myid
```

- slave2机器上执行：

```shell
echo ‘3' >> /opt/hadoop/zookeeper_data/myid
```

## 建立各机器 SSH 免密码登录

- 在master、slave1、slave2机器上执行命令：

```shell
cd ~/.ssh
ssh-keygen -t rsa
```

- 在slave1机器上执行命令：

```shell
cd ~/.ssh
scp id_rsa.pub  hadoop@master:~/.ssh/id_rsa.pub.slave1
```

- 在slave1机器上执行命令：

```shell
cd ~/.ssh
scp id_rsa.pub  hadoop@master:~/.ssh/id_rsa.pub.slave1
```

- 在slave2机器上执行命令：

```shell
cd ~/.ssh
scp id_rsa.pub  hadoop@master:~/.ssh/id_rsa.pub.slave2
```

- 在master机器上执行：

```shell
cd ~/.ssh
cat id_rsa.pub >> authorized_keys
cat id_rsa.pub.slave1 >> authorized_keys
cat id_rsa.pub.slave2 >> authorized_keys
```

- 在master机器上同步`authorized_keys`同步至slave1和slave2：

```shell
scp ~/.ssh/authorized_keys hadoop@slave1:~/.ssh/
scp ~/.ssh/authorized_keys hadoop@slave2:~/.ssh/
```

- 测试ssh，分别在master、slave1、slave2上执行ssh登录另两台机器，如在master机器ssh登录另两台机器的命令如下：

```shell
ssh hadoop@slave1
ssh hadoop@slave2
```

## 启动
### 启动zookeeper

由于之前设置了PATH，所以可以直接执行以下命令启动：

```shell
zkServer.sh start
```

`zkServer.sh`文件位于`zookeeper/bin/`目录下。
分别在3台机器上执行这条命令启动zookeeper。

### 启动Hadoop

- 在master执行`start-all.sh`启动全部。

## 验证
### 验证zookeeper

查看状态可以执行以下命令：

```shell
zkServer.sh status
```

输出为：

```shell
JMX enabled by default
Using config: /opt/hadoop/zookeeper/bin/../conf/zoo.cfg
Mode: follower
```

### 验证Hadoop

- 命令`jps`，master输出：

```
SecondaryNameNode
JobTracker
NameNode
Jps
```

slave输出：

```
DataNode
Jps
TaskTracker
```

- 打开浏览器，输入<http://master:50070/>，可看到namenode主页。
- 在浏览器中输入<http://master:50030/>，可看到Map/Reduce主页。

## 关闭
### 关闭zookeeper

分别在3台机器上执行以下命令：

```shell
zkServer.sh stop
```

### 关闭Hadoop

在master主机执行以下命令：

```shell
stop-all.sh
```

