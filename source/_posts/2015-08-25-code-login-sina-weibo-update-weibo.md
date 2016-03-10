---
layout: post
category: [python,程序设计,开源]
title:  Python代码登录新浪微博并自动发微博
keywords: Python纯代码登录新浪微博并自动发微博，不使用SDK
tags: [Python,微博]
---

##前言
对于很少玩微博[@張行之_][1]的我来说，微博内容少的可怜。所以本人就想：能不能写个成功程序来帮我发微博。这个程序要满足以下要求：

1.  自动化，自动登录微博，自动发微博。
2.  微博内容要有意义，不能是随机生成的字符。
3.  可以设置每隔一段时间发一条微博，频率不能太快，当然也不能太慢。

于是，就诞生了这个程序，代码使用纯Python实现，没有使用微博SDK，主要是模拟HTTP操作。程序运行后会先登录微博，然后启动一个定时器，每隔一段时间爬取[秒拍][9]、[cnBeta][2]、[博客园][3]、[TechWeb][4]、[推酷][5]最新的内容，再转发到微博。

代码托管在GitHub上，项目地址：<https://github.com/zchlong/sinaWeibo.git>

试用了几天，效果可以查看我的微博：[@張行之_][1]。

<!--more-->


##整体结构
程序分为3个模块：***微博登录***、***定时发微博***、***微博内容生产***。代码结构如下：

```
sinaWeibo
|----main.py
|----sinaWeiboLogin.py
|----config.py
|----logger.py
|----sendWeibo.py
|----TextFactory.py
      |spider
      |----__init__.py
      |----utility.py
      |----http.py
      |----spider.py
      |----cnbeta.py          //解析cbbeta
      |----cnblog.py          //解析博客园
      |----techweb.py       /解析techweb
      |----tuicool.py          //解析推酷
      |----.....                      //更多解析
```


##使用
如果你只想使用该代码来发微博，并不关心代码是怎么实现的，那么你只需要读这一节内容。

1.  下载，项目地址：<https://github.com/zchlong/sinaWeibo.git>。
2.  修改配置文件`config.py`，把微博账号和密码改为你自己的，还可以设置发送时间（默认是30分钟一次）。
4.  在`TextFactory.py`可以设置微博内容生成规则，也可以使用默认的规则。
3.  运行`main.py`，`python main.py`。

注意：

1.  该代码使用了代码依赖[requests][7]和[rsa][8]，没有安装的话需要先安装：

```
pip install rsa
pip install requests
```

2.  如果你的微博登录时要输入验证码，该代码是登录不成功的，可以在账号安全的登录保护中设置不输入验证码。


##登录
登录网上有很多现成的方法，在GitHub上找到一个登录新浪微博的[Python代码][6]，使用[requests][7]，比urllib2更方便。代码依赖[requests][7]和[rsa][8]。代码有点旧，需要做一点修改。

```
WBCLIENT = 'ssologin.js(v1.4.5)' => WBCLIENT = 'ssologin.js(v1.4.18)'
```

两个正则表达式也需要修改下：

```
login_url = re.search(r'replace\([\"\']([^\'\"]+)[\"\']', resp.text).group(1)
改为：
login_url = re.search('replace\\(\'([^\']+)\'\\)', resp.text).group(1) 

login_str = re.match(r'[^{]+({.+?}})', resp.text).group(1)
改为：
login_str = login_str = re.search('\((\{.*\})\)', resp.text).group(1)
```

登录时要注意，如果需要输入验证码，这段代码是会登录失败的，可以在账号安全的登录保护中设置不输入验证码。


##定时自动发微博
新浪微博发微博的接口是：

`http://www.weibo.com/aj/mblog/add?ajwvr=6&__rnd=时间戳`

时间戳使用` int(time.time() * 1000`即可设置。
Post提交数据：

```
"location" : "v6_content_home", 
"appkey" : "", 
"style_type" : "1", 
"pic_id" : "", 
"text" : 微博内容, 
"pdetail" : "", 
"rank" : "0", 
"rankid" : "", 
"module" : "stissue", 
"pub_type" : "dialog", 
"_t" : "0", 
```

提交数据时需要设置Headers:

`self.http.headers["Referer"] = "http://www.weibo.com/u/%s/home?wvr=5" % str(self.uid)`

uid在登录时会返回。

在Python中启动一个定时器（Timer），每当定时器触发的时候向这个接口Post数据就能实现自动发微博了。

```python
def newTimer(self):
      self.timer = Timer(TIME_SLOG, self.main, ()).start()

def stop(self):
      log("结束任务")
      self.timer.cancel()
      pass

def main(self):
      self.sendWeibo()

      if TIMER_REPEAT:
            self.newTimer()

def sendWeibo(self):
      text = TextFactory.getText()
      self.update(text)
      log(u"发送微博：" + text)
```

##微博内容生产
要产生有意义的微博内容，一般需要从网站上爬取。当然，也可以把内容写入文本再定时发送。内容都是从网上爬取的，因此需要实现一个爬虫，用Python的requests爬取网页非常方便，几行代码搞定。使用`SGMLParser`解析网页也是非常方便的。爬虫部分在爬取网页都是一样的，解析时不同，所以只需要分别对每一个网站实现一个[`SGMLParser`][10]子类就能实现多个网站的爬取了。

为了从不同网站爬取数据，代码实现一个轮询机制，用一个容器保存各个网站的爬虫对象，在每次获取微博内容时使用不同的爬虫对象。

```python
spiders = [
      Spider(miaopai.HOME_URL, miaopai.MiaopaParser()),
      Spider(cnbeta.HOME_URL, cnbeta.CnbetaParser()),
      Spider(cnblog.HOME_URL, cnblog.CnblogParser()),
      Spider(techweb.HOME_URL, techweb.TechwebParser()),
      Spider(tuicool.HOME_URL, tuicool.TuicoolParser()),
      Spider(miaopai.HOME_URL, miaopai.MiaopaParser()),
]

currentIndex = 0
count = len(spiders)

def getText():
      spider = nextSpider()
      text = spider.getAMessage()
      return text

def nextSpider():
      global currentIndex
      spider = spiders[currentIndex]
      currentIndex = (currentIndex + 1) % count
      return spider
```

###添加爬虫
代码设计具有较好地扩展性，在爬虫类`spder.py`中定义一个解析属性

```python
class Spider(object):
      def __init__(self, homeUrl, parser):
            super(Spider, self).__init__()
            self.homeUrl = homeUrl
            self.parser = parser

      def getAMessage(self):
            html = http.get(self.homeUrl)
            self.parser.feed(html)
            return self.parser.getMsg()
```

在创建`Spider`对象时，只需要注入不同的解析对象，就能解析不同的网站内容了，甚至还可实现从其他渠道获取内容。

在`TextFactory.py`中实现了轮询机制，当有新的解析类时，只需在`TextFactory.py`中的`spiders`添加一个就行。

##结语
该代码已经基本满足了前言的3点要求，不过还存在一些问题：

1.  爬虫部分还存在很多冗余，可以进一步优化。
2.  产生微博内容时可能会生成相同的内容，尤其是目标网站更新频率不高时。

代码托管在码托管在GitHub上，项目地址：<https://github.com/zchlong/sinaWeibo.git>。


[1]: http://weibo.com/5longme
[2]: http://www.cnbeta.com
[3]: http://www.cnblogs.com
[4]: http://www.techweb.com.cn/roll
[5]: http://www.tuicool.com
[6]: https://gist.github.com/mrluanma/3621775
[7]: https://pypi.python.org/pypi/requests
[8]: https://pypi.python.org/pypi/rsa
[9]: http://www.miaopai.com/miaopai/plaza?cateid=2002
[10]: https://docs.python.org/2/library/sgmllib.html