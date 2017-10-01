---
title: Backpropagation算法从公式到代码
date: 2017-10-01 13:45:30
categories: DeepLearning
keywords: Backpropagation,反向传播算法,深度学习
tags: [Deep Learning]
---

## 一、前言

深度学习在近几年无疑是非常热门的一个研究领域。不管是在学术界还是工业界，深度学习都有重要的研究和应用价值。本文主要来探讨一下深度学习中最基础的一个算法——反向传播算法（Backpropagation）。

本文将从定义一个神经元开始，然后定义一个非常简单的两层神经网络，基于这个神经网络来说明反向传播算法的执行流程。

<!--more-->

## 二、神经元

神经网络的前生就是神经网络，维基百科对神经网络的定义如下：

>在机器学习和认知科学领域，人工神经网络（英文：artificial neural network，缩写ANN），简称神经网络（英文：neural network，缩写NN）或类神经网络，是一种模仿生物神经网络(动物的中枢神经系统，特别是大脑)的结构和功能的数学模型或计算模型，用于对函数进行估计或近似。神经网络由大量的人工神经元联结进行计算。大多数情况下人工神经网络能在外界信息的基础上改变内部结构，是一种自适应系统。

神经网络中最基本的成分是神经元（neuron），即上述的人工神经元。神经元是一个多输入单输出的模型，如下图所示：

![神经元，来源于维基百科](/assets/images/2017/bp-01.png)

用数学公式表示如下：

![BP公式](/assets/images/2017/bp-02.png)

其中，xi是输入值，y为输出，wi 是连接权重，b为阈值。f(x)是激活函数（activation function），本文使用Sigmoid函数，Sigmoid函数定义如下：

![sigmoid](/assets/images/2017/bp-03.png)

其图形如下所示：

![sigmoid](/assets/images/2017/bp-04.png)

## 三、神经网络

本文定义一个非常简单的两层神经网络结构（一个输入层，一个隐含层和一个输出层），每一层包含两个神经元，如下图所示：

![nn](/assets/images/2017/bp-05.png)

为了方便描述，对该神经网络做如下定义：

  * w<sup>l</sup><sub>ij</sub>：第l层网络中第i个神经元与地l-1层网络中第j个神经元连接的权重。
  * b<sup>l</sup><sub>i</sub>：第l层网络第i个神经元的阈值。
  * z<sup>l</sup><sub>i</sub>：定义如下：

    ![BP](/assets/images/2017/bp-06.png)

  * a<sup>l</sup><sub>i</sub>：第l层网络第i个神经元的输出。对于第一层，a<sup>1</sup><sub>i</sub>=x<sub>i</sub>；对于隐含层和输出层，有：

  ![BP](/assets/images/2017/bp-07.png)

其中f(x)为激活函数，使用sigmoid函数，则：

  ![BP](/assets/images/2017/bp-08.png)

## 四、BP算法（Backpropagation）

### 4.1 正向传播

正向传播比较简单，对于输入层：

![BP](/assets/images/2017/bp-09.png)

隐含层：

![BP](/assets/images/2017/bp-10.png)

输出层和隐含层一样：

![BP](/assets/images/2017/bp-11.png)

输出为：

![BP](/assets/images/2017/bp-12.png)


### 4.2 Loss function
Loss function也称Cost function，就是一种度量误差的方式，Loss function有很多种，本文以平方误差函数为例，公式如下：

![BP](/assets/images/2017/bp-13.png)

即：

![BP](/assets/images/2017/bp-14.png)

其中y为实际值，y'为输出值。

### 4.3 求误差偏导

要知道某一个w和b对最终误差的影响程度，可以对其求偏导。

隐含层到输出层的偏导为：

![BP](/assets/images/2017/bp-15.png)

使用链式法则（chain rule）可以得到：

![BP](/assets/images/2017/bp-16.png)

其中：


![BP](/assets/images/2017/bp-17.png)
![BP](/assets/images/2017/bp-18.png)
![BP](/assets/images/2017/bp-19.png)

设：

![BP](/assets/images/2017/bp-20.png)

即δ<sup>3</sup><sub>i</sub>代表第三层中第i个神经元的误差。 则：

![BP](/assets/images/2017/bp-21.png)

同理可得：

![BP](/assets/images/2017/bp-22.png)

至此，输出层的权重和阈值的偏导数就已求出。接下来求隐含层的偏导数，同样使用链式法则，可得：

![BP](/assets/images/2017/bp-23.png)

设：

![BP](/assets/images/2017/bp-24.png)

即δ<sup>2</sup><sub>i</sub>代表第三层中第i个神经元的误差。 则：

![BP](/assets/images/2017/bp-25.png)

同理可得：

![BP](/assets/images/2017/bp-26.png)

至此，两层神经网络的求导就已经完成。通过观察整个推导过程可知，反向传播其实就是链式法则的重复使用。

### 4.4 更新权重
BP算法采用梯度下降（gradient descent）策略更新权重和阈值，对于误差E和给定学习率n，有：

![BP](/assets/images/2017/bp-27.png)

至此BP算法就已完成。

## 五、代码实现

1. sigmoid和平方误差函数以及对应的导数：

  ```shell
  def sigmoid(x):
      return 1 / (1 + np.exp(-x))

  def d_sigmoid(y):
      return y * (1 - y)  

  def loss_mse(target, output):
      return 1 / 2 * (target - output) ** 2

  def d_loss_mse(target, output):
      return target - output
  ```

2. 定义一个神经元，代码如下：

  ```python
  class Neuron:  

      def __init__(self, num_inputs):
          self.weights = []
          self.num_inputs = num_inputs
          self.bais = 0
          self.activation = sigmoid
          self.d_activation = d_sigmoid
          self.loss_fun = loss_mse
          self.d_loss_fun = d_loss_mse
          self.inputs = None
          self.output = None
          self.derivation = []
          self.g = None
          self.Z = 0    

      def init_weights(self, weights=None, bais=None):
          if weights is None or len(weights) != self.num_inputs:
              self.weights = random.rand(self.num_inputs)
          else:
              self.weights = np.array(weights)
          if bais is None:
              self.bais = random.rand(1)[0]
          else:
              self.bais = bais    

      def forward(self, inputs):
          self.inputs = inputs
          I = 0.0
          for i in range(len(inputs)):
              I += inputs[i] * self.weights[i]
          self.Z = I + self.bais
          O = self.activation(self.Z)
          self.output = O
          return self.output    

      def compute_loss(self, target):
          loss = self.loss_fun(target=target, output=self.output)
          return loss    

      def backward(self, target):
          self.compute_derivation(target)
          self.update_weights()    

      def compute_derivation(self, target):
          '''
          compute derivation for output layer
          '''
          self.g = self.d_loss_fun(target=target, output=self.output) * self.d_activation(y=self.output)
          self.derivation = []
          for i in self.inputs:
              self.derivation.append(self.g * i)    

      def compute_derivation_hide(self, next_layer, current_layer_index):
          '''
          compute derivation for hide layers
          '''
          g = 0.0
          for next_layer_unit in next_layer.units:
              g += next_layer_unit.g * next_layer_unit.weights[current_layer_index]
          self.g = g * self.d_activation(y=self.output)
          self.derivation = []
          for i in self.inputs:
              self.derivation.append(self.g * i)    

      def update_weights(self, rate=0.5):
          for i in range(len(self.derivation)):
              self.weights[i] += rate * self.derivation[i]    

      def update_bais(self, rate):
          self.bais += rate * self.g
  ```

3. 定义网络层，代码如下：  

  ```python
  class Layer:    

      def __init__(self, num_units, num_inputs):
          self.num_units = num_units
          self.units = [Neuron(num_units) for _ in range(num_inputs)]
          self.inputs = None
          self.outputs = None
          self.num_inputs = num_inputs
          self.bais = 0.0    

      def init_weights(self, weights=None, bais=None):
          self.bais = bais
          for i, unit in enumerate(self.units):
              w = weights[i] if weights is not None and len(weights) > i else None
              b = bais
              unit.init_weights(weights=w, bais=b)    

      def forward(self, inputs):
          self.inputs = inputs
          self.outputs = []
          for unit in self.units:
              output = unit.forward(inputs)
              self.outputs.append(output)
          return self.outputs    

      def compute_loss(self, targets):
          loss = 0.0
          for i, unit in enumerate(self.units):
              loss += unit.compute_loss(targets[i])
          return loss    

      def backward(self, targets):
          self.compute_derivation(targets)
          self.update_weights()    

      def compute_derivation(self, targets):
          for i, unit in enumerate(self.units):
              unit.compute_derivation(targets[i])    

      def compute_derivation_hide(self, next_layer):
          for i, unit in enumerate(self.units):
              unit.compute_derivation_hide(next_layer=next_layer, current_layer_index=i)    

      def update_weights(self, rate=0.5):
          for unit in self.units:
              unit.update_weights(rate)    

      def update_bais(self, rate):
          for unit in self.units:
              unit.update_bais(rate)
  ```

4. 定义一个两层网络，代码如下：

  ```python
  class NeuralNetworksLayer2:  

      def __init__(self, num_inputs=2, num_units_hide=2, num_outputs=2, rate=0.5):
          self.num_inputs = num_inputs
          self.num_units_hide = num_units_hide
          self.hide_layer = Layer(num_units=num_units_hide, num_inputs=num_inputs)
          self.num_outputs = num_outputs
          self.output_layer = Layer(num_units=num_outputs, num_inputs=num_units_hide)
          self.inputs = None
          self.rate = rate  

      def init_weights(self, weights=None, bais=None):
          '''
          init weights of networks  

          Args:
              weights: [[[float]]]
              bais:  [float]
          '''
          for i, layer in enumerate([self.hide_layer, self.output_layer]):
              w = weights[i] if weights is not None and len(weights) > i else None
              b = bais[i] if bais is not None and len(bais) > i else None
              layer.init_weights(w, b)  

      def forward(self, inputs, targets=None):
          self.inputs = None
          hide_layer_output = self.hide_layer.forward(inputs)
          outputs = self.output_layer.forward(hide_layer_output)
          if targets is not None:
              loss = self.output_layer.compute_loss(targets=targets)
              print('loss: %f' % loss)
          return outputs  

      def backward(self, targets):  

          # compute derivation
          self.output_layer.compute_derivation(targets=targets)
          self.hide_layer.compute_derivation_hide(next_layer=self.output_layer)  

          # update weights for each layers
          self.output_layer.update_weights(rate=self.rate)
          self.hide_layer.update_weights(rate=self.rate)  

          # update bais
          self.output_layer.update_bais(rate=self.rate)
          self.hide_layer.update_bais(rate=self.rate)
  ```

4. 编写测试代码，如下：

  ```python
  if __name__ == '__main__':
      inputs = [0.05, 0.10]
      target = [0.01, 0.99]
      rate = 0.5
      epochs = 100
      print('input:', inputs)
      print('target:', target)
      print('epochs:', epochs)
      networks = NeuralNetworksLayer2(num_inputs=2, num_units_hide=2, num_outputs=2, rate=rate)
      networks.init_weights()  

      for _ in range(epochs):
          networks.forward(inputs)
          networks.backward(target)  

      outputs = networks.forward(inputs, targets=target)
      print('target: %s output: %s' % (target, outputs))
  ```

执行后输出：

  ```shell
  input: [0.05, 0.1]
  target: [0.01, 0.99]
  epochs: 100
  loss: 0.005217
  target: [0.01, 0.99] output: [0.091647705766820978, 0.92861644531550158]
  ```

可见，训练100次后效果还不是很好。再分别训练1,000次和10,000次，输出如下：

  ```shell
  # 1,000
  input: [0.05, 0.1]
  target: [0.01, 0.99]
  epochs: 1000
  loss: 0.000241
  target: [0.01, 0.99] output: [0.025868082566954786, 0.97483122679788203]    

  # 10,000
  input: [0.05, 0.1]
  target: [0.01, 0.99]
  epochs: 10000
  loss: 0.000003
  target: [0.01, 0.99] output: [0.011611209033754864, 0.98837942389784894]
  ```

从输出中可以看出，训练10,000次后，输出已经非常接近目标值了，这说明了神经网络学习的有效性。

完整代码可以到[GitHubGist](https://gist.github.com/chaolongzhang/5142045d9c2cec38c0cfbf54850a52a0)下载。

## 六、结语

本文探讨了深度学习/神经网络中最基础的一个算法——BP算法。本文首先对BP算法的前向与后向传播的公式进行了推导，然后使用Python实现了一个简易的神经网络，最后通过测试证明了该网络的有效性。本文的代码是为了学习和易于阅读而写的，所以性能并不高。如果需要有优化性能，可以采用Numpy的矩阵乘法来实现。当然，更高效地方法是使用深度学习库，如[PyTorch](http://pytorch.org/)、[Tensorflow](https://www.tensorflow.org/)和[Caffe2](https://caffe2.ai/)等。

## Ref

* [Neural Networks and Deep Learning-How the backpropagation algorithm works](http://neuralnetworksanddeeplearning.com/chap2.html)
* [维基百科-人工神经网络](https://zh.wikipedia.org/wiki/%E4%BA%BA%E5%B7%A5%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C)
