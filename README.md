# Karakara
A neural network library

## 說明
這是NTNU OOAD Spring 2020 Project的一部分  
是一個手動微分的神經網路函式庫  

## 功能
打勾代表已完成
* 主要網路層
  * -[x] Dense
  * -[x] Dropout
  * -[x] BatchNormalization
  * -[x] Conv2D
  * -[x] MaxPoolind2D
* 激勵函數
  * -[x] Sigmoid
  * -[x] Tanh
  * -[x] ReLU
  * -[x] LeakyReLU
* 優化器
  * -[x] SGD
  * -[x] Momentum
  * -[x] RMSProp
  * -[x] Adam
* 特殊功能
  * -[x] GPU支援
  * -[x] 半精度浮點數支援
  * -[ ] 多GPU支援
  * -[ ] 儲存模型

## 使用範例
mnist_mlp.py : Mnist手寫數字分類  
mnist_deep_gan.py : 基於Deep GAN的Mnist手寫數字生成  
cifar10_cnn.py : 使用CNN的Cifar10分類  
cifar10_resnet_v2.py：使用ResNet v2的Cifar10分類  
