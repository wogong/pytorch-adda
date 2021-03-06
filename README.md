# PyTorch-ADDA

A PyTorch implementation for [Adversarial Discriminative Domain Adaptation](https://arxiv.org/abs/1702.05464).

## Environment

- Python 3.6
- PyTorch 0.3.1post2

## Network

In this experiment, I use three types of network. They are very simple.

- LeNet encoder

  ```
  LeNetEncoder (
    (encoder): Sequential (
      (0): Conv2d(1, 20, kernel_size=(5, 5), stride=(1, 1))
      (1): MaxPool2d (size=(2, 2), stride=(2, 2), dilation=(1, 1))
      (2): ReLU ()
      (3): Conv2d(20, 50, kernel_size=(5, 5), stride=(1, 1))
      (4): Dropout2d (p=0.5)
      (5): MaxPool2d (size=(2, 2), stride=(2, 2), dilation=(1, 1))
      (6): ReLU ()
    )
    (fc1): Linear (800 -> 500)
  )
  ```

- LeNet classifier

  ```
  LeNetClassifier (
    (fc2): Linear (500 -> 10)
  )
  ```

- Discriminator

  ```
  Discriminator (
    (layer): Sequential (
      (0): Linear (500 -> 500)
      (1): ReLU ()
      (2): Linear (500 -> 500)
      (3): ReLU ()
      (4): Linear (500 -> 2)
      (5): LogSoftmax ()
    )
  )
  ```

## Result

|                       | MNIST-USPS     | USPS-MNIST | SVHN-MNIST |SVHN-MNIST(BN) |AMAZON-WEBCAM |
| :-------------------: | :------------: | :--------: | :--------: |:------------: |:--------: |
| ADDA: Source Only     |   0.7520       |  0.5710    |  0.6010    | 0.6010        | 0.6260    |
| ADDA                  |   0.8940       |  0.9010    |  0.7600    | 0.7600        | 0.7510    |
| This Repo: Source Only|   0.8617       |  0.6305    |  0.6326    | 0.6414        | -    |
| This Repo             |   0.9100       |  0.8815    |not converge| 0.8188        | -    |

1. mnist-usps 实验使用的是部分的样本（2000/1800），注意修改代码。实验设置参考文献 \cite{long2013transfer}. sampling 2000 images from MNIST and 1800 from USPS
2. svhn-mnist 重现，不使用 BN 无法收敛，同样的结论来自<https://github.com/aabbas90/ADDA-PyTorch>

## Other implementations

- authors(tensorflow) <https://github.com/erictzeng/adda>
- pytorch <https://github.com/corenel/pytorch-adda>
- pytorch forks <https://github.com/Zzlongjuanfeng/pytorch-adda>
- pytorch <https://github.com/aabbas90/ADDA-PyTorch>