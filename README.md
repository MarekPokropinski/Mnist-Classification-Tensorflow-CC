# Mnist-Classification-Tensorflow-CC
Classification on MNIST dataset with use of Tensorflow C++ API

## Prerequisites
* [tensorflow_cc](https://github.com/FloopCZ/tensorflow_cc)

## Training
To train put MNIST dataset files in main folder:
* train-images-idx3-ubyte
* train-labels-idx1-ubyte
* t10k-images-idx3-ubyte
* t10k-labels-idx1-ubyte

Build project with cmake:
```
$ mkdir build && cd build
$ cmake .. 
$ cmake --build .
```
