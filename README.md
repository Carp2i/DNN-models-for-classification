# DNN models for classification
这个仓库是仓库作者用于托管练手模型的 

仓库作者还处于写pytorch的 **轮椅阶段**


## 目前已经更新
1. [lenet](https://github.com/Carp2i/DNN-models-for-classification/tree/master/lenet)
2. [alexnet](https://github.com/Carp2i/DNN-models-for-classification/tree/master/alexnet)
3. [vgg](https://github.com/Carp2i/DNN-models-for-classification/tree/master/vgg)
4. [googLeNet](https://github.com/Carp2i/DNN-models-for-classification/tree/master/googlenet)
5. ResNet(待完成)

### LeNet
这个模型经常被作为 [PyTorch](https://pytorch.org/get-started/locally/) 框架的 tutorial 模型，基本数据集会使用 FashionMNIST 或 cifar-10。

![](https://pic.imgdb.cn/item/625a70ee239250f7c59dea18.jpg)


### AlexNet
2012年的ImageNet比赛冠军，首次使用GPU加速深度学习，带来了革命性的分类效果的提升。相比较和传统的，feature engineering 的做法，**提升了10个百分点** 

![](https://pic.imgdb.cn/item/625a714c239250f7c59eaeff.jpg)

| layer_name | kernel_size | kernel_num | padding | stride |
|------------|-------------|------------|---------|--------|
| Conv1 | 11 | 96 | [1, 2] | 4 |
| Maxpool1 | 3 | None | 0 | 2 |
| Conv2 | 5 | 256 | [2, 2] | 1 |
| Maxpool2 | 3 | None | 0 | 2 |
| Conv3 | 3 | 384 | 0 | 2 |
| Conv4 | 3 | 384 | 0 | 2 |
| Conv5 | 3 | 256 | 0 | 2 |
| Maxpool3 | 3 | None | 0 | 2 |
| FC1 | 2048 | None | None | None |
| FC2 | 2048 | None | None | None |
| FC3 | classes_num | None | None |


### Vgg
非常Fat的网络（**指参数量特别大**），自己train出来的VGG16，只有75%左右的Validation acc，还是在ImageNet的数据集上，仓库作者的vgg16跑在 <u>花分类数据集</u> 上

原论文： [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)

![](https://pic.imgdb.cn/item/625a74b7239250f7c5a55bf9.jpg)


### GoogLenet
致敬Lenet玩了一个谐音梗
...未完待续(2022/4/16)
做完了 (2022/4/17)

原论文：[Going Deeper with Convolutions](https://arxiv.org/abs/1409.4842)

使用的显卡为 N卡2060-6GB的显卡，感觉过来和VGG的区别特别明显：
VGG：batch_size=27 **显存吃满**
GoogLeNet：batch_size=96 **只用5.5GB**显存

根据计算，大概两个模型参数差了20倍


#### Inception a/b
![](https://pic.imgdb.cn/item/625bbe8c239250f7c5a34192.jpg)

#### model structure

![](https://pic.imgdb.cn/item/625bbea1239250f7c5a36586.jpg)

#### layers table

![](https://pic.imgdb.cn/item/625bbebb239250f7c5a39959.jpg)

### ResNet

写是已经写好了

## dataset
仓库的 root path 下有一个data文件夹，被我ignore了
root下的split_data文件是用来处理 <u>**花分类数据**</u> 集的
用于 shuffle 还有 split train/val dataset

## reference

https://github.com/WZMIAOMIAO/deep-learning-for-image-processing