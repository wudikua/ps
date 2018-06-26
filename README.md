# 参数服务器

    本项目使用非常少的代码编写了深度学习训练的全过程，有完整的结构，通过面向对象的封装，在算法上有一定扩展性，不仅支持单机模式还支持分布式模式

    使用java实现的dnn训练框架，底层矩阵库使用Jblas(https://github.com/mikiobraun/jblas)，参数服务器使用Grpc+protobuf

    支持单机多CPU

    支持分布式，同步更新

    实现FNN模型和Wide And Deep模型


## 架构

![](./src/main/resources/structure.png "单机多机架构")

    单机多CPU是使用多个相同的模型，提交到线程池，各自做训练，然后再主线程等待所有线程执行完毕对梯度进行更新

    多机与单机的不同在于KVStore是分布式实现，获取参数都是从PS获取，多机还有一点是需要调用参数服务器的barrier测是否需要阻塞

## 代码结构


![](./src/main/resources/code.png "代码结构")


- __activations__

    激活函数，目前支持Sigmoid，Relu，LeakyRelu

    forward，输入x，输出y

    backward，dy是前向累计梯度，preY一般没什么用，Y比较有用，因为sigmoid的导数是通过y计算出来的y（1-y）

- __context__

    主要是配置参数，init方法通过java的-D参数初始化

    - -Dmode=dist 分布式模式 -Dmode=standalone 单机模式

    - -DnTermDump=20 训练20次打印train auc和预估值实际值

    - -Dthread=4 单机多线程训练使用的线程数量，默认为cpu核心数，单机多线程训练会同步对梯度做平均
    
    - -DisPs 分布式模式下当前节点是否是参数服务器
    
    - -DworkerNum 分布式模式下worker数量
    
    - -DpsPort 参数服务器端口号
    
    - -DpsHost 参数服务器端host地址

- __data__

    TestDataSet 从文件读取libsvm数据，该模块还没有封装，以后会做成队列，在训练期间异步填充

- __evaluate__

    AUC 计算auc工具

- __layer__

    每个layer有两个重要方法，分别是forward正想推导和backward反向求解梯度
    
    还有两个重要属性是A，为当前这一层的输出，delta，为当前这一层的累计梯度
    
    以及两个layer指针next和pre，正向的时候pre.A就是当前层的输入，反向的时候next.delta就可以得到前向梯度

- __AddLayer__

    对两个layer的结果做加法，再使用激活函数，反向时候梯度不变

- __ConcatLayer__

    对两个layer的结果做合并，作为下一个layer的输入

- __EmbeddingLayer__

    由若干个EmbeddingField组成

- __EmbeddingField__

    通过map<string, FloatMatrix> 这样的结构做embedding lookup

- __FcLayer__

    全链接网络，正向为A=activation(WX+B)

- __loss__

    计算损失，forward接口计算损失值，backward计算损失函数导数
    
    具体实现包括平方差损失和交叉熵损失

- __model__

    通过组织layer组成模型
    
    DNN模型，离散特征通过embedding加上连续特征，合并以后放入多层全链接网络
    
    WideDeepNN模型，在DNN的基础上，在全链接网络的最后一层增加了Wide层

- __net__

    PSClient，访问参数服务器的客户端，主要API有三个
    
    getList取参数，updateList更新参数，push推送梯度，barrier请求参数服务器是否继续下一轮训练
    
    PServer，参数服务器，实现PSClient的主要方法，将参数保存在KVStore内存中
    
    通过globalStep和workerStep的差值控制是否barrier

- __store__

    KVStore是保存参数，统一的接口为get取参数，sum累计梯度，update更新参数
    
    KVStore有两种工作模式，本地模式使用内存中的hashmap，分布式模式通过PSClient在远程获取，缓存在本地hashmap中

- __train__

    Trainer 组织训练过程，分布式和单机都用这一个实现，通过环境变量做些不同的事

- __update__

    更新参数，实现了固定学习率更新和Adam更新，其中Adam通过本地map存储历史值

- __util__

    MatrixUtil，现有矩阵库操作的扩容，以及proto中定义矩阵的互相转换

- __resource/proto/ps.proto__

    因为网络使用的是GRPC + protobuf的方式，这个文件为参数服务器的接口定义



