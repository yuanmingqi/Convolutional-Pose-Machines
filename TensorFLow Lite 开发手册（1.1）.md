@[TOC](TensorFLow Lite 开发手册（1.1）)

# 0 独家鸣谢
独家感谢**庆喜哥**、**苏睿哥**在C++部分提供的不可或缺的帮助！！！

# 1 开发环境（已测试）
|TensorFlow 版本  | 2.0（stable） |
|:--:|:--:|
|**开发平台**|Ubuntu 18.04 LTS |
|  **Python 版本** |3.6  |
|**gcc**|gcc (Ubuntu 7.4.0-1ubuntu1~18.04.1) 7.4.0|
|**g++**|g++ (Ubuntu 7.4.0-1ubuntu1~18.04.1) 7.4.0|
|**cmake**|cmake version 3.10.2|
|**opencv**|3.2.0|

# 2 TensorFlow 2.0安装与使用
## 2.1 TensorFlow 2.0安装
创建虚拟环境：
```
conda create --name py36-tf20 python=3.6
conda activate py36-tf20
```
目前conda源已维护至2.0版本，可以使用conda命令安装：
```
# 安装CPU版本
conda install tensorflow==2.0.0

# 安装GPU版本，CUDA支持请查看TensoFlow官网
conda install tensorflow-gpu==2.0.0
```
## 2.2 TensorFlow 2.0 新特性简介

 - **API Cleanup**
移除了许多库，如**tf.app，tf.logging，tf.flags**等，将原有的函数库整合进了**tf.keras**，如**tf.layers->tf.keras.layers**
 
 - **Eager execution**
 在2.0中，动态图机制成为默认机制，不再需要用户手动创建会话，也不需要使用 **sess.run()** 来指定输入输出的张量。

 - **No more globals**
不再依赖隐式全局命名空间，即不再依赖tf.Variable()来声明变量，而是采用默认机制：“ Keep track of your variables!”，如果不再追溯某个tf.Variable，其就会被回收。

 - **Functions, not sessions**（个人认为是很重要、也很厉害的一点）
  在2.0中提供了名为 **@tf.function()** 的装饰器，它可以对普通的Python函数进行标记以进行JIT编译，然后TensorFlow就可以将其作为单一的计算图来运行，这使得该函数**可以直接被优化**和**作为模型导出**。并且为了帮助用户在添加**@tf.function**时避免重写代码，**AutoGraph**将python中的一些函数转换为其**TensorFlow**包含的等价函数：
  
```python
  for/while -> tf.while_loop (break and continue are supported)
  if -> tf.cond
  for _ in dataset -> dataset.reduce
```
 - 个人补充
在2.0中，**Keras**被全面整合，**Google**也推荐大家使用**tf.keras**更高效构建模型，并且使用**tf.data**构建数据流（有关**tf.data**使用的流程可以参照我的博客[https://blog.csdn.net/weixin_42499236/article/category/8331677](https://blog.csdn.net/weixin_42499236/article/category/8331677)），而**tf.keras**保存的模型也可以直接被转换为**TensorFlow Lite**模型，所以还是用**Keras**比较好。
## 2.3 训练模型的一般流程
训练模型的基本流程如下：

 - 将 Tensorflow 导入程序：
```python
import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
```
 - 加载数据
```python
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Add a channels dimension
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]
```
 - 使用 tf.data 来将数据集切分为 batch 以及shuffle数据集：
```python
train_ds = tf.data.Dataset.from_tensor_slices(
    (x_train, y_train)).shuffle(10000).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)
```
 - 使用 Keras 模型子类化（model subclassing） API 构建 tf.keras 模型：
```python
class MyModel(Model):
  def __init__(self):
    super(MyModel, self).__init__()
    self.conv1 = Conv2D(32, 3, activation='relu')
    self.flatten = Flatten()
    self.d1 = Dense(128, activation='relu')
    self.d2 = Dense(10, activation='softmax')

  def call(self, x):
    x = self.conv1(x)
    x = self.flatten(x)
    x = self.d1(x)
    return self.d2(x)

model = MyModel()
```
 - 为训练选择优化器与损失函数：
```python
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

optimizer = tf.keras.optimizers.Adam()
```
 - 选择衡量指标来度量模型的损失值（loss）和准确率（accuracy）。这些指标在 epoch 上累积值，然后打印出整体结果。
```python
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
```
 - 使用 tf.GradientTape 来训练模型：
```python
@tf.function
def train_step(images, labels):
  with tf.GradientTape() as tape:
    predictions = model(images)
    loss = loss_object(labels, predictions)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  train_loss(loss)
  train_accuracy(labels, predictions)
```
 - 测试模型：
```python
@tf.function
def test_step(images, labels):
  predictions = model(images)
  t_loss = loss_object(labels, predictions)

  test_loss(t_loss)
  test_accuracy(labels, predictions)
EPOCHS = 5

for epoch in range(EPOCHS):
  for images, labels in train_ds:
    train_step(images, labels)

  for test_images, test_labels in test_ds:
    test_step(test_images, test_labels)

  template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
  print (template.format(epoch+1,
                         train_loss.result(),
                         train_accuracy.result()*100,
                         test_loss.result(),
                         test_accuracy.result()*100))  
```

 - 输出如下：
```python
Epoch 1, Loss: 0.13822732865810394, Accuracy: 95.84833526611328, Test Loss: 0.07067110389471054, Test Accuracy: 97.75
Epoch 2, Loss: 0.09080979228019714, Accuracy: 97.25, Test Loss: 0.06446609646081924, Test Accuracy: 97.95999908447266
Epoch 3, Loss: 0.06777264922857285, Accuracy: 97.93944549560547, Test Loss: 0.06325332075357437, Test Accuracy: 98.04000091552734
Epoch 4, Loss: 0.054447807371616364, Accuracy: 98.33999633789062, Test Loss: 0.06611879169940948, Test Accuracy: 98.00749969482422
Epoch 5, Loss: 0.04556874558329582, Accuracy: 98.60433197021484, Test Loss: 0.06510476022958755, Test Accuracy: 98.10400390625
```

## 2.4 训练示例模型
基于以下代码训练示例模型，并得到模型文件**model.h5**：
```python
import tensorflow as tf
from tensorflow import keras

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = tf.expand_dims(x_train, 3)
y_train = keras.utils.to_categorical(y_train, num_classes=10)
datasets = tf.data.Dataset.from_tensor_slices((x_train, y_train))
datasets = datasets.repeat(1).batch(10)

img = keras.Input(shape=[28, 28, 1])

x = keras.layers.Conv2D(filters=64, kernel_size=4,strides=1, padding='SAME',activation='relu')(img)
x = keras.layers.AveragePooling2D(pool_size=2, strides=2, padding='SAME')(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Dropout(0.15)(x)

x = keras.layers.Conv2D(filters=64, kernel_size=4,strides=1, padding='SAME',activation='relu')(img)
x = keras.layers.MaxPooling2D(pool_size=2, strides=2, padding='SAME')(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Dropout(0.15)(x)

x = keras.layers.Flatten()(x)
x = keras.layers.Dense(512, activation='relu')(x)
y_pred = keras.layers.Dense(10, activation='softmax')(x)

model = keras.Model(inputs=img, outputs=y_pred)
model.compile(optimizer= keras.optimizers.Adam(0.01),
             loss= keras.losses.categorical_crossentropy,
             metrics = ['AUC', 'accuracy'])

model.fit(datasets, epochs=1)
model.save('/home/ai/model.h5')
```
## 2.5 模型转换
### 2.5.1 模型转换方法
基本工作流程如下：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191021190339639.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MjQ5OTIzNg==,size_16,color_FFFFFF,t_70)
**TensorFlow Lite** 提供以下三种模型转换方法：

 - tf.lite.TFLiteConverter.from_keras_model()，转换实例化的**Keras**模型
 - tf.lite.TFLiteConverter.from_saved_model()，转换**pb**文件
 - tf.lite.TFLiteConverter.from_concrete_functions()，转换**具体的函数**
### 2.5.2 转换示例模型
```python
import numpy as np
# 转换模型。
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
open('/home/ai/converted_model.tflite', 'wb').write(tflite_model)
```
最终得到转化后的模型——**converted_model.tflite**

## 2.6 模型调用
### 2.6.1 Python 接口
```python
# 加载 TFLite 模型并分配张量（tensor）。
interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()

# 获取输入和输出张量。
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 使用随机数据作为输入测试 TensorFlow Lite 模型。
input_shape = input_details[0]['shape']
input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()

# 函数 `get_tensor()` 会返回一份张量的拷贝。
# 使用 `tensor()` 获取指向张量的指针。
tflite_results = interpreter.get_tensor(output_details[0]['index'])

# 使用随机数据作为输入测试 TensorFlow 模型。
tf_results = model(tf.constant(input_data))

print("tflite result:", tflite_results)
print("tf result:", tf_results)
```
输出结果如下：
```python
tflite result: [[2.80235346e-09 9.99904633e-01 1.81704600e-05 2.76264545e-09
  8.59975898e-06 3.23225287e-08 6.01521315e-05 1.01964176e-07
  8.37052448e-06 3.09832138e-09]]
tf result: tf.Tensor(
[[2.8023428e-09 9.9990463e-01 1.8170409e-05 2.7626350e-09 8.5997262e-06
  3.2322404e-08 6.0152019e-05 1.0196379e-07 8.3704927e-06 3.0983038e-09]], shape=(1, 10), dtype=float32)
```

### 2.6.2 C++接口
```cpp
// Load the model
std::unique_ptr<tflite::FlatBufferModel> model =
    tflite::FlatBufferModel::BuildFromFile(filename);

// Build the interpreter
tflite::ops::builtin::BuiltinOpResolver resolver;
std::unique_ptr<tflite::Interpreter> interpreter;
tflite::InterpreterBuilder(*model, resolver)(&interpreter);

// Resize input tensors, if desired.
interpreter->AllocateTensors();

float* input = interpreter->typed_input_tensor<float>(0);
// Fill `input`.

interpreter->Invoke();

float* output = interpreter->typed_output_tensor<float>(0);
```

# 3 安装TensorFlow Lite
## 3.1 平台支持
|操作系统| iOS、Android、Linux |
|--|--|
|**开发平台**  | ARM64（支持RK3399）、Raspberry Pi、iOS |

## 3.2 下载源代码
到GitHub仓库下载2.0全部代码：
```
wget https://github.com/tensorflow/tensorflow/archive/master.zip
```
解压后进入：
```
unzip master.zip
cd ./tensorflow/lite/tools/make/
```
## 3.3 安装依赖库
### 3.3.1 安装工具链
```
sudo apt-get install build-essential
```
### 3.3.2 安装依赖库
```
./tensorflow/lite/tools/make/download_dependencies.sh
```
## 3.4 编译TensorFlow Lite
在该目录下有多个**.lib**文件，根据运行环境进行编译：
```
# 在ARM64上运行：
bash ./tensorflow/lite/tools/make/build_aarch64_lib.sh
# 在普通Ubuntu上运行：
bash ./tensorflow/lite/tools/make/build_lib.sh
```
这会编译出一个静态库在： 
```
./tensorflow/lite/tools/make/gen/aarch64_armv8-a/lib/libtensorflow-lite.a
```
编译时可能会提示如下错误：
```
/usr/bin/ld: 找不到 -lz collect2: error: ld returned 1 exit status
```
运行下列命令安装zlib：
```
sudo apt-get install zlib1g-dev
```
# 4 TensorFlow Lite模型使用实例（分类模型）
## 4.1 新建CLion工程
到（[https://download.csdn.net/download/weixin_42499236/11892106](https://download.csdn.net/download/weixin_42499236/11892106)）下载该工程，解压后如下图所示：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191021190041635.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MjQ5OTIzNg==,size_16,color_FFFFFF,t_70)

## 4.2 编写Cmakelist
```c
cmake_minimum_required(VERSION 3.15)
project(testlite)

set(CMAKE_CXX_STANDARD 14)

include_directories(/home/ai/CLionProjects/tensorflow-master/)
include_directories(/home/ai/CLionProjects/tensorflow-master/tensorflow/lite/tools/make/downloads/flatbuffers/include)
include_directories(/home/ai/CLionProjects/tensorflow-master/tensorflow/lite/tools/make/downloads/absl)

add_executable(testlite main.cpp bitmap_helpers.cc utils.cc)

target_link_libraries(testlite /home/ai/CLionProjects/tensorflow-master/tensorflow/lite/tools/make/gen/linux_x86_64/lib/libtensorflow-lite.a -lpthread -ldl -lrt)
```
## 4.3 编写main.cpp

 - 导入头文件
```cpp
#include <fcntl.h>      // NOLINT(build/include_order)
#include <getopt.h>     // NOLINT(build/include_order)
#include <sys/time.h>   // NOLINT(build/include_order)
#include <sys/types.h>  // NOLINT(build/include_order)
#include <sys/uio.h>    // NOLINT(build/include_order)
#include <unistd.h>     // NOLINT(build/include_order)

#include <iostream>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

#include "bitmap_helpers.h"
#include "get_top_n.h"

#include "tensorflow/lite/model.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/optional_debug_tools.h"
#include "tensorflow/lite/string_util.h"
#include "tensorflow/lite/profiling/profiler.h"
#include "tensorflow/lite/delegates/nnapi/nnapi_delegate.h"
#include "absl/memory/memory.h"
#include "utils.h"

using namespace std;
```

 - 调用GPU、NNAPI加速（若无GPU，则默认使用CPU）

```cpp
#define LOG(x) std::cerr

double get_us(struct timeval t) { return (t.tv_sec * 1000000 + t.tv_usec); }

using TfLiteDelegatePtr = tflite::Interpreter::TfLiteDelegatePtr;
using TfLiteDelegatePtrMap = std::map<std::string, TfLiteDelegatePtr>;

// 调用GPU
TfLiteDelegatePtr CreateGPUDelegate(tflite::label_image::Settings* s) {
#if defined(__ANDROID__)
    TfLiteGpuDelegateOptionsV2 gpu_opts = TfLiteGpuDelegateOptionsV2Default();
  gpu_opts.inference_preference =
      TFLITE_GPU_INFERENCE_PREFERENCE_SUSTAINED_SPEED;
  gpu_opts.is_precision_loss_allowed = s->allow_fp16 ? 1 : 0;
  return evaluation::CreateGPUDelegate(s->model, &gpu_opts);
#else
    return tflite::evaluation::CreateGPUDelegate(s->model);
#endif
}

TfLiteDelegatePtrMap GetDelegates(tflite::label_image::Settings* s) {
    TfLiteDelegatePtrMap delegates;
    if (s->gl_backend) {
        auto delegate = CreateGPUDelegate(s);
        if (!delegate) {
            LOG(INFO) << "GPU acceleration is unsupported on this platform.";
        } else {
            delegates.emplace("GPU", std::move(delegate));
        }
    }

    if (s->accel) {
        auto delegate = tflite::evaluation::CreateNNAPIDelegate();
        if (!delegate) {
            LOG(INFO) << "NNAPI acceleration is unsupported on this platform.";
        } else {
            delegates.emplace("NNAPI", tflite::evaluation::CreateNNAPIDelegate());
        }
    }
    return delegates;
}
```

 - 读取标签文件

```cpp
TfLiteStatus ReadLabelsFile(const string& file_name,
                            std::vector<string>* result,
                            size_t* found_label_count) {
    std::ifstream file(file_name);
    if (!file) {
        LOG(FATAL) << "Labels file " << file_name << " not found\n";
        return kTfLiteError;
    }
    result->clear();
    string line;
    while (std::getline(file, line)) {
        result->push_back(line);
    }
    *found_label_count = result->size();
    const int padding = 16;
    while (result->size() % padding) {
        result->emplace_back();
    }
    return kTfLiteOk;
}
```

 - 打印模型节点信息

```cpp
void PrintProfilingInfo(const tflite::profiling::ProfileEvent* e,
                        uint32_t subgraph_index, uint32_t op_index,
                        TfLiteRegistration registration) {
    // output something like
    // time (ms) , Node xxx, OpCode xxx, symblic name
    //      5.352, Node   5, OpCode   4, DEPTHWISE_CONV_2D

    LOG(INFO) << std::fixed << std::setw(10) << std::setprecision(3)
              << (e->end_timestamp_us - e->begin_timestamp_us) / 1000.0
              << ", Subgraph " << std::setw(3) << std::setprecision(3)
              << subgraph_index << ", Node " << std::setw(3)
              << std::setprecision(3) << op_index << ", OpCode " << std::setw(3)
              << std::setprecision(3) << registration.builtin_code << ", "
              << EnumNameBuiltinOperator(
                      static_cast<tflite::BuiltinOperator>(registration.builtin_code))
              << "\n";
}
```

 - 定义模型推理函数

```cpp
void RunInference(tflite::label_image::Settings* s){
    if (!s->model_name.c_str()) {
        LOG(ERROR) << "no model file name\n";
        exit(-1);
    }

// 读取.tflite模型
    std::unique_ptr<tflite::FlatBufferModel> model;
    std::unique_ptr<tflite::Interpreter> interpreter;
    model = tflite::FlatBufferModel::BuildFromFile(s->model_name.c_str());
    if (!model) {
        LOG(FATAL) << "\nFailed to mmap model " << s->model_name << "\n";
        exit(-1);
    }
    s->model = model.get();
    LOG(INFO) << "Loaded model " << s->model_name << "\n";
    model->error_reporter();
    LOG(INFO) << "resolved reporter\n";
// 生成解释器
    tflite::ops::builtin::BuiltinOpResolver resolver;

    tflite::InterpreterBuilder(*model, resolver)(&interpreter);
    if (!interpreter) {
        LOG(FATAL) << "Failed to construct interpreter\n";
        exit(-1);
    }

    interpreter->UseNNAPI(s->old_accel);
    interpreter->SetAllowFp16PrecisionForFp32(s->allow_fp16);
// 打印解释器参数，包括张量大小、输入节点名称等
    if (s->verbose) {
        LOG(INFO) << "tensors size: " << interpreter->tensors_size() << "\n";
        LOG(INFO) << "nodes size: " << interpreter->nodes_size() << "\n";
        LOG(INFO) << "inputs: " << interpreter->inputs().size() << "\n";
        LOG(INFO) << "input(0) name: " << interpreter->GetInputName(0) << "\n";

        int t_size = interpreter->tensors_size();
        for (int i = 0; i < t_size; i++) {
            if (interpreter->tensor(i)->name)
                LOG(INFO) << i << ": " << interpreter->tensor(i)->name << ", "
                          << interpreter->tensor(i)->bytes << ", "
                          << interpreter->tensor(i)->type << ", "
                          << interpreter->tensor(i)->params.scale << ", "
                          << interpreter->tensor(i)->params.zero_point << "\n";
        }
    }

    if (s->number_of_threads != -1) {
        interpreter->SetNumThreads(s->number_of_threads);
    }

// 定义输入图像参数
    int image_width = 224;
    int image_height = 224;
    int image_channels = 3;
// 读取bmp图像
    std::vector<uint8_t> in = tflite::label_image::read_bmp(s->input_bmp_name, &image_width,
                                       &image_height, &image_channels, s);

    int input = interpreter->inputs()[0];

    if (s->verbose) LOG(INFO) << "input: " << input << "\n";

    const std::vector<int> inputs = interpreter->inputs();
    const std::vector<int> outputs = interpreter->outputs();

    if (s->verbose) {
        LOG(INFO) << "number of inputs: " << inputs.size() << "\n";
        LOG(INFO) << "number of outputs: " << outputs.size() << "\n";
    }

// 创建图
    auto delegates_ = GetDelegates(s);
    for (const auto& delegate : delegates_) {
        if (interpreter->ModifyGraphWithDelegate(delegate.second.get()) !=
            kTfLiteOk) {
            LOG(FATAL) << "Failed to apply " << delegate.first << " delegate.";
        } else {
            LOG(INFO) << "Applied " << delegate.first << " delegate.";
        }
    }

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        LOG(FATAL) << "Failed to allocate tensors!";
    }

    if (s->verbose) PrintInterpreterState(interpreter.get());

// 获取输入张量元数据的维度等信息
    TfLiteIntArray* dims = interpreter->tensor(input)->dims;
    int wanted_height = dims->data[1];
    int wanted_width = dims->data[2];
    int wanted_channels = dims->data[3];

// 对图像进行resize
    switch (interpreter->tensor(input)->type) {
        case kTfLiteFloat32:
            s->input_floating = true;
            tflite::label_image::resize<float>(interpreter->typed_tensor<float>(input), in.data(),
                          image_height, image_width, image_channels, wanted_height,
                          wanted_width, wanted_channels, s);
            break;
        case kTfLiteUInt8:
            tflite::label_image::resize<uint8_t>(interpreter->typed_tensor<uint8_t>(input), in.data(),
                            image_height, image_width, image_channels, wanted_height,
                            wanted_width, wanted_channels, s);
            break;
        default:
            LOG(FATAL) << "cannot handle input type "
                       << interpreter->tensor(input)->type << " yet";
            exit(-1);
    }

// 调用解释器
    auto profiler =
            absl::make_unique<tflite::profiling::Profiler>(s->max_profiling_buffer_entries);
    interpreter->SetProfiler(profiler.get());

    if (s->profiling) profiler->StartProfiling();
    if (s->loop_count > 1)
        for (int i = 0; i < s->number_of_warmup_runs; i++) {
            if (interpreter->Invoke() != kTfLiteOk) {
                LOG(FATAL) << "Failed to invoke tflite!\n";
            }
        }
// 进行模型推理并计算运行时间
    struct timeval start_time, stop_time;
    gettimeofday(&start_time, nullptr);
    for (int i = 0; i < s->loop_count; i++) {
        if (interpreter->Invoke() != kTfLiteOk) {
            LOG(FATAL) << "Failed to invoke tflite!\n";
        }
    }
    gettimeofday(&stop_time, nullptr);
    LOG(INFO) << "invoked \n";
    LOG(INFO) << "average time: "
              << (get_us(stop_time) - get_us(start_time)) / (s->loop_count * 1000)
              << " ms \n";
// 打印运行事件
    if (s->profiling) {
        profiler->StopProfiling();
        auto profile_events = profiler->GetProfileEvents();
        for (int i = 0; i < profile_events.size(); i++) {
            auto subgraph_index = profile_events[i]->event_subgraph_index;
            auto op_index = profile_events[i]->event_metadata;
            const auto subgraph = interpreter->subgraph(subgraph_index);
            const auto node_and_registration =
                    subgraph->node_and_registration(op_index);
            const TfLiteRegistration registration = node_and_registration->second;
            PrintProfilingInfo(profile_events[i], subgraph_index, op_index,
                               registration);
        }
    }

    const float threshold = 0.001f;

    std::vector<std::pair<float, int>> top_results;

// 获取Top-N结果
    int output = interpreter->outputs()[0];
    TfLiteIntArray* output_dims = interpreter->tensor(output)->dims;
    // assume output dims to be something like (1, 1, ... ,size)
    auto output_size = output_dims->data[output_dims->size - 1];
    switch (interpreter->tensor(output)->type) {
        case kTfLiteFloat32:
            tflite::label_image::get_top_n<float>(interpreter->typed_output_tensor<float>(0), output_size,
                             s->number_of_results, threshold, &top_results, true);
            break;
        case kTfLiteUInt8:
            tflite::label_image::get_top_n<uint8_t>(interpreter->typed_output_tensor<uint8_t>(0),
                               output_size, s->number_of_results, threshold,
                               &top_results, false);
            break;
        default:
            LOG(FATAL) << "cannot handle output type "
                       << interpreter->tensor(input)->type << " yet";
            exit(-1);
    }

    std::vector<string> labels;
    size_t label_count;

    if (ReadLabelsFile(s->labels_file_name, &labels, &label_count) != kTfLiteOk)
        exit(-1);
// 打印Top-N结果
    for (const auto& result : top_results) {
        const float confidence = result.first;
        const int index = result.second;
        LOG(INFO) << confidence << ": " << index << " " << labels[index] << "\n";
    }
}

int main() {
    tflite::label_image::Settings s;
    RunInference(&s);
}

```
## 4.4 下载预训练模型
```
# Get model
curl https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_1.0_224.tgz | tar xzv -C /tmp

# Get labels
curl https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_1.0_224_frozen.tgz  | tar xzv -C /tmp  mobilenet_v1_1.0_224/labels.txt

mv /tmp/mobilenet_v1_1.0_224/labels.txt /tmp/
```
## 4.5 修改模型配置
在label_image.h中修改Settings：
```cpp
struct Settings {
  bool verbose = false;
  bool accel = false;
  bool old_accel = false;
  bool input_floating = false;
  bool profiling = false;
  bool allow_fp16 = false;
  bool gl_backend = false;
  int loop_count = 1;
  float input_mean = 127.5f;
  float input_std = 127.5f;
  string model_name = "/home/ai/CLionProjects/tflite/mobilenet_v1_1.0_224/mobilenet_v1_1.0_224.tflite";
  tflite::FlatBufferModel* model;
  string input_bmp_name = "/home/ai/CLionProjects/tflite/grace_hopper.bmp";
  string labels_file_name = "/home/ai/CLionProjects/tflite/mobilenet_v1_1.0_224/labels.txt";
  string input_layer_type = "uint8_t";
  int number_of_threads = 4;
  int number_of_results = 5;
  int max_profiling_buffer_entries = 1024;
  int number_of_warmup_runs = 2;
};
```
## 4.6 运行实例
Top5分类结果输出如下：
```
Loaded model /tmp/mobilenet_v1_1.0_224.tflite
resolved reporter
invoked
average time: 68.12 ms
0.860174: 653 653:military uniform
0.0481017: 907 907:Windsor tie
0.00786704: 466 466:bulletproof vest
0.00644932: 514 514:cornet, horn, trumpet, trump
0.00608029: 543 543:drumstick
```
结果显示该图像被正确分类，平均耗时68.12ms，速度非常快！

# 5 TensorFlow Lite模型使用通用流程（以CPM算法为例）
## 5.1 流程示意
```mermaid
flowchat
st=>start: 加载tflite模型
e=>end: 从输出tensor中取出推理结果
op1=>operation: 我的操作
op2=>operation: 加载所有tensor
op3=>operation: 获取输入输出层tensor信息
op4=>operation: 加载图像数据至输入tensor中
op5=>operation: 调用invoke方法进行推理
st->op1->op2->op3->op4->op5->e
```
## 5.2 主要函数说明
|函数方法| 作 用 |
|:--:|:--:|
| tflite::FlatBufferModel::BuildFromFile(filename) | 加载tflite模型 |
|tflite::ops::builtin::BuiltinOpResolver resolver;|生成resolver|
|std::unique_ptr<<tflite::Interpreter>>interpreter;|创建interpreter|
|tflite::InterpreterBuilder(*model, resolver)(&interpreter);|生成interpreter|
|interpreter->AllocateTensors();|加载所有tensor|
|float* input = interpreter->typed_input_tensor<float>(0);|取出输入tensor|
|interpreter->Invoke();|调用模型进行推理|
|float* output = interpreter->typed_output_tensor<float>(0);|取出输出tensor|

## 5.3 操作流程
### 5.3.1 CPM算法介绍
CPM是一个关键点检测算法，示意图如下所示：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191101164303987.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MjQ5OTIzNg==,size_16,color_FFFFFF,t_70)
该示例中使用的模型使用6个stage，模型输出6个结果，每个结果的形状为（1，46，46，10），即返回10张46x46的概率图，分别预测10个点的位置。

### 5.3.2 加载模型
```cpp
// load model file and build the interpreter.
    if (!s->model_name.c_str()) {
        LOG(ERROR) << "no model file name\n";
        exit(-1);
    }

    std::unique_ptr<tflite::FlatBufferModel> model;
    std::unique_ptr<tflite::Interpreter> interpreter;
    model = tflite::FlatBufferModel::BuildFromFile(s->model_name.c_str());
    std::cout << s->model_name << std::endl;
    if (!model) {
        LOG(FATAL) << "\nFailed to mmap model " << s->model_name << "\n";
        exit(-1);
    }

    // invoke error reporter
    if (s->verbose){
        model->error_reporter();
    }
    LOG(INFO) << "resolved reporter\n";

    tflite::ops::builtin::BuiltinOpResolver resolver;

    tflite::InterpreterBuilder(*model, resolver)(&interpreter);
    if (!interpreter) {
        LOG(FATAL) << "Failed to construct interpreter\n";
        exit(-1);
    }
```
在此段代码中，程序首先从s->model_name.c_str()读取tflite模型，然后创建interpreter，之后的所有操作都要基于interpreter中定义的方法进行。

### 5.3.3 加载所有tensor
```cpp
    // allocate all of the tensors
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        LOG(FATAL) << "Failed to allocate tensors!";
    }
```
上述代码会加载模型中所有定义的tensor，此步必须先执行，否则后续操作无效。

### 5.3.4 获取输入输出信息
```cpp
// fetch the inputs' tensor and outputs' tensor
    const std::vector<int> inputs = interpreter->inputs();
    const std::vector<int> outputs = interpreter->outputs();
    // print the number of inputs and outputs.
    LOG(INFO) << "number of inputs: " << inputs.size() << "\n";
    LOG(INFO) << "number of outputs: " << outputs.size() << "\n";
```
调用interpreter中的inputs()、outputs()方法，获取模型输入、输出信息

### 5.3.5 构建输入
```cpp
int input = interpreter->inputs()[0];
// get the dims of input tensor
TfLiteIntArray* dims = interpreter->tensor(input)->dims;
    int wanted_height = dims->data[1];
    int wanted_width = dims->data[2];
    int wanted_channels = dims->data[3];

    // fill input tensor with image data
    std::string imgpath = "/home/ai/Downloads/00010100000006E8-20190906-073144-359.jpg";
    cv::Mat img = cv::imread(imgpath);
    int cols = wanted_width;
    int rows = wanted_height;
    cv::Size dsize = cv::Size(cols, rows);
    cv::Mat img_rs;
    cv::resize(img, img_rs, dsize);
    cv::cvtColor(img_rs, img_rs, cv::COLOR_BGR2RGB);

    auto img_inputs = interpreter->typed_tensor<float>(input);
    for (int i = 0; i < img_rs.cols * img_rs.rows * 3; i++){
        img_inputs[i] =  img_rs.data[i]/ 255.0;
    }
```
typed_tensor方法会返回一个经过固定数据类型转换的tensor指针，使用opencv进行图像读取以及预处理，将其逐像素填充到输入tensor中

### 5.3.6 调用模型
```cpp
// invoke the model to inference
    struct timeval start_time, stop_time;
    gettimeofday(&start_time, nullptr);

    if (interpreter->Invoke() != kTfLiteOk) {
        LOG(FATAL) << "Failed to invoke tflite!\n";
    }

    gettimeofday(&stop_time, nullptr);
    LOG(INFO) << "invoked \n";
    LOG(INFO) << "average time: "
              << (get_us(stop_time) - get_us(start_time)) / (s->loop_count * 1000)
              << " ms \n";
```
调用invoke方法进行推理，并打印运行时间，调用一次invoke方法，即进行一次推理。

### 5.3.7 取出输出结果
```cpp
// get the dims of outputs
    // the index of line 175 depends on the which output you want.
    // output = interpreter->outputs()[i];
    int output = interpreter->outputs()[5];
    TfLiteIntArray* output_dims = interpreter->tensor(output)->dims;
    auto output_size = output_dims->data[output_dims->size - 1];

    cout << "Output dims" << endl;
    for (int i=0;i<4;i++){
        cout<<output_dims->data[i]<<endl;
    }

    // get outputs
    auto* score_map = interpreter->typed_output_tensor<float>(5);
```
typed_output_tensor方法会返回模型的输出，index根据需求设置，此处有6个输出，只需要最后一个stage输出的结果，故设置index=5。

### 5.3.8 输出结果后处理
模型返回的结果会默认进行展平，形成一维数组，展平的顺序如下：
![在这里插入图片描述](https://img-blog.csdnimg.cn/2019110116544650.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MjQ5OTIzNg==,size_16,color_FFFFFF,t_70)
在该示例中需要返回10张46x46的概率图，处理程序如下：
```cpp
auto* feature_map = new float[46*46];

cv::Mat ori_img = cv::imread("/home/ai/Downloads/00010100000006E8-20190906-073144-359.jpg");

    for (int j = 0; j < 10;j++){
        int index = 0;
        for (int i=j;i<46*46*10;i+=10){
            feature_map[index] = (float)score_map[i];
            index++;
        }
        cv::Mat feature_map_mat(46,46,CV_32FC1, feature_map);
        double countMinVal = 0;
        double countMaxVal = 0;
        cv::Point minPoint;
        cv::Point maxPoint;
        cv::minMaxLoc(feature_map_mat, &countMinVal, &countMaxVal, &minPoint, &maxPoint);

        cout << " -- " << maxPoint.x  * 1280/ 46  << " - " << minPoint.y  * 720 / 46 << endl;
        auto px = (float)maxPoint.x;
        auto py = (float)maxPoint.y;
        float x = px / 46 * 1280;
        float y = py / 46 * 720 ;
        int xx = (int)x;
        int yy = (int)y;

        cv::circle(ori_img, cv::Point(xx,yy), 20, cv::Scalar(255 - j* 20,100 + j*10,0), -1);
    }

    cv::imshow("", ori_img);
    cv::waitKey(0);
```
### 5.3.8 标记结果
标记结果如下：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191101162058695.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MjQ5OTIzNg==,size_16,color_FFFFFF,t_70)
### 5.3.9 完整程序
```cpp
#include <sys/time.h>   // NOLINT(build/include_order)
#include <iostream>
#include <cstdlib>
#include <iomanip>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "tensorflow/lite/model.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/optional_debug_tools.h"
#include "tensorflow/lite/string_util.h"
#include "tensorflow/lite/profiling/profiler.h"
#include "settings.h"
#include "utils.h"

#include "opencv2/opencv.hpp"

using namespace std;

#define LOG(x) std::cerr

double get_us(struct timeval t) { return (t.tv_sec * 1000000 + t.tv_usec); }

using TfLiteDelegatePtr = tflite::Interpreter::TfLiteDelegatePtr;
using TfLiteDelegatePtrMap = std::map<std::string, TfLiteDelegatePtr>;

TfLiteDelegatePtr CreateGPUDelegate(tflite::cpm::Settings* s) {
#if defined(__ANDROID__)
    TfLiteGpuDelegateOptionsV2 gpu_opts = TfLiteGpuDelegateOptionsV2Default();
  gpu_opts.inference_preference =
      TFLITE_GPU_INFERENCE_PREFERENCE_SUSTAINED_SPEED;
  gpu_opts.is_precision_loss_allowed = s->allow_fp16 ? 1 : 0;
  return evaluation::CreateGPUDelegate(s->model, &gpu_opts);
#else
    return tflite::evaluation::CreateGPUDelegate(s->model);
#endif
}

TfLiteDelegatePtrMap GetDelegates(tflite::cpm::Settings* s) {
    TfLiteDelegatePtrMap delegates;
    if (s->gl_backend) {
        auto delegate = CreateGPUDelegate(s);
        if (!delegate) {
            LOG(INFO) << "GPU acceleration is unsupported on this platform.";
        } else {
            delegates.emplace("GPU", std::move(delegate));
        }
    }

    if (s->accel) {
        auto delegate = tflite::evaluation::CreateNNAPIDelegate();
        if (!delegate) {
            LOG(INFO) << "NNAPI acceleration is unsupported on this platform.";
        } else {
            delegates.emplace("NNAPI", tflite::evaluation::CreateNNAPIDelegate());
        }
    }
    return delegates;
}

void RunInference(tflite::cpm::Settings* s){
    // load model file and build the interpreter.
    if (!s->model_name.c_str()) {
        LOG(ERROR) << "no model file name\n";
        exit(-1);
    }

    std::unique_ptr<tflite::FlatBufferModel> model;
    std::unique_ptr<tflite::Interpreter> interpreter;
    model = tflite::FlatBufferModel::BuildFromFile(s->model_name.c_str());
    std::cout << s->model_name << std::endl;
    if (!model) {
        LOG(FATAL) << "\nFailed to mmap model " << s->model_name << "\n";
        exit(-1);
    }

    // invoke error reporter
    if (s->verbose){
        model->error_reporter();
    }
    LOG(INFO) << "resolved reporter\n";

    tflite::ops::builtin::BuiltinOpResolver resolver;

    tflite::InterpreterBuilder(*model, resolver)(&interpreter);
    if (!interpreter) {
        LOG(FATAL) << "Failed to construct interpreter\n";
        exit(-1);
    }

    // allocate all of the tensors
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        LOG(FATAL) << "Failed to allocate tensors!";
    }

    // print the tensors which interpreter has got
    if (s->verbose) PrintInterpreterState(interpreter.get());

    // invoke the NNAPI and set the computing precision
    interpreter->UseNNAPI(s->old_accel);
    interpreter->SetAllowFp16PrecisionForFp32(s->allow_fp16);

    // If you have acceleration device, then load the graph into computing device
    auto delegates_ = GetDelegates(s);
    for (const auto& delegate : delegates_) {
        if (interpreter->ModifyGraphWithDelegate(delegate.second.get()) !=
            kTfLiteOk) {
            LOG(FATAL) << "Failed to apply " << delegate.first << " delegate.";
        } else {
            LOG(INFO) << "Applied " << delegate.first << " delegate.";
        }
    }

    // set the number of threads
    if (s->number_of_threads != -1) {
        interpreter->SetNumThreads(s->number_of_threads);
    }

    // fetch the inputs' tensor and outputs' tensor
    const std::vector<int> inputs = interpreter->inputs();
    const std::vector<int> outputs = interpreter->outputs();

    // print the number of inputs and outputs.
    LOG(INFO) << "number of inputs: " << inputs.size() << "\n";
    LOG(INFO) << "number of outputs: " << outputs.size() << "\n";

    // construct the input vector
    int input = interpreter->inputs()[0];
    TfLiteIntArray* dims = interpreter->tensor(input)->dims;
    int wanted_height = dims->data[1];
    int wanted_width = dims->data[2];
    int wanted_channels = dims->data[3];

    // fill input tensor with image data
    std::string imgpath = "/home/ai/Downloads/00010100000006E8-20190906-073144-359.jpg";
    cv::Mat img = cv::imread(imgpath);
    int cols = wanted_width;
    int rows = wanted_height;
    cv::Size dsize = cv::Size(cols, rows);
    cv::Mat img_rs;
    cv::resize(img, img_rs, dsize);
    cv::cvtColor(img_rs, img_rs, cv::COLOR_BGR2RGB);

    auto img_inputs = interpreter->typed_tensor<float>(input);
    for (int i = 0; i < img_rs.cols * img_rs.rows * 3; i++){
        img_inputs[i] =  img_rs.data[i]/ 255.0;
    }

    // invoke the model to inference
    struct timeval start_time, stop_time;
    gettimeofday(&start_time, nullptr);

    if (interpreter->Invoke() != kTfLiteOk) {
        LOG(FATAL) << "Failed to invoke tflite!\n";
    }

    gettimeofday(&stop_time, nullptr);
    LOG(INFO) << "invoked \n";
    LOG(INFO) << "average time: "
              << (get_us(stop_time) - get_us(start_time)) / (s->loop_count * 1000)
              << " ms \n";

    // get the dims of outputs
    // the index of line 175 depends on the which output you want.
    // output = interpreter->outputs()[i];
    int output = interpreter->outputs()[5];
    TfLiteIntArray* output_dims = interpreter->tensor(output)->dims;
    auto output_size = output_dims->data[output_dims->size - 1];

    cout << "Output dims" << endl;
    for (int i=0;i<4;i++){
        cout<<output_dims->data[i]<<endl;
    }

    // get outputs
    auto* score_map = interpreter->typed_output_tensor<float>(5);

    auto* feature_map = new float[46*46];

    cv::Mat ori_img = cv::imread("/home/ai/Downloads/00010100000006E8-20190906-073144-359.jpg");

    for (int j = 0; j < 10;j++){
        int index = 0;
        for (int i=j;i<46*46*10;i+=10){
            feature_map[index] = (float)score_map[i];
            index++;
        }
        cv::Mat feature_map_mat(46,46,CV_32FC1, feature_map);
        double countMinVal = 0;
        double countMaxVal = 0;
        cv::Point minPoint;
        cv::Point maxPoint;
        cv::minMaxLoc(feature_map_mat, &countMinVal, &countMaxVal, &minPoint, &maxPoint);

        cout << " -- " << maxPoint.x  * 1280/ 46  << " - " << minPoint.y  * 720 / 46 << endl;

        auto px = (float)maxPoint.x;
        auto py = (float)maxPoint.y;
        float x = px / 46 * 1280;
        float y = py / 46 * 720 ;
        int xx = (int)x;
        int yy = (int)y;

        cv::circle(ori_img, cv::Point(xx,yy), 20, cv::Scalar(255 - j* 20,100 + j*10,0), -1);
    }

    cv::imshow("", ori_img);
    cv::waitKey(0);


}

int main() {
    tflite::cpm::Settings s;
    RunInference(&s);
}
```
