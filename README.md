# 基于深度学习的分类模板

## 1. 环境要求

~~~
pip install -r requirments
~~~

## 1.1 镜像配置

~~~
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
~~~



## 2. 数据样式

数据表头为：

* `index	sentence	label`
* 数据和表头中间用 `\t` 分隔

~~~
index	sentence	label
0	气死姐姐了，快二是阵亡了吗，尼玛，一个半小时过去了也没上车	3
1	妞妞啊，今天又承办了一个发文登记文号是126~嘻~么么哒~晚安哟	0
2	这里还值得注意另一个事实，就是张鞠存原有一个东溪草堂为其读书处。	1
3	这在前华约国家(尤其是东德)使用R-73的首次联合演习期间，被一些北约组织的飞行员所证实。	1
4	每天都以紧张的心情工作，真的是太累，太不放松了，想要爆发一下	3
~~~

## 3. 训练

基于 `huggingface` 的 `transformers` 的 `example` 修改。

参考链接：https://github.com/huggingface/transformers/tree/main/examples/pytorch

训练脚本：`train` 文件夹下的 `run.sh`

## 4.模型优化

基于已训练完成的 `pytorch` 模型进行 `ONNX` `蒸馏` `量化` `裁剪` 操作。

- ##### 建议优化顺序：

  1. 将已训练好的 `pytorch` 进行 `transformer裁剪`，将前馈全连接层设置为 `2048`，头的个数设置为 `8` 。
  2. 对裁剪后的模型进行 `蒸馏`，其中学生模型可以设置为 `hfl-rbt3` 。
  3. 对蒸馏后的模型转为 `ONNX`，然后做 `FP16` 和 `图融合` 的操作。

- ##### 以上每个步骤必须有评估指标，倘若评估结果无法达到产品落地要求，可适当减少优化步骤。

**<font color=red>注：以下过程必须进行评价指标的比较</font>**

### 4.1 模型加速

模型加速采用 `ONNXRuntime` ，具体参考：[ONNXRUNTIME](https://onnxruntime.ai/)

1. 导出 `ONNX` ：`optimize` 文件夹下的 `acceleration` 文件夹下的 `export_pytorch2onnx.py`

   > 需要设定已训练好的 `pytorch` 模型**文件夹路径**，以及转换完成的 `onnx` 模型**文件路径**。

2. 将已导出的 `ONNX` 模型进行 `FP16` 转换：`optimize` 文件夹下的 `acceleration` 文件夹下的 `export_onnx2fp16.sh`

   > 需要设定**已转换**完成的 `onnx` 模型**文件路径**，以及以及转换完成的 `fp16.onnx` 模型**文件路径**。

### 4.2 模型蒸馏

使用哈工大讯飞联合实验室出品的 `TextBrewer` 作为模型蒸馏工具，经实验：蒸馏后的模型性能提升效果非常明显，精度降低 0.5~1 个点，但是蒸馏时间较长。

参考链接：[TextBrewer](http://textbrewer.hfl-rc.com/)

1. 设置蒸馏脚本参数：`optimize` 文件夹下的 `distill` 文件夹下的 `distill.sh`

   ~~~shell
   # 必填路径参数
   BERT_DIR=str  # 教师模型文件夹
   OUTPUT_ROOT_DIR=str  # 输出文件夹
   DATA_ROOT_DIR=str  # 数据位置
   trained_teacher_model=str  # 教师模型文件
   student_init_model=str  # 学生模型文件
   STUDENT_CONF_DIR=str   # 学生模型配置文件夹
   
   ~~~

   例如蒸馏一个已训练完成的 4 分类模型：

   ~~~shell
   # 必填路径参数
   BERT_DIR=../../model/emotional_cls_4  # 教师模型文件夹
   OUTPUT_ROOT_DIR=output_root_dir  # 输出
   DATA_ROOT_DIR=../../data  # 数据位置
   trained_teacher_model=../../model/emotional_cls_4/pytorch_model.bin  # 教师模型文件
   student_init_model=../../model/hfl_rbt3/pytorch_model.bin  # 学生模型文件
   STUDENT_CONF_DIR=../../model/hfl_rbt3/  # 学生模型文件夹
   
   ~~~

   <font color=red>需要根据教师模型设置 `num_labels` 的数量，且蒸馏完成后需要调整学生模型的`config`文件的 `num_labels` 的数量。</font>

2. 设置完成后运行 `distill.sh` 文件。

### 4.3 模型量化

参考链接：https://onnxruntime.ai/docs/performance/quantization.html

模型量化过程主要将权重转换为 `INT8`，在最终指标损失 0.1~0.3% 的基础上，提升模型推理速度，目前只能用于CPU 服务器，GPU 服务部署无法使用。

1. `ONNX`量化：`optimize` 文件夹下的 `quantify` 文件夹下的 `quantize.py`

   > 需要设定 `.onnx` 模型**文件路径**，以及以及量化完成的 `.onnx` 模型**文件路径**。

2. 蒸馏完成后量化：`optimize` 文件夹下的 `quantify` 文件夹下的 `distilled_quantify.py`

   > 按照文件内的 `main` 函数注释填写相关参数

### 4.4 模型剪裁

试验完成后再写。



### 4.5 评估

此处评估均基于 `ONNX` 模型：`optimize` 文件夹下的 `evaluate_onnx.py` 文件

> 需要设定待评估的 `onnx` 模型**文件路径**，以及评估数据**文件路径**

对于优化后模型，需要进行评测，得出每次优化后的评价指标的变化，用于最终的模型决策。

# 5. 部署

部署脚本位置：`deploy` 下的 `run_app.sh`，根据所选模型调整脚本中对应的 `模型类型`

## 5.1 接口样式

### 5.1.1 单条：

- ##### request

  ~~~python
  {
      "text": str
  }
  ~~~

- ##### response

  ~~~python
  {
      "categories": int,
      "probabilities": float
  }
  ~~~

### 5.1.2 批量：

- ##### request

  ~~~python
  {
      "text": List[str]
  }
  ~~~

- ##### response

  ~~~python
  {
      "categories":  List[int],
      "probabilities":  List[float]
  }
  ~~~

  

























