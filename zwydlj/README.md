# 基于LSTM的阅读理解模型
该模型为用Attention整合题干和选项，然后进行预测的模型。

## 准备工作

使用`pip install -r Freeze.txt`安装运行所需环境。

## 1. 生成词表

数据预处理命令： 
``python3 wordcutter.py --data data --output dataprocessed --gen_word2id``

将在data目录下生成一个词表``word2id.txt``，用于对文本编码。

## 2. 训练
训练命令： 
``python3 train.py --config config/model.config --gpu 0``。

## 3. 生成结果文件
推理命令：
``python3 test.py  --config config/model.config --gpu 0 --checkpoint output/model/model.bin --output output/result.csv``

生成的result.csv即为结果

# 附录
对config/model.config简单说明
```
[train] #train parameters
epoch = 10
batch_size = 8
shuffle = True
reader_num = 0

optimizer = adam
learning_rate = 8e-4
step_size = 1
lr_multiplier = 0.9995
gradient_accumulation_steps = 1
max_grad_norm=1e-2

[eval] 
batch_size = 8
shuffle = False
reader_num = 0

[data]
train_dataset_type=JsonListFromFiles
train_formatter_type=WordFormatter
train_data_path=data              #训练数据集目录
train_file_list=train_train.json  #训练数据集文件

valid_dataset_type=JsonListFromFiles
valid_formatter_type=WordFormatter
valid_data_path=data               #验证集目录
valid_file_list=train_val.json     #验证集文件

test_dataset_type=JsonListFromFiles
test_formatter_type=WordFormatter
test_data_path=data                 #测试集目录
test_file_list=validation.json      #测试集文件

max_question_len = 2048
max_option_len = 64

word2id = data/word2id.txt         #通过训练集生成的词编码文件

[model]
model_name=Model
dropout=0.05

hidden_size = 256                  #LSTM 隐含层
bi_direction = True                #双向LSTM
num_layers = 1                     #LSTM层数

[output]
output_time=1                     #多少轮模型保存义词
test_time=2                       #每多少轮运行验证结果
model_path=output/model           #模型保存目录
model_name=attention
tensorboard_path=output/tensorboard

accuracy_method=SingleLabelTop1  
output_function=Basic
output_value=micro_precision      #micro accuracy评价方法
```
# 运行范例

```
0      train  711/710     5:56/ 0:00    1.373   {"micro_precision": 0.287}
0      valid  79/79       0:28/ 0:00    1.370   {"micro_precision": 0.295}
1      train  711/710     5:34/ 0:00    1.353   {"micro_precision": 0.328}
2      train  711/710     5:37/ 0:00    1.235   {"micro_precision": 0.46}
2      valid  79/79       0:29/ 0:00    1.425   {"micro_precision": 0.297}
```