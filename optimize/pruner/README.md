**以基于情绪分类的四分类模型为例：**

### 单条数据

- ##### 原始bert

~~~
Device: cuda:0
Mean inference time: 18.27ms
Standard deviation: 4.72ms
accuracy：82.84566838783705
~~~

- ##### pruned_H8.0F2048n_iters8

~~~
Device: cuda:0
Mean inference time: 16.17ms
Standard deviation: 5.31ms
accuracy：81.87033849684452
~~~

- ##### pruned_H6.0F1536n_iters16

~~~
Device: cuda:0
Mean inference time: 17.08ms
Standard deviation: 2.49ms
accuracy：74.69879518072288
~~~

- ##### pruned_H6.0F1536n_iters16ffn-first

~~~
Device: cuda:0
Mean inference time: 6.89ms
Standard deviation: 0.12ms
accuracy：81.41135972461274
~~~

- ##### pruned_H6.0F1536n_iters32ffn-first

~~~
Device: cuda:0
Mean inference time: 7.01ms
Standard deviation: 0.21ms
accuracy：76.01835915088927
~~~

- ##### 蒸馏为3层

~~~
Device: cuda:0
Mean inference time: 5.94ms
Standard deviation: 5.30ms
accuracy：82.09982788296041
~~~



### 多条数据(30条)

- ##### 原始bert

~~~
Device: cuda:0
Mean inference time: 233.47ms
Standard deviation: 13.70ms
accuracy：82.84566838783705
~~~

- ##### pruned_H8.0F2048n_iters8

~~~
Device: cuda:0
Mean inference time: 189.19ms
Standard deviation: 15.93ms
accuracy：81.87033849684452
~~~

- ##### pruned_H6.0F1536n_iters16

~~~
Device: cuda:0
Mean inference time: 164.29ms
Standard deviation: 8.27ms
accuracy：74.69879518072288
~~~

- ##### pruned_H6.0F1536n_iters16ffn-first

~~~
Device: cuda:0
Mean inference time: 76.30ms
Standard deviation: 1.04ms
accuracy：81.41135972461274
~~~

- ##### pruned_H6.0F1536n_iters32ffn-first

~~~
Device: cuda:0
Mean inference time: 65.07ms
Standard deviation: 2.01ms
accuracy：76.01835915088927
~~~

- ##### 蒸馏为3层

~~~
Device: cuda:0
Mean inference time: 58.96ms
Standard deviation: 9.06ms
accuracy：82.09982788296041
~~~

### 结论：

transformer裁剪中：`target_ffn_size=1536, target_num_of_heads=6` 精度损耗较为严重，即使设置32轮迭代依旧很低，建议采用：`target_ffn_size=2048, target_num_of_heads=8` 
