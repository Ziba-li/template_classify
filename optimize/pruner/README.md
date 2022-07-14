基于情绪分类的四分类模型。

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

- ##### 蒸馏为3层

~~~
Device: cuda:0
Mean inference time: 58.96ms
Standard deviation: 9.06ms
accuracy：82.09982788296041
~~~



