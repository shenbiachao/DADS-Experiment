接口函数load_data()
返回格式为dataset_a, dataset_u, dataset_test, test_label
分别对应异常数据集，未标记数据集，测试集和测试数据标签，前三个为tensor类型，最后一个为list类型

按照dplan文中的实验，dataset_u中异常数据比例固定为0.02，dataset_test中异常比例则与原数据集一致。
但我认为存在两个问题
1. dataset_u中的异常比例为什么要与dataset_test不同，这个没有太大意义
2. 文中说dataset_test的异常比例为0.96%~5.23%，但是这明显与HAR数据集不符（HAR达到了百分之二十几）

所以我这边将dataset_u和dataset_test的异常比例均不做调整，与原数据集保持一致
但是HAR数据集除外，这里我将HAR数据集的dataset_a和dataset_u都固定在了0.02。

load_data会调用根据config.py中的manual_dataset这个参数调用load_manual_data或load_original_data
前者用于返回调整后的数据，后者用于返回调整前的数据。
所以根据上面的分析，我计划在实验中对除HAR数据集外的其他数据集都设置manual_dataset=False，而对于HAR则为True。

normalization参数表示是否需要对数据做规范化预处理，在实验后我认为设置为True会好一些。
对于除cardio外的其他四个数据集，normalization都是合理且有效的
不过对于cardio数据集，有些标量的属性并不适合简单的normalization，我认为还需要再斟酌一下，所以cardio的实验可以放到后面。