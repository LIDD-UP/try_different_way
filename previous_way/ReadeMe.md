1：对我们当前的数据：一共有15列排除4列不用：
    分别是：tradetypeid,buildingtypeid,delislingDate,postalcode
当然也可以使用id号来做预测，我这里用的是name，理论上结果应该是
相同的；

2：对于地址数据，其实有两个，字符表示的详细地址，以及经纬度表示的精确地址；
    1）：对于省份城市和address可以通过hash_bucket进行分桶，由于不知道该地方
    的具体情况，省份有多少，城市有多少，地址如何表示，暂定用hash_bucket来定义，如果类别
    不多也可以用 categorical_column_with_vocabulary_list进行，但是这三个特征分离之后表示的
    意义并没有他们三个组合起来大，所以我考虑使用cross_column把他们三个特征组合起来，表示一个具体的地址；
    
    2）同理经纬度也是一样，可以使用bucketized_column先进行分桶，然后再用cross_column进行组合，表示一定区域范围，
    比单独拿一个精度或者是维度值更加有意义；

3:对于价格和期望价格，我采用bucketized_colum分桶的方式使用，观察数据得到，对于小于10000的交易类型都是出租，对于介于10000到100000的
一般都是NObuilding的房屋类型，对于高于100000的一般都是出售类型；
所以前两个做两个区间，高于100000的每100000为一个区间；
4：期望价格（expectedprice）同理

5：房屋类型：观察数据有19个，有点多，用hash_bucket进行分桶，具体分类多还是少没有具体定义，还是要看模型的具体情况，这里暂定使用这个方法定义特征；

6：对于交易类型只有两个那就使用categorical_column_with_vocabulary_list了
7：最后一个对于日期的处理：
我的方式是把日期分为三个，分别是年月和日；
最后通过pandas再把他处理成了DataFrame的结构形式，列明分别是year,month,day
数据类型是float32类型的数，
对于year观察数据只有2017 和2018类型的，所以用categorical_column_with_vocabulary_list 

对于month和day都采用bucketized_column分桶，
month分为四个季度，
day分为三旬，

特征处理我觉的差不多就是这样了，

对于模型的优化函数还不是很了解，目前就使用的默认的，
还有一个就是权重参数了，官方上有weighted_categorical_column，linear_model这两个加权重的，第一个看字面上是对分类权重进行加权，第二个是对所有特征进行线性加权；
但是我感觉没啥必要，因为，在模型学习的过程中，它自动都会自己调节权重，不知道设置这两个东西是干嘛的，

其实在用tensorflow模块下的estimator模块是，可以直接用sklearn包下的train_data_splite
from sklearn.model_selection import train_test_split,这个可以用来切分训练数据和测试数据；
