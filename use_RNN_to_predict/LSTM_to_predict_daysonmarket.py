#-*- coding:utf-8 _*-  
""" 
@author:Administrator
@file: LSTM_to_predict_daysonmarket.py
@time: 2018/7/30
"""

from sklearn.datasets import load_boston

from sklearn import preprocessing
import tensorflow as tf
import numpy as np
np.set_printoptions(suppress=True)
import pandas as pd
# from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import matplotlib as mpl
mpl.rcParams['font.sans-serif']=['SimHei'] #指定默认字体 SimHei为黑体
mpl.rcParams['axes.unicode_minus']=False #用来正常显示负号

# 波士顿房价数据
# boston = load_boston()
# boston_df = pd.DataFrame(boston.target)
# boston_df.to_csv('boston.csv')
# x = boston.data
# y = boston.target
# print(x[1:100,:])
# print(y)

# 载入训练集数据
data = pd.read_csv('month_456_1.csv')
data = data.dropna(axis=0)
x = np.array(data[['longitude','latitude','price','buildingTypeId','bedrooms']])
y = np.array(data['daysOnMarket'])
print(x.shape,y.shape)
# 载入测试集数据
data_test = pd.read_csv('test_data_6_1.csv')
data_test = data_test.dropna(axis=0)
x_test = np.array(data_test[['longitude','latitude','price','buildingTypeId','bedrooms']])
y_test = np.array(data_test['daysOnMarket'])
print(x_test.shape,y_test.shape)

'''
波士顿数据X: (125515, 5)
波士顿房价Y: (125515,)
'''


print('波士顿数据X:', x.shape)  # (506, 13)
# print(x[::100])
print('波士顿房价Y:', y.shape)
# print(y[::100])
# 数据标准化 训练集数据
ss_x = preprocessing.StandardScaler()
train_x = ss_x.fit_transform(x)
ss_y = preprocessing.StandardScaler()
train_y = ss_y.fit_transform(y.reshape(-1, 1))


# 数据标准化测试集数据
ss_x_test = preprocessing.StandardScaler()
train_x_test = ss_x.fit_transform(x_test)
ss_y_test = preprocessing.StandardScaler()
train_y_test = ss_y.fit_transform(y_test.reshape(-1, 1))






'''
(125515, 5) (125515,)
(862, 5) (862,)
'''

BATCH_START = 0  # 建立 batch data 时候的 index
TIME_STEPS = 10  # backpropagation through time 的 time_steps
BATCH_SIZE = 86
INPUT_SIZE = 5  # sin 数据输入 size
OUTPUT_SIZE = 1  # cos 数据输出 size
CELL_SIZE = 10  # RNN 的 hidden unit size
LR = 0.006  # learning rate

def get_batch_boston_test():
    global train_x_test, train_y_test, BATCH_START, TIME_STEPS
    x_part1 = train_x_test[BATCH_START: BATCH_START + TIME_STEPS * BATCH_SIZE]
    print(x_part1.shape)
    y_part1 = train_y_test[BATCH_START: BATCH_START + TIME_STEPS * BATCH_SIZE]
    print('时间段=', BATCH_START, BATCH_START + TIME_STEPS * BATCH_SIZE)

    seq = x_part1.reshape((BATCH_SIZE, TIME_STEPS, INPUT_SIZE))
    res = y_part1.reshape((BATCH_SIZE, TIME_STEPS, 1))

    BATCH_START += TIME_STEPS

    # returned seq, res and xs: shape (batch, step, input)
    # np.newaxis 用来增加一个维度 变为三个维度，第三个维度将用来存上一批样本的状态
    return [seq, res]





def get_batch_boston():
    global train_x, train_y, BATCH_START, TIME_STEPS
    x_part1 = train_x[BATCH_START: BATCH_START + TIME_STEPS * BATCH_SIZE]
    y_part1 = train_y[BATCH_START: BATCH_START + TIME_STEPS * BATCH_SIZE]
    print('时间段=', BATCH_START, BATCH_START + TIME_STEPS * BATCH_SIZE)

    seq = x_part1.reshape((BATCH_SIZE, TIME_STEPS, INPUT_SIZE))
    res = y_part1.reshape((BATCH_SIZE, TIME_STEPS, 1))

    BATCH_START += TIME_STEPS

    # returned seq, res and xs: shape (batch, step, input)
    # np.newaxis 用来增加一个维度 变为三个维度，第三个维度将用来存上一批样本的状态
    return [seq, res]


def get_batch():
    global BATCH_START, TIME_STEPS
    # xs shape (50batch, 20steps)
    xs = np.arange(BATCH_START, BATCH_START + TIME_STEPS * BATCH_SIZE).reshape((BATCH_SIZE, TIME_STEPS)) / (10 * np.pi)
    print('xs.shape=', xs.shape)
    seq = np.sin(xs)
    res = np.cos(xs)
    BATCH_START += TIME_STEPS
    # import matplotlib.pyplot as plt
    # plt.plot(xs[0, :], res[0, :], 'r', xs[0, :], seq[0, :], 'b--')
    # plt.show()
    print('增加维度前:', seq.shape)
    print(seq[:2])
    print('增加维度后:', seq[:, :, np.newaxis].shape)
    print(seq[:2])
    # returned seq, res and xs: shape (batch, step, input)
    # np.newaxis 用来增加一个维度 变为三个维度，第三个维度将用来存上一批样本的状态
    return [seq[:, :, np.newaxis], res[:, :, np.newaxis], xs]


class LSTMRNN(object):
    def __init__(self, n_steps, input_size, output_size, cell_size, batch_size):
        '''
        :param n_steps: 每批数据总包含多少时间刻度
        :param input_size: 输入数据的维度
        :param output_size: 输出数据的维度 如果是类似价格曲线的话，应该为1
        :param cell_size: cell的大小
        :param batch_size: 每批次训练数据的数量
        '''
        self.n_steps = n_steps
        self.input_size = input_size
        self.output_size = output_size
        self.cell_size = cell_size
        self.batch_size = batch_size
        with tf.name_scope('inputs'):
            self.xs = tf.placeholder(tf.float32, [None, n_steps, input_size], name='xs')  # xs 有三个维度
            self.ys = tf.placeholder(tf.float32, [None, n_steps, output_size], name='ys')  # ys 有三个维度
        with tf.variable_scope('in_hidden'):
            self.add_input_layer()
        with tf.variable_scope('LSTM_cell'):
            self.add_cell()
        with tf.variable_scope('out_hidden'):
            self.add_output_layer()
        with tf.name_scope('cost'):
            self.compute_cost()
        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(LR).minimize(self.cost)

    # 增加一个输入层
    def add_input_layer(self, ):
        # l_in_x:(batch*n_step, in_size),相当于把这个批次的样本串到一个长度1000的时间线上，每批次50个样本，每个样本20个时刻
        l_in_x = tf.reshape(self.xs, [-1, self.input_size], name='2_2D')  # -1 表示任意行数
        # Ws (in_size, cell_size)
        Ws_in = self._weight_variable([self.input_size, self.cell_size])
        # bs (cell_size, )
        bs_in = self._bias_variable([self.cell_size, ])
        # l_in_y = (batch * n_steps, cell_size)
        with tf.name_scope('Wx_plus_b'):
            l_in_y = tf.matmul(l_in_x, Ws_in) + bs_in
        # reshape l_in_y ==> (batch, n_steps, cell_size)
        self.l_in_y = tf.reshape(l_in_y, [-1, self.n_steps, self.cell_size], name='2_3D')

    # 多时刻的状态叠加层
    def add_cell(self):
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.cell_size, forget_bias=1.0, state_is_tuple=True)
        with tf.name_scope('initial_state'):
            self.cell_init_state = lstm_cell.zero_state(self.batch_size, dtype=tf.float32)
        # time_major=False 表示时间主线不是第一列batch
        self.cell_outputs, self.cell_final_state = tf.nn.dynamic_rnn(
            lstm_cell, self.l_in_y, initial_state=self.cell_init_state, time_major=False)

    # 增加一个输出层
    def add_output_layer(self):
        # shape = (batch * steps, cell_size)
        l_out_x = tf.reshape(self.cell_outputs, [-1, self.cell_size], name='2_2D')
        Ws_out = self._weight_variable([self.cell_size, self.output_size])
        bs_out = self._bias_variable([self.output_size, ])
        # shape = (batch * steps, output_size)
        with tf.name_scope('Wx_plus_b'):
            self.pred = tf.matmul(l_out_x, Ws_out) + bs_out  # 预测结果

    def compute_cost(self):
        losses = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
            logits = [tf.reshape(self.pred, [-1], name='reshape_pred')],
            targets = [tf.reshape(self.ys, [-1], name='reshape_target')],
            weights = [tf.ones([self.batch_size * self.n_steps], dtype=tf.float32)],
            average_across_timesteps=True,
            softmax_loss_function=self.ms_error,
            name='losses'
        )
        with tf.name_scope('average_cost'):
            self.cost = tf.div(
                tf.reduce_sum(losses, name='losses_sum'),
                self.batch_size,
                name='average_cost')
            tf.summary.scalar('cost', self.cost)

    def ms_error(self, labels=None, logits=None):
        return tf.square(tf.subtract(labels, logits))

    def _weight_variable(self, shape, name='weights'):
        initializer = tf.random_normal_initializer(mean=0., stddev=1., )
        return tf.get_variable(shape=shape, initializer=initializer, name=name)

    def _bias_variable(self, shape, name='biases'):
        initializer = tf.constant_initializer(0.1)
        return tf.get_variable(name=name, shape=shape, initializer=initializer)


if __name__ == '__main__':
    is_train = 1
    # seq, res = get_batch_boston()
    model = LSTMRNN(TIME_STEPS, INPUT_SIZE, OUTPUT_SIZE, CELL_SIZE, BATCH_SIZE)
    sess = tf.Session()
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("logs", sess.graph)
    # tf.initialize_all_variables() no long valid from
    # 2017-03-02 if using tensorflow >= 0.12
    saver = tf.train.Saver()
    tf.add_to_collection('pred_network', model.pred)
    sess.run(tf.global_variables_initializer())
    # relocate to the local dir and run this line to view it on Chrome (http://0.0.0.0:6006/):
    # $ tensorboard --logdir='logs'
    if is_train == 1:
        for j in range(1):  # 训练200次
            pred_res = None
            for i in range(12460):  # 把整个数据分为1246个时间段
                seq, res = get_batch_boston()

                if i == 0:
                    feed_dict = {
                        model.xs: seq,
                        model.ys: res,
                        # create initial state
                    }
                else:
                    feed_dict = {
                        model.xs: seq,
                        model.ys: res,
                        model.cell_init_state: state  # use last state as the initial state for this run
                    }

                _, cost, state, pred = sess.run(
                    [model.train_op, model.cost, model.cell_final_state, model.pred],
                    feed_dict=feed_dict)
                pred_res = pred

                result = sess.run(merged, feed_dict)
                writer.add_summary(result, i)
            print('{0} cost: '.format(j), round(cost, 4))
            BATCH_START = 0  # 从头再来一遍

            saver.save(sess, "predict_days/my-model", global_step=j)
    else:
        with tf.Session() as sess:
            new_saver = tf.train.import_meta_graph('./predict_days/my-model-644.meta')
            new_saver.restore(sess, './predict_days/my-model-644')
            # tf.get_collection() 返回一个list. 但是这里只要第一个参数即可
            y = tf.get_collection('pred_network')[0]

            graph = tf.get_default_graph()
            # print(graph)

            # 因为y中有placeholder，所以sess.run(y)的时候还需要用实际待预测的样本以及相应的参数来填充这些placeholder，而这些需要通过graph的get_operation_by_name方法来获取。
            input_x = graph.get_operation_by_name('inputs/xs').outputs[0]

            seq, res = get_batch_boston_test()
            print(seq.shape)
            # 使用y进行预测
            pred = sess.run(y, feed_dict={input_x: seq})
            print(pred.shape)
            pred_res = pred

    # 画图
    print("结果:", pred_res.shape)


    #训练时的预测情况图：
    # 与最后一次训练所用的数据保持一致
    train_y = train_y[12450:13310]
    pred_res_true = ss_y.inverse_transform(pred_res)
    train_y_true = ss_y.inverse_transform(train_y)
    print(mean_absolute_error(train_y_true,pred_res_true))
    print('实际', train_y.flatten().shape)



    # # 测试时候的预测情况图：
    # pred_res_true = ss_y.inverse_transform(pred_res)
    # train_y_true = ss_y.inverse_transform(train_y_test)
    # print(mean_absolute_error(train_y_true,pred_res_true))
    # print('实际', train_y_true.flatten().shape)




    r_size = BATCH_SIZE * TIME_STEPS
    ###画图###########################################################################
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(20, 3))  # dpi参数指定绘图对象的分辨率，即每英寸多少个像素，缺省值为80
    axes = fig.add_subplot(1, 1, 1)
    # 为了方便看，只显示了后100行数据
    line1, = axes.plot(range(100), pred.flatten()[-100:], 'b--', label='rnn计算结果')
    # line2,=axes.plot(range(len(gbr_pridict)), gbr_pridict, 'r--',label='优选参数')
    line3, = axes.plot(range(100), train_y.flatten()[- 100:], 'r', label='实际')

    axes.grid()
    fig.tight_layout()
    # plt.legend(handles=[line1, line2,line3])
    plt.legend(handles=[line1, line3])
    plt.title('递归神经网络')
    plt.show()