import tensorflow as tf


# 批量读取数据
def read_data(file_queue):
    '''
    the function is to get features and label (即样本特征和样本的标签）
    数据来源是csv的文件，采用tensorflow 自带的对csv文件的处理方式
    :param file_queue:
    :return: features,label
    '''
    # 读取的时候需要跳过第一行
    reader = tf.TextLineReader(skip_header_lines=1)
    key, value = reader.read(file_queue)
    # 对于数据源中空的值设置默认值
    record_defaults = [[''], [''], [''], [0.], [0.], [0.], [0.], [0.], [''], [0.], [0.]]
    # 定义decoder，每次读取的执行都从文件中读取一行。然后，decode_csv 操作将结果解析为张量列表
    province, city, address,  longitude, latitude, price, buildingTypeId, tradeTypeId, listingDate,  daysOnMarket, bedrooms = tf.decode_csv(value, record_defaults)

    features = tf.stack([latitude,longitude,price,buildingTypeId])
    return features, daysOnMarket


def create_pipeline(filename,batch_size,num_epochs=None):
    '''
    the function is to get every batch example and label
    此处使用的是tf.train.batch，即顺序获取，非随机获取，随机获取采用的方法是：tf.train.shuffle_batch
    :param filename:
    :param batch_size:
    :param num_epochs:
    :return:example_batch,label_batch
    '''
    file_queue = tf.train.string_input_producer([filename],num_epochs=num_epochs)
    # example,label 样本和样本标签,batch_size 返回一个样本batch样本集的样本个数
    example,dayOnMarket = read_data(file_queue)
    # 出队后队列至少剩下的数据个数，小于capacity（队列的长度）否则会报错，
    min_after_dequeue = 1000
    #队列的长度
    capacity = min_after_dequeue+batch_size
    # 顺序获取每一批数据
    example_batch,daysOnMarket_batch= tf.train.batch([example,dayOnMarket],batch_size=batch_size,capacity=capacity)#顺序读取
    return example_batch,daysOnMarket_batch


diminput  = 2
#隐藏层128个神经元
dimhidden = 128
dimoutput = 1
#把输入拆分成多少个序列
nsteps    = 2

weights = {
    'hidden': tf.Variable(tf.random_normal([diminput, dimhidden])),
    'out': tf.Variable(tf.random_normal([dimhidden, dimoutput]))
}
biases = {
    'hidden': tf.Variable(tf.random_normal([dimhidden])),
    'out': tf.Variable(tf.random_normal([dimoutput]))
}

#定义RNN网络模型
def _RNN(_X, _W, _b, _nsteps, _name):
    #   => [nsteps, batchsize, diminput]
    _X = tf.transpose(_X, [1, 0, 2])
    # 2. Reshape input to [nsteps*batchsize, diminput]
    _X = tf.reshape(_X, [-1, diminput])
    # 3. Input layer => Hidden layer
    _H = tf.matmul(_X, _W['hidden']) + _b['hidden']
    # 把输入的数据切片
    _Hsplit = tf.split(_H, _nsteps, 0)

    with tf.variable_scope(_name) as scope:
        # scope.reuse_variables()
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(dimhidden, forget_bias=1.0)
        _LSTM_O, _LSTM_S = tf.contrib.rnn.static_rnn(lstm_cell, _Hsplit, dtype=tf.float32)
    # 6. Output
    _O = tf.matmul(_LSTM_O[-1], _W['out']) + _b['out']
    # Return!
    return {
        'X': _X, 'H': _H, 'Hsplit': _Hsplit,
        'LSTM_O': _LSTM_O, 'LSTM_S': _LSTM_S, 'O': _O
    }


print("Network ready")


learning_rate = 0.001
x      = tf.placeholder("float", [None, nsteps, diminput])
y      = tf.placeholder("float", [None, dimoutput])
myrnn  = _RNN(x, weights, biases, nsteps, 'basic')
pred   = myrnn['O']

cost   = tf.reduce_mean(tf.square(y-pred))
optm   = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost) # Adam Optimizer
accr   = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pred,1), tf.argmax(y,1)), tf.float32))

print ("Network Ready!")


training_epochs = 50
batch_size      = 16
display_step    = 1

example_batch, daysOnMarket_batch = create_pipeline('month_456_1.csv', batch_size)
test_example_batch, test_daysOnMarket_batch = create_pipeline('month_6_1.csv', batch_size)

# 初始化全局和局部变量
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
saver = tf.train.Saver()

print ("Start optimization")

with tf.Session() as sess:
    sess.run(init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = 1200
        # Loop over all batches
        for i in range(total_batch):

            batch_xs, batch_ys = sess.run([example_batch, daysOnMarket_batch])
            test_batch_xs, test_batch_ys = sess.run([test_example_batch, test_daysOnMarket_batch])
            batch_xs = batch_xs.reshape((batch_size, nsteps, diminput))
            batch_ys = batch_ys.reshape(-1 ,1)

            # Fit training using batch data
            feeds = {x: batch_xs, y: batch_ys}
            sess.run(optm, feed_dict=feeds)
            # Compute average loss
            avg_cost += sess.run(cost, feed_dict=feeds)/total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            print ("Epoch: %03d/%03d cost: %.9f" % (epoch, training_epochs, avg_cost))
            feeds = {x: batch_xs, y: batch_ys}
    save_path = saver.save(sess, "./save/model.ckpt")
    for i in range(10):
        test_batch_xs, test_batch_ys = sess.run([test_example_batch, test_daysOnMarket_batch])
        test_batch_xs = test_batch_xs.reshape((batch_size, nsteps, diminput))
        test_batch_ys = test_batch_ys.reshape(-1, 1)

        print('pred:%s' % sess.run(pred, feed_dict={x: test_batch_xs}))

    coord.request_stop()
    coord.join(threads)
print ("Optimization Finished.")