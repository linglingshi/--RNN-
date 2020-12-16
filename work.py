import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import random
import matplotlib.pyplot as plt
import time
import shutil
from collections import Counter
import os

start_time = time.time()

AverageAccuracyList = []

'''
    准备阶段：首先准备一个数据集，数据集中包含的内容为任意的一段文字。将该文字放到一个txt文档中，与程序主代码放于同一个文件夹下即可
            同时还要新建一个result分别存放不同数据集训练的结果。
'''


def elapsed(sec):
    if sec < 60:
        return str(sec) + " sec"
    elif sec < (60 * 60):
        return str(sec / 60) + " min"
    else:
        return str(sec / (60 * 60)) + " hr"


# 中文多文件
def readalltxt(txt_files):
    labels = []
    for txt_file in txt_files:
        target = get_ch_lable(txt_file)
        labels.append(target)
    return labels


# 中文字 函数从文件里获取文本
def get_ch_lable(txt_file):
    labels = ""
    with open(txt_file, 'rb') as f:
        for label in f:
            labels = labels + label.decode('utf-8')
            # labels = labels + label.decode('gb2312')
    return labels


# 函数将文本数组转换成向量，优先转文件里的字符到向量
def get_ch_lable_v(txt_file, word_num_map, txt_label=None):
    words_size = len(word_num_map)
    to_num = lambda word: word_num_map.get(word, words_size)
    if txt_file != None:
        txt_label = get_ch_lable(txt_file)

    labels_vector = list(map(to_num, txt_label))
    return labels_vector



saveDir = "./result/test3/"



# 训练集文件的名称，位置在同一个目录下
tf.reset_default_graph()
training_file = saveDir+'wordstest.txt'


test_arr = ["学习党章", "深化改革"]



#


##################### 样本预处理 #########################

# 取出样本，存放到training_data中去

training_data = get_ch_lable(training_file)
print("Loaded training data...")

# 获取全部的字表words
counter = Counter(training_data)
words = sorted(counter)
words_size = len(words)
word_num_map = dict(zip(words, range(words_size)))

# 生成样本向量wordlabel和与向量对应关系的word_num_map
print('字表大小:', words_size)
wordlabel = get_ch_lable_v(training_file, word_num_map)

################### 参数和占位符定义 #########################


# 定义参数

learning_rate = 0.001  # 学习率
training_iters = 200000  # 训练迭代次数

display_step = 500  # 每多少次输出一下中间状态

# 每输出4字来预测下一个字
n_input = 4

##### 使用三层网络 LSTM_RNN 每一层的隐层节点数目

n_hidden1 = 256
n_hidden2 = 512
n_hidden3 = 512


filename = saveDir+'testResult.txt'
# ### 保存相关参数信息
with open(saveDir+"parameters.txt", mode="ta", encoding="GBK") as name:
    name.write("迭代相关参数\n")
    name.write("learning_Rate="+str(learning_rate) + "\n")
    name.write("train_iteration="+str(training_iters) + "\n")
    name.write("display_step="+str(display_step)+"\n")
    name.write("\n")
    name.write("三层LSTM隐层节点数目\n")
    name.write("layer1 = "+str(n_hidden1)+"\n")
    name.write("layer2 = "+str(n_hidden2)+"\n")
    name.write("layer3 = "+str(n_hidden3)+"\n")
    name.close()

    # name.close()



# 定义占位符
x = tf.placeholder("float", [None, n_input, 1])
wordy = tf.placeholder("float", [None, words_size])

x1 = tf.reshape(x, [-1, n_input])
x2 = tf.split(x1, n_input, 1)

################### 模型训练与优化 #####################

# 放入3层LSTM网络，最终通过一个全连接生成words_size个节点，为后面的softmax做准备
# 2-layer LSTM，每层有 n_hidden 个units
rnn_cell = rnn.MultiRNNCell([rnn.LSTMCell(n_hidden1), rnn.LSTMCell(n_hidden2), rnn.LSTMCell(n_hidden3)])

# 通过RNN得到输出
outputs, states = rnn.static_rnn(rnn_cell, x2, dtype=tf.float32)

# 通过全连接输出指定维度
pred = tf.contrib.layers.fully_connected(outputs[-1], words_size, activation_fn=None)

# 优化器使用的是AdamOptimizer，loss使用的是softmax的交叉熵，正确率是统计one_hot中索引对应的位置相同的个数
# 定义loss与优化器
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=wordy))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# 模型评估
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(wordy, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

############################## 模型训练环节 ######################


# 训练中，在session中每次随机取一个偏移量，然后取后面4个文字向量当作输入，第5个文字向量当作标签用来计算loss


if os.path.exists("./log/rnnword/"):
    shutil.rmtree("./log/rnnword")
    os.mkdir("./log/rnnword/")

savedir = "log/rnnword/"
saver = tf.train.Saver(max_to_keep=1)  # 生成saver

# 启动session
with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    step = 0
    offset = random.randint(0, n_input + 1)
    end_offset = n_input + 1
    acc_total = 0
    loss_total = 0

    # 添加保存检查点功能
    kpt = tf.train.latest_checkpoint(savedir)
    print("kpt:", kpt)
    startepo = 0
    if kpt != None:
        saver.restore(session, kpt)
        ind = kpt.find("-")
        startepo = int(kpt[ind + 1:])
        print(startepo)
        step = startepo

    while step < training_iters:

        # 随机取一个位置偏移
        if offset > (len(training_data) - end_offset):
            offset = random.randint(0, n_input + 1)

        inwords = [[wordlabel[i]] for i in range(offset, offset + n_input)]  # 按照指定的位置偏移获得后4个文字向量，当作输入

        inwords = np.reshape(np.array(inwords), [-1, n_input, 1])

        out_onehot = np.zeros([words_size], dtype=float)
        out_onehot[wordlabel[offset + n_input]] = 1.0
        out_onehot = np.reshape(out_onehot, [1, -1])  # 所有的字都变成onehot

        _, acc, lossval, onehot_pred = session.run([optimizer, accuracy, loss, pred],
                                                   feed_dict={x: inwords, wordy: out_onehot})
        loss_total += lossval
        acc_total += acc
        if (step + 1) % display_step == 0:
            print("Iter= " + str(step + 1) + ", Average Loss= " + \
                  "{:.6f}".format(loss_total / display_step) + ", AverageAccuracy= " + \
                  "{:.2f}%".format(100 * acc_total / display_step))
            # 将准确率存放到数组中去，用于绘制图形
            AverageAccuracyList.append(100 * acc_total / display_step)

            save_accuracy_name= saveDir + "averageAccuracy.txt"



            with open(save_accuracy_name,mode='ta',encoding="GBK") as names:
                names.write("iters:"+str(step+1)+",accuracy:"+str(100 * acc_total / display_step)+"%\n")
                names.close()



            acc_total = 0
            loss_total = 0
            in2 = [words[wordlabel[i]] for i in range(offset, offset + n_input)]
            out2 = words[wordlabel[offset + n_input]]
            out_pred = words[int(tf.argmax(onehot_pred, 1).eval())]
            print("%s - [%s] vs [%s]" % (in2, out2, out_pred))
            saver.save(session, savedir + "rnnwordtest.cpkt", global_step=step)
        step += 1
        offset += (n_input + 1)  # 调整下一次迭代使用的偏移量

    print("Finished!")
    saver.save(session, savedir + "rnnwordtest.cpkt", global_step=step)
    print("Elapsed time: ", elapsed(time.time() - start_time))

    ################ 对训练的准确率绘制的折线图 ########################
    # 从图中可以看到随着训练轮数的增加，训练的准确率在逐步提升。





    a = range(len(AverageAccuracyList))

    plt.plot(a, AverageAccuracyList, label='AverageAccuracy', linewidth=3, color='r', marker='o',
             markerfacecolor='blue', markersize=3)
    plt.xlabel('Plot Number')
    plt.ylabel('Important var')
    plt.title('Interesting Graph\nCheck it out')
    plt.legend()
    # plt.show()
    plt.savefig(saveDir + "result.png")

    # 对数据进行测试，输入n_input个字，收到输入的文本后，通过eval计算onehot_pred节点，
    # 并进行文字的转义，得到预测文字接下来将预测文字再循环输入模型中，预测下一个文字

    # shutil.copy("./wordstest.txt",saveDir)




    # 循环测试的次数
    count = 2

    while count > 0:
        prompt = "请输入%s个字: " % n_input
        # sentence = input(prompt)
        sentence = test_arr[len(test_arr) - count]
        inputword = sentence.strip()

        if len(inputword) != n_input:
            print("您输入的字符长度为：", len(inputword), "请输入4个字")
            continue
        try:
            inputword = get_ch_lable_v(None, word_num_map, inputword)
            for i in range(32):
                keys = np.reshape(np.array(inputword), [-1, n_input, 1])

                onehot_pred = session.run(pred, feed_dict={x: keys})
                onehot_pred_index = int(tf.argmax(onehot_pred, 1).eval())
                sentence = "%s%s" % (sentence, words[onehot_pred_index])
                inputword = inputword[1:]
                inputword.append(onehot_pred_index)
            # print(sentence)


            with open(filename, mode="ta", encoding="GBK") as name:
                name.write(test_arr[len(test_arr) - count] + "\n")
                name.write(sentence + "\n")
                name.close()
            count = count - 1
        except:
            print("该字我还没学会")
            break
