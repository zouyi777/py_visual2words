import numpy as np
from sklearn.model_selection import train_test_split
from py_visual2words.net import img2tag_netmodel
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import Callback
from py_visual2words.utils import pre_data_util

# 准备数据集
image_list, tag_list = pre_data_util.readImageAndLabel()

# 将图片数据从整数转换为浮点数
data = np.array(image_list) / 255.0
# 生成标签向量
labels = pre_data_util.gen_label_vec(tag_list)

print(data.shape)
print(labels.shape)

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.01)

# 构造网络
model = img2tag_netmodel.CBAMModel(len(labels[0]))
model.summary()
# 添加优化器、损失函数和评估指标
# Adam()函数的参数：keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.99, epsilon=1e-08, decay=0.0, amsgrad=False)
# lr：float> = 0.学习率
# beta_1：float，0 <beta <1。一般接近1。一阶矩估计的指数衰减率
# beta_2：float，0 <beta <1。一般接近1。二阶矩估计的指数衰减率
# epsilon：float> = 0,模糊因子。如果None，默认为K.epsilon()。该参数是非常小的数，其为了防止在实现中除以零
# decay：float> = 0,每次更新时学习率下降
# amsgrad: 布尔型，是否使用AMSGrad变体
optimizer = keras.optimizers.Adam(0.005)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
# 训练与验证
# 自定义回调类，当loss小于某个值时，学习率设为0，不再更新,避免越过最优值
class LossHistory(Callback):  # 继承自Callback类
    def on_epoch_end(self, batch, logs={}):
        print("\r")
        lr = K.get_value(model.optimizer.lr)
        loss = logs.get('loss')
        if loss < 1e-03:
            K.set_value(model.optimizer.lr, 0)
        print("     lr={}     loss={}".format(lr, loss))
loss_history = LossHistory()
model.fit(x_train, y_train, batch_size=pre_data_util.batch_size, epochs=pre_data_util.epochs,
                            shuffle=True, callbacks=[loss_history])  # shuffle 表示随机打乱输入样本
model.evaluate(x_test, y_test)
# 保存模型
model.save_weights(pre_data_util.model_path)

