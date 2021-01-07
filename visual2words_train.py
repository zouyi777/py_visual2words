from PIL import Image # 载入图片 此依赖库需要下载
import numpy as np
from sklearn.model_selection import train_test_split
from py_visual2words.net import net_model

# 准备训练数据集
data = []
# 定义训练图片的标签：0 代表猫 1 代表狗
labels = []
# 读取猫狗图片
for i in range(50):
    cat = np.array(Image.open('data/data_train_images/cat ('+str(i+1)+').jpg'))
    data.append(cat)
    labels.append(0)
    dog = np.array(Image.open('data/data_train_images/dog (' + str(i + 1) + ').jpg'))
    data.append(dog)
    labels.append(1)
dog = np.array(Image.open('data/data_train_images/dog (51).jpg'))
data.append(dog)
labels.append(1)

data = np.array(data)
labels = np.array(labels)
print(data.shape)
print(labels)

# 将样本从整数转换为浮点数
data = data / 255.0
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.1)

# 构造网络
model = net_model.CBAMModel()
model.summary()
# 添加优化器、损失函数和评估指标
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
# 训练与验证
model.fit(x_train, y_train, epochs=50, shuffle=True)  # shuffle 表示随机打乱输入样本
model.evaluate(x_test, y_test)
# 保存模型
model.save_weights("model/visual2words.h5")

