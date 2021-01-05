import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
from TensorFlowStudy_CatAndDog.net_model import net_model
from PIL import Image
from matplotlib.font_manager import FontProperties # 解决中文乱码的问题
font_set = FontProperties(fname=r"C:\Windows\Fonts\simsun.ttc", size=12)

my_img = np.array(Image.open('data/data_test_images/test (16).jpg'))
# my_img = np.array(Image.open('data/data_train_images/cat (1).jpg'))

class_names = ['猫', '狗']
xticks = [0, 1]
yticks = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
plt.figure("猫狗图像识别", figsize=(6, 3))  # figsize表示figure的大小为宽、长（单位为inch）
plt.subplot(1, 2, 1)  # 表示整个figure分成1行2列，共2个子图，这里子图在第一行第一列
plt.title("要识别的图像", fontproperties=font_set)
plt.imshow(my_img)

channel_axis = 1 if K.image_data_format() == "channels_first" else 3

test_img = my_img / 255.0
# 加载件训练好的保存下来的模型
model = net_model.CBAMModel()
model.load_weights('model/visual2words.h5')
# 预测
test_images = np.array([test_img])  # 将图片装进一个数组里面
predictions = model.predict(test_images)
argmax = np.argmax(predictions[0])
print("predictions[0] = ",predictions[0])
print("argmax= ",argmax)
print(class_names[argmax])

# 对识别结果绘制条形图
def plot_value_array(predictions_array):
    plt.grid(False)
    plt.xticks(xticks,class_names,rotation=45,fontproperties=font_set)
    plt.yticks(yticks)
    thisplot = plt.bar(range(2), predictions_array, color="#33ccdd")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

# 表示整个figure分成1行2列，共2个子图，这里子图在第一行第二列
plt.subplot(1, 2, 2)
plot_value_array(predictions[0])
plt.title("识别结果:"+ class_names[argmax], fontproperties=font_set)
plt.show()