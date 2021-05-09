import numpy as np
from py_visual2words.net import img2tag_netmodel
from PIL import Image
from py_visual2words.utils import pre_data_util

# 准备数据集
image_list, tag_list = pre_data_util.readImageAndLabel()
dict, dict_reverse, char_len = pre_data_util.gen_tag_dict(tag_list)
class_names =list(dict.keys())

im = Image.open('data/img2tag_images_test/下载 (1).jpg')
im_resize = im.resize((100, 100))  # 统一缩放成100*100的尺寸
test_img = np.array(im_resize) / 255.0

# 加载件训练好的保存下来的模型
model = img2tag_netmodel.CBAMModel(char_len)
model.load_weights(pre_data_util.model_path)

# 预测
test_images = np.array([test_img])  # 将图片装进一个数组里面
predictions = model.predict(test_images)[0]
pre_argsort = np.argsort(predictions)
pre_out = pre_argsort[-3:]

pre_tag = []
for pre in pre_out:
    pre_tag.append(dict_reverse[pre+1])
pre_tag.reverse() # 倒序，因为最后一个是概率最大的，放到最前面来
print(pre_tag)

