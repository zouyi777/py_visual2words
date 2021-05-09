from PIL import Image
import glob, os
from tensorflow import keras
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np

batch_size = 64
epochs = 50
model_path = 'model/visual2words.h5'
img_root_path ='D://development//python//workspace//SingleLayerPerceptron-matlab//py_visual2words//data//'
def readImageAndLabel():
    image_list = []
    tag_list = []
    for file in glob.glob(img_root_path+'img2tag_images//*.jpg'):
        filepath, filename = os.path.split(file)  # 文件路径和文件名

        filterame, exts = os.path.splitext(filename)  # 文件名和后缀

        if '(' in filterame:
            filterame, _ = filterame.split('(')  # 去掉括号和数字
        if '、' in filterame:
            filterame = filterame.split('、')  # 根据 '、'切割成多个tag
        if not isinstance(filterame, list):
            filterame = [filterame]
        tag_list.append(filterame)

        im = Image.open(file)
        im_resize = im.resize((100, 100))  # 统一缩放成100*100的尺寸
        image_list.append(np.array(im_resize))
        if(np.array(im_resize).shape==(100,100,4)):
            print(filename)
    return image_list, tag_list


def gen_tag_dict(sensentence):
    tok = keras.preprocessing.text.Tokenizer()
    tok.fit_on_texts(sensentence)
    dict = tok.word_index
    dict_reverse = tok.index_word
    char_len = len(dict)
    return dict, dict_reverse, char_len

def gen_label_vec(sentence_list):
    dict, dict_reverse, char_len = gen_tag_dict(sentence_list)
    classes = list(dict.keys())
    mlb = MultiLabelBinarizer(classes)
    labels = mlb.fit_transform(sentence_list)
    return labels

if __name__ == '__main__':
    image_list, tag_list = readImageAndLabel()
    labels = gen_label_vec(tag_list)
    # print(labels)


