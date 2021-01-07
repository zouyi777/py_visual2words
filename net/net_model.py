from tensorflow.keras import models
import tensorflow.keras.layers as kl
import tensorflow.keras.backend as k

# ----CBAM 卷积注意力网络结构
# 网上原理讲解文章：https://zhuanlan.zhihu.com/p/101590167

# CAM(Channel Attention Module) 通道注意力模型:压缩空间维度
def channel_attention(input_xs, reduction_ratio):  # input_xs (None,50,50,64)
    # 判断输入数据格式，是channels_first还是channels_last
    channel_axis = 1 if k.image_data_format() == "channels_first" else 3
    # get channel 彩色图片为 3
    channel = int(input_xs.shape[channel_axis])  # 64

    maxpool_channel = kl.GlobalMaxPooling2D()(input_xs)  # (None,64)
    maxpool_channel = kl.Reshape((1, 1, channel))(maxpool_channel)  # ( None,1,1,64)

    avgpool_channel = kl.GlobalAvgPool2D()(input_xs)  # (None,64)
    avgpool_channel = kl.Reshape((1, 1, channel))(avgpool_channel)  # (None,1,1,64)

    dense_one = kl.Dense(units=int(channel * reduction_ratio), activation='relu', kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')
    dense_two = kl.Dense(units=int(channel), activation='relu', kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')
    # max path
    mlp_1_max = dense_one(maxpool_channel)  # (None,1,1,32)
    mlp_2_max = dense_two(mlp_1_max)  # (None,1,1,64)
    mlp_2_max = kl.Reshape(target_shape=(1, 1, int(channel)))(mlp_2_max)  # (None,1,1,64)
    # avg path
    mlp_1_avg = dense_one(avgpool_channel)  # (None,1,1,32)
    mlp_2_avg = dense_two(mlp_1_avg)  # (None,1,1,64)
    mlp_2_avg = kl.Reshape(target_shape=(1, 1, int(channel)))(mlp_2_avg)  # (None,1,1,64)

    channel_attention_feature = kl.Add()([mlp_2_max, mlp_2_avg])  # (None,1,1,64)
    channel_attention_feature = kl.Activation('sigmoid')(channel_attention_feature)  # (None,1,1,64)

    multiply_channel_input = kl.Multiply()([channel_attention_feature, input_xs])  # (None,50,50,64)

    return multiply_channel_input

# SAM(Spatial Attention Module) 空间注意力模型：压缩通道维度
def spatial_attention(channel_refined_feature): # channel_refined_feature(None,50,50,64)

    maxpool_spatial = kl.Lambda(lambda x: k.max(x, axis=3, keepdims=True))(channel_refined_feature)  # (None,50,50,1)

    avgpool_spatial = kl.Lambda(lambda x: k.mean(x, axis=3, keepdims=True))(channel_refined_feature)  # (None,50,50,1)

    max_avg_pool_spatial = kl.Concatenate(axis=3)([maxpool_spatial, avgpool_spatial])  # (None,50,50,2)

    spatial_attention_feature = kl.Conv2D(filters=1, kernel_size=(3, 3), padding="same", activation='sigmoid', kernel_initializer='he_normal',
                                          use_bias=False)(max_avg_pool_spatial)  # conv(None,50,50,1)

    multiply_channel_spatial = kl.Multiply()([channel_refined_feature, spatial_attention_feature])  # (None,50,50,64)
    return multiply_channel_spatial

# cbam 注意力模块
def cbam_block(input_xs, reduction_ratio=0.5): # input_xs(None,50,50,64)
    channel_refined_feature = channel_attention(input_xs, reduction_ratio=reduction_ratio)  # (None,50,50,64)
    spatial_attention_feature = spatial_attention(channel_refined_feature)  # (None,50,50,64)
    add_refined_input = kl.Add()([spatial_attention_feature, input_xs])  # (None,50,50,64)
    return add_refined_input

# 构建CBAM网络模型
def CBAMModel():
    img_input = kl.Input(shape=(100, 100, 3))  # 输入层

    # 卷积层
    x = kl.Conv2D(64, (3, 3), padding='same')(img_input)
    x = kl.BatchNormalization(axis=3)(x)
    x = kl.Activation('relu')(x)
    x = kl.MaxPool2D(pool_size=[2, 2], strides=2)(x)

    x = cbam_block(x)  # 调用cbam 注意力模块

    x = kl.MaxPool2D(pool_size=[2, 2], strides=2)(x)
    x = kl.Reshape(target_shape=(25 * 25 * 64,))(x)
    x = kl.Dense(2, activation='softmax', name='fc')(x)

    model = models.Model(img_input, x)
    return model