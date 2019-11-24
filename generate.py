import numpy as np
from PIL import Image
import tensorflow as tf
from network import *
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]='2'  # 忽略AVX2的提示信息


# 用 DCGAN 的生成器模型和训练得到的生成器参数文件来生成图片
def generate():
    # 构造生成器
    g = generator_model()

    # 优化器用 Adam Optimizer
    g_optimizer = tf.keras.optimizers.Adam(lr=LEARNING_RATE, beta_1=BETA_1)

    # 配置生成器
    g.compile(loss="binary_crossentropy", optimizer=g_optimizer)

    # 加载训练好的生成器参数
    g.load_weights("generator_weight")

    # 服从标准正态分布的随机数据（噪声）
    random_data = np.random.normal(size=(BATCH_SIZE, 100))

    # 用随机数据作为输入, 生成器生成图片数据
    images = g.predict(random_data, verbose=1)

    # 用生成的图片数据生成PNG图片
    for i in range(BATCH_SIZE):
        image = images[i] * 127.5 + 127.5
        Image.fromarray(image.astype(np.uint8)).save("./generate/image-%s.png" % i)    # 保存图片的地址，根据需要可以更改


if __name__ == "__main__":
    generate()
