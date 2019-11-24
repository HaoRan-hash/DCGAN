import glob
import numpy as np
import imageio
import tensorflow as tf
from network import *
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]='2'  # 忽略AVX2的提示信息


def train():
    # 获取训练数据
    data = []
    for image in glob.glob("./faces/*"):     # 这里是你的图片存放地址，根据需要进行更改
        image_data = imageio.imread(image)    # 读取图片数据
        data.append(image_data)
    input_data = np.array(data)

    # 将数据标准化成 [-1, 1]的取值
    input_data = (input_data.astype(np.float32) - 127.5) / 127.5

    # 构造生成器和判别器
    g = generator_model()
    d = discriminator_model()

    # 构造生成器和判别器组成的DCGAN网络模型
    d_on_g = generator_containing_discriminator(g, d)

    # 优化器用 Adam Optimizer
    g_optimizer = tf.keras.optimizers.Adam(lr=LEARNING_RATE, beta_1=BETA_1)
    d_optimizer = tf.keras.optimizers.Adam(lr=LEARNING_RATE, beta_1=BETA_1)

    # 配置生成器和判别器
    g.compile(loss="binary_crossentropy", optimizer=g_optimizer)
    d_on_g.compile(loss="binary_crossentropy", optimizer=g_optimizer)
    d.trainable = True
    d.compile(loss="binary_crossentropy", optimizer=d_optimizer)

    # 开始训练
    for epoch in range(EPOCHS):
        for index in range(int(input_data.shape[0] / BATCH_SIZE)):
            input_batch = input_data[index * BATCH_SIZE : (index + 1) * BATCH_SIZE]

            # 服从标准正态分布的随机数据（噪声）
            random_data = np.random.normal(size=(BATCH_SIZE, 100))

            # 用随机数据作为输入, 生成器生成图片数据
            generated_images = g.predict(random_data, verbose=0)

            # 真假标签都做平滑处理
            real_label = np.random.uniform(0.9, 1.0, size=(BATCH_SIZE, 1))
            fake_label = np.random.uniform(0, 0.1, size=(BATCH_SIZE, 1))

            # 训练判别器，让他具备识别不合格图片的能力
            d_loss_real = d.train_on_batch(input_batch, real_label)
            d_loss_fake = d.train_on_batch(generated_images, fake_label)
            d_loss = d_loss_real + d_loss_fake

            # 设置判别器不可被训练，因为后面要训练生成器
            d.trainable = False

            # 重新生成随机数据（噪声），很关键。
            random_data = np.random.normal(size=(BATCH_SIZE, 100))
            real_label = np.random.uniform(0.9, 1.0, size=(BATCH_SIZE, 1))

            # 训练生成器，并通过不可被训练的判别器去判别
            g_loss = d_on_g.train_on_batch(random_data, real_label)

            # 恢复判别器可被训练
            d.trainable = True

            # 打印损失
            print("Epoch {}, 第 {} 步, 生成器的损失: {:.3f}, 判别器的损失: {:.3f}".format(epoch, index, g_loss, d_loss))

        # 保存生成器的参数
        g.save_weights("./generator_weight", True)


if __name__ == "__main__":
    train()
