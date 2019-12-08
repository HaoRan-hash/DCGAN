import tensorflow as tf


# 超参数（包括训练的轮数，学习率等，可以根据需要进行更改）
EPOCHS = 500
BATCH_SIZE = 128
LEARNING_RATE = 0.0002
BETA_1 = 0.5

# 根据图集的图片尺寸可以调整网络结构，但是图片不要太大，否则会造成训练时间过长。


# 定义判别器模型
def discriminator_model():
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Conv2D(
        filters=64,    # 64个过滤器，输出的深度是 64
        kernel_size=(5, 5),    # 过滤器在二维的大小是（5 * 5）
        strides=(2, 2),   # 步长为 2
        padding='same',    # same 表示外围补零
        input_shape=(160, 128, 3),   # 输入形状 [160, 128, 3]。3 表示 RGB 三原色
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.02)   # 权值用均值为0，标准差为0.02的正态分布
    ))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.2))

    model.add(tf.keras.layers.Conv2D(
        128,
        (5, 5),
        strides=(2, 2),
        padding="same",
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.02)
    ))
    model.add(tf.keras.layers.BatchNormalization(epsilon=1e-5, momentum=0.9))   # 加入BN层防止模式崩塌，同时加速收敛
    model.add(tf.keras.layers.LeakyReLU(alpha=0.2))

    model.add(tf.keras.layers.Conv2D(
        256,
        (5, 5),
        strides=(2, 2),
        padding="same",
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.02)
    ))
    model.add(tf.keras.layers.BatchNormalization(epsilon=1e-5, momentum=0.9))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.2))

    model.add(tf.keras.layers.Conv2D(
        512,
        (5, 5),
        strides=(2, 2),
        padding="same",
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.02)
    ))
    model.add(tf.keras.layers.BatchNormalization(epsilon=1e-5, momentum=0.9))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.2))

    model.add(tf.keras.layers.Conv2D(
        1024,
        (5, 5),
        strides=(2, 2),
        padding="same",
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.02)
    ))
    model.add(tf.keras.layers.BatchNormalization(epsilon=1e-5, momentum=0.9))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.2))

    model.add(tf.keras.layers.Flatten())  # 扁平化处理
    model.add(tf.keras.layers.Dense(1))  # 1 个神经元的全连接层
    model.add(tf.keras.layers.Activation("sigmoid"))  # sigmoid 激活层

    return model


# 定义生成器模型
def generator_model():
    model = tf.keras.models.Sequential()

    # 输入的维度是 100
    model.add(tf.keras.layers.Dense(1024 * 5 * 4, input_shape=(100, )))
    model.add(tf.keras.layers.BatchNormalization(epsilon=1e-5, momentum=0.9))
    model.add(tf.keras.layers.Activation("relu"))
    model.add(tf.keras.layers.Reshape((5, 4, 1024)))  # 5 x 4 像素

    model.add(tf.keras.layers.Conv2DTranspose(
        512,
        (5, 5),
        strides=(2, 2),
        padding="same",
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.02)
    ))
    model.add(tf.keras.layers.BatchNormalization(epsilon=1e-5, momentum=0.9))
    model.add(tf.keras.layers.Activation("relu"))

    model.add(tf.keras.layers.Conv2DTranspose(
        256,
        (5, 5),
        strides=(2, 2),
        padding="same",
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.02)
    ))
    model.add(tf.keras.layers.BatchNormalization(epsilon=1e-5, momentum=0.9))
    model.add(tf.keras.layers.Activation("relu"))

    model.add(tf.keras.layers.Conv2DTranspose(
        128,
        (5, 5),
        strides=(2, 2),
        padding="same",
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.02)
    ))
    model.add(tf.keras.layers.BatchNormalization(epsilon=1e-5, momentum=0.9))
    model.add(tf.keras.layers.Activation("relu"))

    model.add(tf.keras.layers.Conv2DTranspose(
        64,
        (5, 5),
        strides=(2, 2),
        padding="same",
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.02)
    ))
    model.add(tf.keras.layers.BatchNormalization(epsilon=1e-5, momentum=0.9))
    model.add(tf.keras.layers.Activation("relu"))

    model.add(tf.keras.layers.Conv2DTranspose(
        3,
        (5, 5),
        strides=(2, 2),
        padding="same",
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.02)
    ))
    model.add(tf.keras.layers.Activation("tanh"))   # tanh 激活层

    return model


# 构造一个 DCGAN 对象，包含一个生成器和一个判别器
# 输入 -> 生成器 -> 判别器 -> 输出
def generator_containing_discriminator(generator, discriminator):
    model = tf.keras.models.Sequential()
    model.add(generator)
    discriminator.trainable = False  # 初始时判别器不可被训练
    model.add(discriminator)
    return model
