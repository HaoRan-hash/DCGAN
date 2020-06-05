import glob
import imageio
import numpy as np
from network96 import *
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# 获取真实数据
data = []
for image in (glob.glob("./cartoon/*"))[0: 1280]:     # 这里是你的图片存放地址，根据需要进行更改
    image_data = imageio.imread(image)    # 读取图片数据
    data.append(image_data)
x1 = np.array(data)
x1 = (x1.astype(np.float32) - 127.5) / 127.5
y1 = np.ones((1280, ))

# 获取 GAN生成的数据
g = generator_model()
g.load_weights("D:/训练结果/cartoon-smooth/generator_weight")
random_data = np.random.normal(size=(1280, 100))
x2 = g.predict(random_data, verbose=1)
y2 = np.zeros((1280, ))

# 制作训练测试数据集
x = np.concatenate((x1, x2), axis=0)
y = np.concatenate((y1, y2), axis=0)
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=666)
x_gan_test = x_test[y_test == 0]


# 配置分类器
d = discriminator_model()
optimizer = tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.5)
d.compile(loss='binary_crossentropy', optimizer=optimizer)

# 定义一些东西用于做图
outputs = []
list_epoch = []

# 训练
epochs = 20
batchsize = 128
for epoch in range(epochs):
    for index in range(x_train.shape[0] // batchsize):
        x_batch = x_train[index * batchsize: (index + 1) * batchsize]
        y_batch = y_train[index * batchsize: (index + 1) * batchsize]

        loss = d.train_on_batch(x_batch, y_batch)
        print(loss)
    y_gan_predict = d.predict(x_gan_test)
    output = np.mean(y_gan_predict)
    outputs.append(output)
    list_epoch.append(epoch)

print(outputs)
plt.plot(list_epoch, outputs)
plt.xlabel('epoch')
plt.ylabel('output')
plt.show()

# 计算面积
area = np.trapz(outputs, list_epoch, dx=0.01)
print(area)
