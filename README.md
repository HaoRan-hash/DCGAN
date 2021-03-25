# 项目简介
该项目是利用DCGAN生成图片，我使用了3个图集进行训练，分别是鲜花、celeba人脸、动漫头像。 

项目中的每个py文件功能如下：network.py是实现DCGAN的神经网络结构(不同分辨率的放到了extra文件夹里)，train.py是训练过程，generate.py是利用训练好的模型参数生成图片，NRDS.py是使用NRDS指标来对比不同的GAN模型好坏，handle_image.py是修改图片尺寸的(有时候需要对数据集预处理一下)。 

每一个py文件中都有详细的注释说明。

最后的结果我个人感觉还不错，如果大佬们有更好的优化方案欢迎Pull requests或者在Issues里面讨论。

# 资源获取
鲜花图集我已经上传在我的百度网盘上，下载地址：[https://pan.baidu.com/s/1uvU61kB-xPowufwVU8Vzqg](https://pan.baidu.com/s/1uvU61kB-xPowufwVU8Vzqg)，提取码：een8  

celeba人脸图集是由香港中文大学开源的，可以在官网进行下载，官网地址：[http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)  

动漫头像图集也已经上传在我的百度网盘上，下载地址：[https://pan.baidu.com/s/1FoWdyGqYZI80YMmu8SHGMA](https://pan.baidu.com/s/1FoWdyGqYZI80YMmu8SHGMA)，提取码：yi17  

使用的服务器资源获取地址：[https://www.deepbrainchain.org/ailanding.html](https://www.deepbrainchain.org/ailanding.html)

我训练好的一些模型参数文件，上传到了百度网盘。(因为文件比较大，国内用户直接在Github上下载会比较慢，所以我放到了网盘上)  
[https://pan.baidu.com/s/13J34FhWa_9BQMhK2SxfXmw](https://pan.baidu.com/s/13J34FhWa_9BQMhK2SxfXmw)，提取码：fdai  
因为我跑了几十次(调参做实验)，不可能每个模型参数文件都保存下来，所以网盘只放了几个，建议还是用我提供的代码自己训练。

# 使用方法
先运行train.py，再运行generate.py，想做不同GAN对比的使用NRDS.py就行。  
使用代码一定要注意，要在里面配置好你的图集地址，还有你加载的模型参数地址等。同时注意你的图集的分辨率，import对应的network.py。  
重要的事情说三遍：注意修改成你自己的地址！注意import对应的network.py！  

# 训练结果
鲜花图集训练500轮的生成结果如下：
![](./images/flowers-500.PNG)

celeba人脸图集训练500轮的生成结果如下：
![](./images/celeba-500.PNG)

动漫头像图集训练500轮的生成结果如下(这个图集的效果一直不太好，因为图集本身质量就不太行)：
![](./images/cartoon-500.PNG)

# 补充说明   
鲜花图集大概是3000张，图片尺寸是64×64；动漫头像图集大概是7万张，图片尺寸是96×96。  

对于celeba人脸图集这里单独说一下，官方给了20万张，但是如果全部拿来使用会导致训练时间过长，所以这里只用了3万张。同时，原尺寸为178×218，这不利于构造神经网络模型，我将它处理为了128×160的尺寸，既保证模型构造起来容易，又保证图片比例不失调。  

因为训练不同尺寸的图集需要对神经网络模型进行修改（也就是修改network.py)，为了防止一些人不会修改，这里我将对应不同尺寸图集的network.py也上传了，放在extra文件夹下面。首页放的network.py是针对96×96尺寸的，可以用来训练动漫头像图集。(如果你理解了DCGAN的结构，可以自己做任意尺寸的network.py)  

如果想要较好的生成效果，原始图集质量需要比较好(做过深度学习的应该都懂。。。)。  

# 学习建议
我建议大家阅读DCGAN原文的同时，可以在youtube上观看台湾大学李宏毅教授的GAN讲解，配合学习效率更高。  
一些GAN的训练优化技巧可以参考[https://github.com/soumith/ganhacks](https://github.com/soumith/ganhacks)  
推荐一个GAN的知乎专栏[https://zhuanlan.zhihu.com/c_155535936](https://zhuanlan.zhihu.com/c_155535936)
