import glob
from PIL import Image


if __name__ == "__main__":
    i = 1
    for image in glob.glob("D:/BaiduNetdiskDownload/CelebA/Img/img_align_celeba/*"):    # 要处理的图片存放的位置
        picture = Image.open(image)

        new_picture = picture.resize((128, 160))   # 图片修改的尺寸，可以根据自己需要进行调整
        new_picture.save("./faces/%s.png" % i)   # 修改过的图片保存的位置
        i = i + 1
