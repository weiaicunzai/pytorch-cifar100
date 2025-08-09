import sys

import requests
from PIL import Image
import numpy as np
from io import BytesIO
import io

# 设置标准输出编码为UTF-8,防止中文乱码
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


def getImageArrayByURL(url):
    # 发送 HTTP 请求获取图片
    response = requests.get(url)

    # 检查请求是否成功
    if response.status_code == 200:
        # 打开图片
        image = Image.open(BytesIO(response.content))

        # 将图片转换为数组
        image_array = np.array(image)

        # 输出数组的形状，且该输出会作为java调用该python程序的返回结果！！！！！！！
        print(f"图片形状为：{image_array.shape}")
    else:
        print("无法获取图片，状态码:", response.status_code)
    # print address_image(image_array)
    # return 返回的值不会传回给java程序的！！，print的内容才会有java程序直接读取出（相当于返回）。
    print(["这是老虎", "相似度90%"])
    # return ["这是老虎", "相似度90%"]


def address_image(image_array):
    pass


if __name__ == "__main__":
    # 图片的 URL
    # urlStr = "https://sky-take-out-wb333.oss-cn-beijing.aliyuncs.com/414f094a-5489-4cb7-be71-f9f5731f48f1.jpeg"
    # getImageArrayByURL(urlStr)
    print("The name of this python script is：", sys.argv[0])
    # while True:
    #     print("你好")
    # 调用第一个参数
    getImageArrayByURL(sys.argv[1])
