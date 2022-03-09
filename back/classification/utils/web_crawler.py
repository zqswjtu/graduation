# 找到图片的链接
import os
import time

from bs4 import BeautifulSoup
from urllib import request, parse


# 从得到的图片链接下载图片，并保存
def save_image(link, input_data, count):
    try:
        time.sleep(0.2)
        request.urlretrieve(link, './'+input_data+'/'+str(count)+'.jpg')
    except Exception as e:
        time.sleep(1)
        print(e)
    else:
        print("图+1,已有" + str(count) + "张图")


def find_link(page_num, input_data, word):
    for i in range(page_num):
        print(i)
        try:
            url = "http://cn.bing.com/images/async?q={0}&first={1}&count=35&relp=35&lostate=r&mmasync=1&dgState=x*175_y*848_h*199_c*1_i*106_r*0"
            # 定义请求头
            agent = {
                'User-Agent': "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/31.0.165063 Safari/537.36 AppEngine-Google."}
            page1 = request.Request(url.format(input_data, i * 35 + 1), headers=agent)
            page = request.urlopen(page1)
            # 使用beautifulSoup进行解析网页
            soup = BeautifulSoup(page.read(), 'html.parser')
            # 创建文件夹
            if not os.path.exists("./" + word):
                os.mkdir('./' + word)

            for StepOne in soup.select('.mimg'):
                link = StepOne.attrs['src']
                count = len(os.listdir('./' + word)) + 1
                save_image(link, word, count)
        except Exception as e:
            print(e)


if __name__ == "__main__":
    # 输入需要加载的页数，每页35幅图像
    page_num = 3
    # 输入需要搜索的关键字
    word = "牙刷"
    # UTF-8编码
    input_data = parse.quote(word)
    # print(InputData)
    find_link(page_num, input_data, word)
