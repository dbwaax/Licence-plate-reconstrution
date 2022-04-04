# -- coding:utf-8 --
import os
import cv2
import numpy as np
from PIL import Image, ImageFont, ImageDraw
# imread
def cv_imread(path):
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8),-1)
    return img 
# imwrite
def cv_imwrite(path,img):
    cv2.imencode('.jpg', img)[1].tofile(path)
class Draw:
    _font = [
        ImageFont.truetype(os.path.join(os.path.dirname(__file__), "res/eng_92.ttf"), 126),
        ImageFont.truetype(os.path.join(os.path.dirname(__file__), "res/zh_cn_92.ttf"), 95)
    ]
    _bg = cv2.resize(cv2.cvtColor(cv_imread(os.path.join(os.path.dirname(__file__),"res/blue_bg.png")),cv2.COLOR_RGBA2BGR), (440, 140))
    # _bg = cv2.resize(cv2.cvtColor(cv_imread(os.path.join(os.path.dirname(__file__))), "res/blue_bg.png"),cv2.COLOR_RGBA2BGR), (440, 140))).squeeze()
    # np.squeeze
    def __call__(self, plate):
        if len(plate) != 7:
            print("ERROR: Invalid length")
            return None
        fg = self._draw_fg(plate)
        return cv2.resize(cv2.cvtColor(cv2.bitwise_or(fg, self._bg), cv2.COLOR_BGR2RGB),(94,24))

    def _draw_char(self, ch):
        img = Image.new("RGB", (45 if ch.isupper() or ch.isdigit() else 95, 140), (0, 0, 0))
        draw = ImageDraw.Draw(img)
        draw.text(
            (0, -11 if ch.isupper() or ch.isdigit() else 3), ch,
            fill = (255, 255, 255),
            font = self._font[0 if ch.isupper() or ch.isdigit() else 1]
        )
        if img.width > 45:
            img = img.resize((45, 140))
        return np.array(img)

    def _draw_fg(self, plate):
        img = np.array(Image.new("RGB", (440, 140), (0, 0, 0)))
        offset = 15
        img[0:140, offset:offset+45] = self._draw_char(plate[0])
        offset = offset + 45 + 12
        img[0:140, offset:offset+45] = self._draw_char(plate[1])
        offset = offset + 45 + 34
        for i in range(2, len(plate)):
            img[0:140, offset:offset+45] = self._draw_char(plate[i])
            offset = offset + 45 + 12
        return img


if __name__ == "__main__":
    import argparse
    import matplotlib.pyplot as plt
    import os 
    from tqdm import tqdm
    import shutil
    from green_plate import Draw_green
    parser = argparse.ArgumentParser(description="Generate a blue plate.")
    parser.add_argument("plate", help="license plate number (default: 京A12345)", type=str, nargs="?", default="京A12345")
    args = parser.parse_args()
    raw_path = r'E:\Documents\自我进修\研究生\2020.6.28 车牌识别项目\License_Plate_Detection_Pytorch-master\SRRN_combine\test'
    save_path_label = r'E:\Documents\自我进修\研究生\2020.6.28 车牌识别项目\License_Plate_Detection_Pytorch-master\SRRN_combine\label'
    save_path_input = r'E:\Documents\自我进修\研究生\2020.6.28 车牌识别项目\License_Plate_Detection_Pytorch-master\SRRN_res18\transfer\train'
    filelist = os.listdir(raw_path) 
    filelist = [i.split('.')[0] for i in filelist]
    with open(r'C:\Users\A\Desktop\1.txt', 'r') as f:
        lines = f.readlines()
    filelist = [j.strip() for j in lines]
    draw = Draw()
    draw_g = Draw_green()
    # plate = draw_g('陕AD20806')
    # cv_imwrite('./陕AD20806.jpg',plate)
    count = 0
    for i in tqdm(filelist):
        if(len(i)==7):
            plate = draw(i)
        if(len(i)==8):
            plate = draw_g(i)
        cv_imwrite(save_path_label+'/'+i+'.jpg',plate)
        #shutil.copy(raw_path + '/' + i + '.jpg',save_path_input+'/'+str(count)+'.jpg')
        count+=1
        # draw.save('')
