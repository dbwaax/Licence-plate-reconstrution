# -- coding:utf-8 --
import os
import cv2
import numpy as np
from PIL import Image, ImageDraw,ImageFont
# imread
def cv_imread(path):
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8),-1)
    return img 
# imwrite
def cv_imwrite(path,img):
    cv2.imencode('.jpg', img)[1].tofile(path)

def load_font():
    return {
        "京": cv_imread(os.path.join(os.path.dirname(__file__), "res/ne000.png")),
        "津": cv_imread(os.path.join(os.path.dirname(__file__), "res/ne001.png")),
        "冀": cv_imread(os.path.join(os.path.dirname(__file__), "res/ne002.png")),
        "晋": cv_imread(os.path.join(os.path.dirname(__file__), "res/ne003.png")),
        "蒙": cv_imread(os.path.join(os.path.dirname(__file__), "res/ne004.png")),
        "辽": cv_imread(os.path.join(os.path.dirname(__file__), "res/ne005.png")),
        "吉": cv_imread(os.path.join(os.path.dirname(__file__), "res/ne006.png")),
        "黑": cv_imread(os.path.join(os.path.dirname(__file__), "res/ne007.png")),
        "沪": cv_imread(os.path.join(os.path.dirname(__file__), "res/ne008.png")),
        "苏": cv_imread(os.path.join(os.path.dirname(__file__), "res/ne009.png")),
        "浙": cv_imread(os.path.join(os.path.dirname(__file__), "res/ne010.png")),
        "皖": cv_imread(os.path.join(os.path.dirname(__file__), "res/ne011.png")),
        "闽": cv_imread(os.path.join(os.path.dirname(__file__), "res/ne012.png")),
        "赣": cv_imread(os.path.join(os.path.dirname(__file__), "res/ne013.png")),
        "鲁": cv_imread(os.path.join(os.path.dirname(__file__), "res/ne014.png")),
        "豫": cv_imread(os.path.join(os.path.dirname(__file__), "res/ne015.png")),
        "鄂": cv_imread(os.path.join(os.path.dirname(__file__), "res/ne016.png")),
        "湘": cv_imread(os.path.join(os.path.dirname(__file__), "res/ne017.png")),
        "粤": cv_imread(os.path.join(os.path.dirname(__file__), "res/ne018.png")),
        "桂": cv_imread(os.path.join(os.path.dirname(__file__), "res/ne019.png")),
        "琼": cv_imread(os.path.join(os.path.dirname(__file__), "res/ne020.png")),
        "渝": cv_imread(os.path.join(os.path.dirname(__file__), "res/ne021.png")),
        "川": cv_imread(os.path.join(os.path.dirname(__file__), "res/ne022.png")),
        "贵": cv_imread(os.path.join(os.path.dirname(__file__), "res/ne023.png")),
        "云": cv_imread(os.path.join(os.path.dirname(__file__), "res/ne024.png")),
        "藏": cv_imread(os.path.join(os.path.dirname(__file__), "res/ne025.png")),
        "陕": cv_imread(os.path.join(os.path.dirname(__file__), "res/ne026.png")),
        "甘": cv_imread(os.path.join(os.path.dirname(__file__), "res/ne027.png")),
        "青": cv_imread(os.path.join(os.path.dirname(__file__), "res/ne028.png")),
        "宁": cv_imread(os.path.join(os.path.dirname(__file__), "res/ne029.png")),
        "新": cv_imread(os.path.join(os.path.dirname(__file__), "res/ne030.png")),
        "A": cv_imread(os.path.join(os.path.dirname(__file__), "res/ne100.png")),
        "B": cv_imread(os.path.join(os.path.dirname(__file__), "res/ne101.png")),
        "C": cv_imread(os.path.join(os.path.dirname(__file__), "res/ne102.png")),
        "D": cv_imread(os.path.join(os.path.dirname(__file__), "res/ne103.png")),
        "E": cv_imread(os.path.join(os.path.dirname(__file__), "res/ne104.png")),
        "F": cv_imread(os.path.join(os.path.dirname(__file__), "res/ne105.png")),
        "G": cv_imread(os.path.join(os.path.dirname(__file__), "res/ne106.png")),
        "H": cv_imread(os.path.join(os.path.dirname(__file__), "res/ne107.png")),
        "J": cv_imread(os.path.join(os.path.dirname(__file__), "res/ne108.png")),
        "K": cv_imread(os.path.join(os.path.dirname(__file__), "res/ne109.png")),
        "L": cv_imread(os.path.join(os.path.dirname(__file__), "res/ne110.png")),
        "M": cv_imread(os.path.join(os.path.dirname(__file__), "res/ne111.png")),
        "N": cv_imread(os.path.join(os.path.dirname(__file__), "res/ne112.png")),
        "P": cv_imread(os.path.join(os.path.dirname(__file__), "res/ne113.png")),
        "Q": cv_imread(os.path.join(os.path.dirname(__file__), "res/ne114.png")),
        "R": cv_imread(os.path.join(os.path.dirname(__file__), "res/ne115.png")),
        "S": cv_imread(os.path.join(os.path.dirname(__file__), "res/ne116.png")),
        "T": cv_imread(os.path.join(os.path.dirname(__file__), "res/ne117.png")),
        "U": cv_imread(os.path.join(os.path.dirname(__file__), "res/ne118.png")),
        "V": cv_imread(os.path.join(os.path.dirname(__file__), "res/ne119.png")),
        "W": cv_imread(os.path.join(os.path.dirname(__file__), "res/ne120.png")),
        "X": cv_imread(os.path.join(os.path.dirname(__file__), "res/ne121.png")),
        "Y": cv_imread(os.path.join(os.path.dirname(__file__), "res/ne122.png")),
        "Z": cv_imread(os.path.join(os.path.dirname(__file__), "res/ne123.png")),
        "0": cv_imread(os.path.join(os.path.dirname(__file__), "res/ne124.png")),
        "1": cv_imread(os.path.join(os.path.dirname(__file__), "res/ne125.png")),
        "2": cv_imread(os.path.join(os.path.dirname(__file__), "res/ne126.png")),
        "3": cv_imread(os.path.join(os.path.dirname(__file__), "res/ne127.png")),
        "4": cv_imread(os.path.join(os.path.dirname(__file__), "res/ne128.png")),
        "5": cv_imread(os.path.join(os.path.dirname(__file__), "res/ne129.png")),
        "6": cv_imread(os.path.join(os.path.dirname(__file__), "res/ne130.png")),
        "7": cv_imread(os.path.join(os.path.dirname(__file__), "res/ne131.png")),
        "8": cv_imread(os.path.join(os.path.dirname(__file__), "res/ne132.png")),
        "9": cv_imread(os.path.join(os.path.dirname(__file__), "res/ne133.png"))
    }


class Draw_green:
    # _font = load_font()
    _font = [
        ImageFont.truetype(os.path.join(os.path.dirname(__file__), "res/eng_92.ttf"), 126),
        ImageFont.truetype(os.path.join(os.path.dirname(__file__), "res/zh_cn_92.ttf"), 95)
    ]
    # _bg = [
    #     cv2.resize(cv_imread(os.path.join(os.path.dirname(__file__), "res/green_bg_0.png")), (480, 140)),
    #     cv2.resize(cv_imread(os.path.join(os.path.dirname(__file__), "res/green_bg_1.png")), (480, 140))
    # ]
    _bg = [
        cv2.resize(cv2.cvtColor(cv_imread(os.path.join(os.path.dirname(__file__),"res/green_bg_0.png")),cv2.COLOR_RGBA2BGR), (495, 140)),
        # cv2.resize(cv2.cvtColor(cv_imread(os.path.join(os.path.dirname(__file__),"res/green_bg_1.png")),cv2.COLOR_RGBA2BGR), (481, 140))
    ]
    def __call__(self, plate, bg=0):
        if len(plate) != 8:
            print("ERROR: Invalid length")
            return None
        try:
            fg = self._draw_fg(plate)
            return  cv2.resize(cv2.cvtColor(cv2.bitwise_and(fg, self._bg[bg]), cv2.COLOR_BGR2RGB),(94,24))
            # return cv2.resize(cv2.cvtColor(cv2.bitwise_or(fg, self._bg), cv2.COLOR_BGR2RGB),(118,30))
        except KeyError:
            print("ERROR: Invalid character")
            return None
        except IndexError:
            print("ERROR: Invalid background index")
            return None

    # def _draw_char(self, ch):
        # return cv2.resize(self._font[ch], (43 if ch.isupper() or ch.isdigit() else 45, 140))
    # def _draw_char(self, ch):
    #     img = Image.new("RGB", (43 if ch.isupper() or ch.isdigit() else 45, 140), (0, 0, 0))
    #     draw = ImageDraw.Draw(img)
    #     draw.text(
    #         (0, -11 if ch.isupper() or ch.isdigit() else 3), ch,
    #         fill = (255, 255, 255),
    #         font = self._font[0 if ch.isupper() or ch.isdigit() else 1]
    #     )
    #     if img.width > 45:
    #         img = img.resize((45, 140))
    #     return np.array(img)
    # def _draw_fg(self, plate):
    #     img = np.array(Image.new("RGB", (480, 140), (255, 255, 255)))
    #     offset = 15
    #     img[0:140, offset:offset+45] = self._draw_char(plate[0])
    #     offset = offset + 45 + 9
    #     img[0:140, offset:offset+43] = self._draw_char(plate[1])
    #     offset = offset + 43 + 49
    #     for i in range(2, len(plate)):
    #         img[0:140, offset:offset+43] = self._draw_char(plate[i])
    #         offset = offset + 43 + 9
    #     return img
    def _draw_char(self, ch):
        img = Image.new("RGB", (45 if ch.isupper() or ch.isdigit() else 95, 140), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw.text(
            (0, -11 if ch.isupper() or ch.isdigit() else 3), ch,
            fill = (0, 0, 0),
            font = self._font[0 if ch.isupper() or ch.isdigit() else 1]
        )
        if img.width > 45:
            img = img.resize((45, 140))
        return np.array(img)

    def _draw_fg(self, plate):
        img = np.array(Image.new("RGB", (495, 140), (255, 255, 255)))
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

    parser = argparse.ArgumentParser(description="Generate a green plate.")
    parser.add_argument("--background", help="set the backgrond index (default: 0)", type=int, default=0)
    parser.add_argument("plate", help="license plate number (default: 京AD12345)", type=str, nargs="?", default="京AD12345")
    args = parser.parse_args()

    draw = Draw_green()
    plate = draw(args.plate, args.background)
    plt.imshow(plate)
    plt.show()
