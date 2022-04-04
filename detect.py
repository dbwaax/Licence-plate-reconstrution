
import torch 
import argparse
import os 
import cv2
from tools import cv_imread,cv_imwrite,checkfiles
from skimage import measure
from PIL import Image, ImageDraw, ImageFont
from model.SRRN import Generator
import numpy as np
import matplotlib.pyplot as plt
from plate_gen.blue_plate import Draw
from plate_gen.green_plate import Draw_green
from tqdm import tqdm
from LPRnet.model.LPRNET import LPRNet, CHARS
from LPRnet.model.STN import STNet
from LPRnet.Evaluation import eval, decode,decodess

def load_GPUS(model,model_path,kwargs):
    state_dict = torch.load(model_path,**kwargs)
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    # load params
    model.load_state_dict(new_state_dict)
    return model
def cv2ImgAddText(img, text, pos, textColor=(255,0,0), textSize=10):
    if (isinstance(img, np.ndarray)):  # detect opencv format or not
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    fontText = ImageFont.truetype("../font/NotoSansCJK-Regular.ttc", textSize, encoding="utf-8")
    draw.text(pos, text, textColor, font=fontText)

    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

parser = argparse.ArgumentParser()                                                                                        
parser.add_argument("--test",default=r'./test')
parser.add_argument("--saved_model_g",default='./saved/weather_GAN.pth')
parser.add_argument("--saved_model_s",default='./saved/weather_stn.pth')
parser.add_argument("--saved_model_l",default='./saved/weather_lpr.pth')
parser.add_argument("--dropout_rate",default=0)
parser.add_argument("--output",default='./output')
args = parser.parse_args()
checkfiles(args.output)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
kwargs={'map_location':lambda storage, loc: storage.cuda(0)}
srrn = Generator().to(device)
# lprnet = LPRNet(class_num=len(CHARS), dropout_rate=args.dropout_rate).to(device)
# STN = STNet().to(device)
srrn = load_GPUS(srrn,args.saved_model_g,kwargs)
# lprnet = load_GPUS(lprnet,args.saved_model_l,kwargs)
# STN = load_GPUS(STN,args.saved_model_s,kwargs)

srrn.eval()
# lprnet.eval()
# STN.eval()

costs = []

filelist = os.listdir(args.test)
filelist = [args.test + '/' + i for i in filelist]

fig = plt.figure()
draw = Draw()
draw_g = Draw_green()
count = 0
psnr = 0.
ssim1 = 0.
with torch.no_grad():
    for i in tqdm(filelist):
        name = os.path.basename(i).split('.')[0]
        if(len(name)==7):
            plate = draw(name)
        if(len(name)==8):
            plate = draw_g(name)
        img = np.array(cv2.resize(cv_imread(i), (94, 24)),dtype=np.float32).transpose(2,0,1)
        input = torch.from_numpy(img).unsqueeze(0).to(device) 
        fake_B = srrn(input)
        # transfer = STN(fake_B)
        # logits = lprnet(transfer)
        
        # preds = logits.cpu().detach().numpy()
        # pred_label, _ = decode(preds, CHARS)

        temp = (np.squeeze(fake_B[0].clamp(0,1).cpu().detach().numpy()).transpose(1,2,0)*255).astype(np.uint8)
        cv_imwrite(args.output+'/' + name + '.jpg',temp)