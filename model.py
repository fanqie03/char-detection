import onnxruntime as ort
ort.set_default_logger_severity(3)  # ERROR level 忽略 warning
from config import *
from crnn import CRNNHandle
# from angnet import  AngleNetHandle
from utils import draw_bbox, crop_rect, sorted_boxes, get_rotate_crop_image
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
import copy
from dbnet.dbnet_infer import DBNET
import time
import traceback
import random
import argparse


class  OcrHandle(object):
    def __init__(self):
        self.text_handle = DBNET(model_path)
        self.crnn_handle = CRNNHandle(crnn_model_path)
        # if angle_detect:
        #     self.angle_handle = AngleNetHandle(angle_net_path)


    def crnnRecWithBox(self,im, boxes_list,score_list):
        """
        crnn模型，ocr识别
        @@model,
        @@converter,
        @@im:Array
        @@text_recs:text box
        @@ifIm:是否输出box对应的img

        """
        results = []
        boxes_list = sorted_boxes(np.array(boxes_list))

        # 取消angle
        # line_imgs = []
        # for index, (box, score) in enumerate(zip(boxes_list[:angle_detect_num], score_list[:angle_detect_num])):
        #     tmp_box = copy.deepcopy(box)
        #     partImg_array = get_rotate_crop_image(im, tmp_box.astype(np.float32))
        #     partImg = Image.fromarray(partImg_array).convert("RGB")
        #     line_imgs.append(partImg)

        # angle_res = False
        # if angle_detect:
        #     angle_res = self.angle_handle.predict_rbgs(line_imgs)

        count = 1
        for index, (box ,score) in enumerate(zip(boxes_list,score_list)):

            tmp_box = copy.deepcopy(box)
            partImg_array = get_rotate_crop_image(im, tmp_box.astype(np.float32))


            partImg = Image.fromarray(partImg_array).convert("RGB")

            # if angle_detect and angle_res:
            #     partImg = partImg.rotate(180)


            # if not is_rgb:
            #     partImg = partImg.convert('L')

            # try:
                # if is_rgb:
            result = self.crnn_handle.predict_rbg(partImg, copy.deepcopy(box))  ##识别的文本
                # else:
                #     simPred = self.crnn_handle.predict(partImg)  ##识别的文本
            # except Exception as e:
            #     print(traceback.format_exc())
            #     continue

            # if simPred.strip() != '':
            #     results.append([tmp_box,"{}、 ".format(count)+  simPred,score])
            #     count += 1
            results.append({'bbox': tmp_box, 'result': result, 'score': score})

        return results


    def text_predict(self,img,short_size):
        boxes_list, score_list = self.text_handle.process(np.asarray(img).astype(np.uint8),short_size=short_size)
        result = self.crnnRecWithBox(np.array(img), boxes_list, score_list)

        return result

def get_color():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--images', default=[], nargs='+')
    parser.add_argument('--level', default='char', choices=['char', 'word', 'line'])
    args = parser.parse_args()

    os.makedirs('output', exist_ok=True)
    ocrhandle = OcrHandle()
    for pic_idx, path in enumerate(args.images):
        short_size = 960
        img = Image.open(path).convert('RGB')
        print(np.array(img).shape)
        # print(img.size)
        res = ocrhandle.text_predict(img,short_size)

        img_detected = img.copy()
        img_draw = ImageDraw.Draw(img_detected)
        colors = ['red', 'green', 'blue', "purple"]

        for i, r in enumerate(res):
            rect, txt, confidence = r['bbox'], r['result'], r['score']
            # print(txt)

            x1,y1,x2,y2,x3,y3,x4,y4 = rect.reshape(-1)
            size = max(min(x2-x1,y3-y2) // 2 , 15 )

            myfont = ImageFont.truetype("仿宋_GB2312.ttf", size=size)
            fillcolor = get_color()
            if args.level == 'line':
                # 整个句子
                img_draw.text((x1, y1 - size ), str(txt['line_pred']), font=myfont, fill=fillcolor,)
                for xy in [(x2, y2, x3, y3 ), (x3 , y3 , x4, y4), (x4, y4, x1, y1)]:  # 右 下 左
                    img_draw.line(xy=xy, fill=fillcolor, width=2)

            elif args.level == 'char':
                # 针对单个字符，
                for j, char_item in enumerate( txt['char_list']):
                    char, ((x1, y1), (x2, y2)) = char_item['char'], char_item['line']
                    img_draw.text((x1, y1 - size ), str(char), font=myfont, fill=fillcolor,)
                    center = ((x2+x1)/2, (y2+y1)/2)

                    x1,y1,x2,y2,x3,y3,x4,y4 = np.int32(char_item['bbox']).reshape(-1)
                    color = get_color()
                    # for xy in [(x2, y2, x3, y3 ), (x3 , y3 , x4, y4), (x4, y4, x1, y1)]:  # 右 下 左
                    #     img_draw.line(xy=xy, fill=fillcolor, width=2)
                    img_draw.ellipse([(x1, y1), (x3, y3)], outline=color)

            elif args.level == 'word':
                # 如果是中文就单个字符单个字符，如果是英文就按照word级别分开
                for j, word in enumerate(txt['words']):

                    x1,y1,x2,y2,x3,y3,x4,y4 = np.int32(word['bbox']).reshape(-1)

                    color = get_color()
                    
                    img_draw.ellipse([(x1, y1), (x3, y3)], outline=color)

                    img_draw.text((x1, y1 - size ), str(word['value']), font=myfont, fill=color,)

        img_detected = img_detected.convert('RGB')
        img_detected.save(f'output/{pic_idx:02}.jpg', format='JPEG')