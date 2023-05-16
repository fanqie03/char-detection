from PIL import  Image
import numpy as np
import math
import cv2
import re
from .keys import alphabetChinese as alphabet
# from keys import alphabetChinese as alphabet

import onnxruntime as rt
# from util import strLabelConverter, resizeNormalize
from .util import strLabelConverter, resizeNormalize
# from ..utils import reverse_rotate_crop_image

# 绕pointx,pointy顺时针旋转
def Srotate(angle,valuex,valuey,pointx,pointy):
  valuex = np.array(valuex)
  valuey = np.array(valuey)
  sRotatex = (valuex-pointx)*math.cos(angle) + (valuey-pointy)*math.sin(angle) + pointx
  sRotatey = (valuey-pointy)*math.cos(angle) - (valuex-pointx)*math.sin(angle) + pointy
  return [sRotatex,sRotatey]
# ————————————————
# 版权声明：本文为CSDN博主「星夜孤帆」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
# 原文链接：https://blog.csdn.net/qq_38826019/article/details/84233397

def reverse_rotate_crop_image(img, part_img, points, part_points, origin):
    """
    get_rotate_crop_image的逆操作
    img为原图
    part_img为crop后的图
    bbox_points为part_img中对应在原图的bbox, 四个点，左上，右上，右下，左下
    part_points为在part_img中的点[(x, y), (x, y)]
    """
    # np.rot
    points = np.float32(points)
    left = int(np.min(points[:, 0]))
    right = int(np.max(points[:, 0]))
    top = int(np.min(points[:, 1]))
    bottom = int(np.max(points[:, 1]))
    # img_crop = img[top:bottom, left:right, :].copy()
    points[:, 0] = points[:, 0] - left
    points[:, 1] = points[:, 1] - top
    img_crop_width = int(np.linalg.norm(points[0] - points[1]))
    img_crop_height = int(np.linalg.norm(points[0] - points[3]))
    pts_std = np.float32([[0, 0], [img_crop_width, 0],\
        [img_crop_width, img_crop_height], [0, img_crop_height]])
    # print(points, points.shape, pts_std, pts_std.shape)


    M = cv2.getPerspectiveTransform(points, pts_std)
    _, IM = cv2.invert(M)
    raw_points = []
    # if img_crop_height * 1.0 / img_crop_width >= 1.5:
    #     part_points.reverse()

    for point in part_points:
        new_point = point
        if img_crop_height * 1.0 / img_crop_width >= 1.5:
            new_point = Srotate(math.radians(-90), new_point[0], new_point[1], 0 , 0)
            new_point[0] = new_point[0] + img_crop_width
        
        p = np.float32(new_point + [1])
        x, y, z = np.dot(IM, p)
        new_point = [x/z , y/z]
        
        new_point = [int(new_point[0] + left), int(new_point[1] + top)]

        raw_points.append(new_point)    
    return raw_points


def expand_line_to_box(line, w):
    """
    line [(x1, y1), (x2, y2)]
    """
    # radio = 



converter = strLabelConverter(''.join(alphabet))

def softmax(x):
    x_row_max = x.max(axis=-1)
    x_row_max = x_row_max.reshape(list(x.shape)[:-1]+[1])
    x = x - x_row_max
    x_exp = np.exp(x)
    x_exp_row_sum = x_exp.sum(axis=-1).reshape(list(x.shape)[:-1]+[1])
    softmax = x_exp / x_exp_row_sum
    return softmax

def order_points(box):
    '''矩形框顺序排列
    :param box: numpy.array, shape=(4, 2)
    :return:
    '''
    box = np.array(box).reshape((-1, 2))
    center_x, center_y = np.mean(box[:, 0]), np.mean(box[:, 1])
    if np.any(box[:, 0] == center_x) and np.any(box[:, 1] == center_y):  # 有两点横坐标相等，有两点纵坐标相等，菱形
        p1 = box[np.where(box[:, 0] == np.min(box[:, 0]))]
        p2 = box[np.where(box[:, 1] == np.min(box[:, 1]))]
        p3 = box[np.where(box[:, 0] == np.max(box[:, 0]))]
        p4 = box[np.where(box[:, 1] == np.max(box[:, 1]))]
    elif np.all(box[:, 0] == center_x):  # 四个点的横坐标都相同
        y_sort = np.argsort(box[:, 1])
        p1 = box[y_sort[0]]
        p2 = box[y_sort[1]]
        p3 = box[y_sort[2]]
        p4 = box[y_sort[3]]
    elif np.any(box[:, 0] == center_x) and np.all(box[:, 1] != center_y):  # 只有两点横坐标相等，先上下再左右
        p12, p34 = box[np.where(box[:, 1] < center_y)], box[np.where(box[:, 1] > center_y)]
        p1, p2 = p12[np.where(p12[:, 0] == np.min(p12[:, 0]))], p12[np.where(p12[:, 0] == np.max(p12[:, 0]))]
        p3, p4 = p34[np.where(p34[:, 0] == np.max(p34[:, 0]))], p34[np.where(p34[:, 0] == np.min(p34[:, 0]))]
    else:  # 只有两点纵坐标相等，或者是没有相等的，先左右再上下
        p14, p23 = box[np.where(box[:, 0] < center_x)], box[np.where(box[:, 0] > center_x)]
        p1, p4 = p14[np.where(p14[:, 1] == np.min(p14[:, 1]))], p14[np.where(p14[:, 1] == np.max(p14[:, 1]))]
        p2, p3 = p23[np.where(p23[:, 1] == np.min(p23[:, 1]))], p23[np.where(p23[:, 1] == np.max(p23[:, 1]))]

    return np.array([p1, p2, p3, p4]).reshape((-1, 2)).tolist()
class CRNNHandle:
    def __init__(self, model_path):

        self.sess = rt.InferenceSession(model_path)

    def predict(self, image):
        """
        预测
        """
        scale = image.size[1] * 1.0 / 32
        w = image.size[0] / scale
        w = int(w)
        transformer = resizeNormalize((w, 32))

        image = transformer(image)

        image = image.transpose(2, 0, 1)
        transformed_image = np.expand_dims(image, axis=0)

        preds = self.sess.run(["out"], {"input": transformed_image.astype(np.float32)})

        preds = preds[0]


        length  = preds.shape[0]
        preds = preds.reshape(length,-1)

        preds = np.argmax(preds,axis=1)

        preds = preds.reshape(-1)


        sim_pred = converter.decode(preds, length, raw=False)

        return sim_pred



    def predict_rbg(self, crop_im, bbox):
        """
        预测
        """
        scale = crop_im.size[1] * 1.0 / 32
        w = crop_im.size[0] / scale
        w = int(w)


        img = crop_im.resize((w, 32), Image.BILINEAR)
        
        print('图片原始size [w, h]', crop_im.size, 'scale后的size [w, h]', img.size)
        img = np.array(img, dtype=np.float32)
        img -= 127.5
        img /= 127.5
        image = img.transpose(2, 0, 1)
        transformed_image = np.expand_dims(image, axis=0)

        preds = self.sess.run(["out"], {"input": transformed_image.astype(np.float32)})

        preds = preds[0]

        print('crnn模型输出shape', preds.shape, f'每个输出代表的像素大小 {w / preds.shape[0] : .2f}',  )
        length  = preds.shape[0]
        preds = preds.reshape(length,-1)

        # preds = softmax(preds)


        preds = np.argmax(preds,axis=1)

        preds = preds.reshape(-1)

        result = converter.decode(preds, length, raw=True)

        # for item in result:
        raw, line_pred, char_list = result['raw'], result['line_pred'], result['char_list']

        prev_line = [[0, 0], [0, crop_im.size[1]]]
        prev_line = reverse_rotate_crop_image(None, None, bbox, prev_line, (0, 0))
        ws = []
        points = []
        for i, char_item in enumerate(char_list):  # 添加原始坐标
            crop_left = char_item['idx'] * 4 * scale  # 在裁剪中的坐标
            # crop_width = (char_item['idx'] + 1) * 4 * scale
            # raw_left = bbox[0][0] + crop_left # 先简单的加上偏移
            # raw_right = raw_left + crop_width
            # 中线
            line = [[crop_left, 0], [crop_left, crop_im.size[1]]]
            # line = [[crop_left, 0], [crop_left, crop_im.size[1]]]
            # line = [[crop_left + bbox[0][0], 0], [crop_left + bbox[0][0], crop_im.size[1]]]
            
            line = reverse_rotate_crop_image(None, None, bbox, line, (0, 0))
            char_item['line'] = line

            w = ((line[0][0] - prev_line[0][0]) ** 2 + (line[0][1] - prev_line[0][1]) ** 2) ** 0.5
            w = min(w, crop_im.size[1] / 2)
            # left = crop_left
            ws.append([crop_left - w, crop_left + w])

            # rect = [[crop_left - w, 0], [crop_left - w, crop_im.size[1]], [crop_left + w, crop_im.size[1]], [crop_left + w, 0]]
            # rect = reverse_rotate_crop_image(None, None, bbox, rect, (0, 0))
            # char_item['bbox'] = rect

            prev_line = line
        # ws.append(ws[-1])
        # for i in range(1, len(ws) - 1):  # 添加原始坐标
        # for (left, right), char_item in zip(ws, char_list):
        # 调整bbox有重叠的地方
        ws.append([crop_im.size[0], crop_im.size[0]])
        for i in range(len(ws) - 1):
            cur, nxt = ws[i], ws[i+1]
            if cur[1] > nxt[0]:  # 有交集
                distance = abs(cur[1] - nxt[0])
                cur[0] += distance / 2
                cur[1] -= distance / 2
                nxt[0] += distance / 2
                nxt[1] -= distance / 2
        ws.pop()

        for (left, right), char_item in zip(ws, char_list):
            rect = [[left, 0], [left, crop_im.size[1]], [right, crop_im.size[1]], [right, 0]]
            rect = reverse_rotate_crop_image(None, None, bbox, rect, (0, 0))
            rect = order_points(rect)
            char_item['bbox'] = rect

        def group_char_to_word(char_items):
            bboxs = [x['bbox'] for x in char_items]
            points = np.int0(bboxs).reshape((-1, 1, 2))
            rect = cv2.minAreaRect(points)
            bbox = cv2.boxPoints(rect)
            bbox = order_points(bbox)
            bbox = np.int0(bbox)
            return {'value': ''.join(x['char'] for x in char_items), 'bbox': bbox}

        words = []
        # 使用中文 \u4e00-\u9fa5 和英文符号 ,.?;\()[] 进行分割
        matchs = list(re.finditer('[\u4e00-\u9fa5 ,.?:/\\\\\(\)\[\]]', line_pred))
        # if len(matchs) != 0:
        #     print(line_pred, matchs)
        prev_idx = 0
        for match in matchs:
            left, right = match.span()
            if prev_idx != left:
                words.append(group_char_to_word(char_list[prev_idx: left]))
                # print(words[-1])
            words.append(group_char_to_word(char_list[left: right]))
            prev_idx = right
        
        if prev_idx != len(char_list):
            words.append(group_char_to_word(char_list[prev_idx:]))

        # if len(words) != 1:
        #     print(words)
        result['words'] = words

        return result



if __name__ == "__main__":
    im = Image.open("471594277244_.pic.jpg")
    crnn_handle = CRNNHandle(model_path="../models/crnn_lite_lstm_bk.onnx")
    print(crnn_handle.predict(im))