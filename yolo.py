# -*- coding: utf-8 -*-
import colorsys
import os
import time
import numpy as np
from keras import backend as K
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw
from yolo3.model import yolo_eval, yolo_body
from yolo3.utils import letterbox_image


class YOLO(object):
    defaults = {
        "modelFilePath": 'saved_model/trained_weights.h5',
        "anchorFilePath": 'model_data/yolo_anchors.txt',
        "classFilePath": 'model_data/voc_classes.txt',
        "score" : 0.3,
        "iou" : 0.45,
        "model_image_size" : (416, 416) #must be a multiple of 32
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls.defaults:
            return cls.defaults[n]
        else:
            return 'Unrecognized attribute name "%s"' %n

    def __init__(self, **kwargs):
        self.__dict__.update(self.defaults) # set up default values
        self.__dict__.update(kwargs) # and update with user overrides
        self.className_list = self.getClassNameList()
        self.anchor_ndarray = self.getAnchorNdarray()
        self.sess = K.get_session()
        self.boxes, self.scores, self.classes = self.generate()


    def getClassNameList(self):
        with open(self.classFilePath) as file:
            className_list = [k.strip() for k in file.readlines() if k.strip() != '']
        return className_list


    def getAnchorNdarray(self):
        with open(self.anchorFilePath) as file:
            number_list = [float(k) for k in file.read().split(',')]
        anchor_ndarray = np.array(number_list).reshape(-1, 2)
        return anchor_ndarray


    def generate(self):
        # 在Keras中，如果模型训练完成后只保存了权重，那么需要先构建网络，再加载权重
        num_anchors = len(self.anchor_ndarray)
        num_classes = len(self.className_list)
        self.yolo_model = yolo_body(Input(shape=(None, None, 3)),
                                    num_anchors//3,
                                    num_classes)
        self.yolo_model.load_weights(self.modelFilePath)
        # 给不同类别的物体准备不同颜色的方框
        hsvTuple_list = [(x / len(self.className_list), 1., 1.)
                      for x in range(len(self.className_list))]
        color_list = [colorsys.hsv_to_rgb(*k) for k in hsvTuple_list]
        color_ndarray = (np.array(color_list) * 255).astype('int')
        self.color_list = [(k[0], k[1], k[2]) for k in color_ndarray]
        # 目标检测的输出：方框box,得分score，类别class
        self.input_image_size = K.placeholder(shape=(2, ))
        boxes, scores, classes = yolo_eval(self.yolo_model.output,
            self.anchor_ndarray,
            len(self.className_list),
            self.input_image_size,
            score_threshold=self.score,
            iou_threshold=self.iou)
        return boxes, scores, classes


    def detect_image(self, image):
        startTime = time.time()
        # 模型网络结构运算所需的数据准备
        boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        image_data = np.array(boxed_image).astype('float') / 255
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
        # 模型网络结构运算
        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_size: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })
        # 调用ImageFont.truetype方法实例化画图字体对象
        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
             size=np.floor(2e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 300
        # 循环绘制若干个方框
        for i, c in enumerate(out_classes):
            # 定义方框上方文字内容
            predicted_class = self.className_list[c]
            score = out_scores[i]
            label = '{} {:.2f}'.format(predicted_class, score)
            # 调用ImageDraw.Draw方法实例化画图对象
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            box = out_boxes[i]
            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            # 如果方框在图片中的位置过于靠上，调整文字区域
            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])
            # 方框厚度为多少，则画多少个矩形
            for j in range(thickness):
                draw.rectangle([left + j, top + j, right - j, bottom - j],
                    outline=self.color_list[c])
            # 绘制方框中的文字
            draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.color_list[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw
        # 打印检测图片使用的时间
        usedTime = time.time() - startTime
        print('检测这张图片用时%.2f秒' %(usedTime))
        return image
    
    def close_session(self):
        self.sess.close()

def detect_video(yolo, video_path, output_path=""):
    import cv2
    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    video_FourCC    = int(vid.get(cv2.CAP_PROP_FOURCC))
    video_fps       = vid.get(cv2.CAP_PROP_FPS)
    video_size      = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    isOutput = True if output_path != "" else False
    if isOutput:
        print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
        print(video_FourCC, video_fps, video_size)
        out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)
    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = time.time()
    cv2.namedWindow("result", cv2.WINDOW_NORMAL)
    cv2.resizeWindow('result', video_size[0], video_size[1])
    while True:
        return_value, frame = vid.read()
        try:
            #图片第1维是宽，第2维是高，第3维是RGB
            #PIL库图片第三维是RGB，cv2库图片第三维正好相反，是BGR
            image = Image.fromarray(frame[...,::-1])
        except Exception as e:
            break
        image = yolo.detect_image(image)
        result = np.asarray(image)
        curr_time = time.time()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1
        if accum_time > 1:
            accum_time = accum_time - 1
            fps = "FPS: " + str(curr_fps)
            curr_fps = 0
        cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.50, color=(255, 0, 0), thickness=2)
        cv2.imshow("result", result[...,::-1])
        if isOutput:
            out.write(result[...,::-1])
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        sleepTime = 0.5
        time.sleep(sleepTime)
    yolo.close_session()

