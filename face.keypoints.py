
from __future__ import division
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf


import numpy as np
import time
import os
import cv2
import kmodel
from utils import transparentOverlay

os.environ['KERAS_BACKEND'] = 'tensorflow'
print(tf.__version__)
config = tf.ConfigProto(log_device_placement=True, allow_soft_placement=True,
                        gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.7))
# allow_growth=True   per_process_gpu_memory_fraction = 0.3
#per_process_gpu_memory_fraction = 0.3

sess = tf.Session(config=config)
set_session(sess)


# os.environ['KMP_DUPLICATE_LIB_OK']='True'

# 加载预先训练好的模型
#my_model = kmodel.load_trained_model('yuan_model_mac')
# 加载自己训练好的模型（测试时取消下面行的注释）
my_model = kmodel.load_trained_model('face_keypoints_detection_cnn_model')

# 创建人脸检测器
face_cascade = cv2.CascadeClassifier(
    'cascades/haarcascade_frontalface_default.xml')

#smileCascade = cv2.CascadeClassifier('cascades/haarcascade_smile.xml')

# 加载摄像头
camera = cv2.VideoCapture(0)

# 加载一个太阳眼镜图像
sunglasses = cv2.imread('sunglass.png', cv2.IMREAD_UNCHANGED)

# 死循环
while True:
    # time.sleep(0.01)

    # 从摄像头获取一张图像
    (_, frame) = camera.read()
    frame = cv2.flip(frame, 1)
    frame2 = np.copy(frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 检测所有的人脸
    faces = face_cascade.detectMultiScale(gray, 1.25, 6)

    # 对每一个检测到的人脸
    for (x, y, w, h) in faces:

        # 只包含人脸的图像
        gray_face = gray[y:y+h, x:x+w]
        color_face = frame[y:y+h, x:x+w]

        # 将人脸图像的值 normalize 在 [0, 1] 之间
        gray_normalized = gray_face / 255

        # 缩放灰度图人脸到 96x96 匹配网络的输入
        original_shape = gray_face.shape  # A Copy for future reference
        face_resized = cv2.resize(
            gray_normalized, (96, 96), interpolation=cv2.INTER_AREA)
        face_resized = face_resized.reshape(1, 96, 96, 1)

        # 预测关键点坐标
        keypoints = my_model.predict(face_resized)

        # 将关键点坐标的值从 [-1, 1] 之间转换为 [0, 96] 之间
        keypoints = keypoints * 48 + 48

        # 缩放彩色图人脸到 96x96 匹配关键点
        face_resized_color = cv2.resize(
            color_face, (96, 96), interpolation=cv2.INTER_AREA)
        face_resized_color2 = np.copy(face_resized_color)

        # 将网络输出的30个值配对为15个tuple对
        points = []
        for i, co in enumerate(keypoints[0][0::2]):
            points.append((co, keypoints[0][1::2][i]))

        # 按照关键点的 left_eyebrow_outer_end_x[7], right_eyebrow_outer_end_x[9]确定眼镜的宽度
        sunglass_width = int((points[7][0]-points[9][0])*1.1)

        # 按照关键点的 nose_tip_y[10], right_eyebrow_inner_end_y[8]确定眼镜的高度
        sunglass_height = int((points[10][1]-points[8][1])/1.1)
        sunglass_resized = cv2.resize(
            sunglasses, (sunglass_width, sunglass_height), interpolation=cv2.INTER_CUBIC)
        face_resized_color = transparentOverlay(face_resized_color, sunglass_resized, pos=(
            int(points[9][0]), int(points[9][1])), scale=1)

        # 将覆盖了眼镜的 face_resized_color 图像转为摄像头捕捉到的原始图像中的大小
        frame[y:y+h, x:x+w] = cv2.resize(face_resized_color,
                                         original_shape, interpolation=cv2.INTER_CUBIC)

        # 在人脸图像中显示关键点坐标
        for keypoint in points:
            cv2.circle(face_resized_color2, keypoint, 1, (0, 255, 0), 1)

        frame2[y:y+h, x:x+w] = cv2.resize(face_resized_color2,
                                          original_shape, interpolation=cv2.INTER_CUBIC)

 

        # 显示加了眼镜的图像
        cv2.imshow("With Glass", frame)
        # 显示添加了关键点的图像
        cv2.imshow("With Keypoints", frame2)

    # 当 'q' 键被点击, 退出循环
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# 释放摄像头, 关闭窗口
camera.release()
cv2.destroyAllWindows()
