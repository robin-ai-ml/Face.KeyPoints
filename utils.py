import numpy as np
from pandas.io.parsers import read_csv
from sklearn.utils import shuffle
import cv2

import matplotlib.pyplot as plt

from keras.models import load_model


def load_data(test=False):
    """
    当 test 为真, 加载测试数据, 否则加载训练数据 
    """
    FTRAIN = './data/training.csv'
    FTEST = './data/test.csv'
    fname = FTEST if test else FTRAIN
    df = read_csv(fname)

    # 将'Image' 列中 '空白键' 分割的数字们转换为一个 numpy array
    df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' '))

    print("total count:{} ".format(df.count()))

    # 丢弃有缺失值的数据
    df = df.dropna()
    print("after dropna count:{} ".format(df.count()))

    # 将图像的数字从 0 到 255 的整数转换为 0 到 1 的实数
    X = np.vstack(df['Image'].values) / 255.
    X = X.astype(np.float32)

    # 将 X 的每一行转换为一个 96 * 96 * 1 的三维数组
    X = X.reshape(-1, 96, 96, 1)

    # 只有 FTRAIN 包含关键点的数据 (target value)
    if not test:
        y = df[df.columns[:-1]].values
        # 将关键点的值 normalize 到 [-1, 1] 之间
        y = (y - 48) / 48
        # 置乱训练数据
        X, y = shuffle(X, y, random_state=42)
        y = y.astype(np.float32)
    else:
        y = None

    return X, y


def transparentOverlay(src, overlay, pos=(0, 0), scale=1):
    """
    将带透明通道(png图像)的图像 overlay 叠放在src图像上方
    :param src: 背景图像
    :param overlay: 带透明通道的图像 (BGRA)
    :param pos: 叠放的起始位置
    :param scale : overlay图像的缩放因子
    :return: Resultant Image
    """
    if scale != 1:
        overlay = cv2.resize(overlay, (0, 0), fx=scale, fy=scale)

    # overlay图像的高和宽
    h, w, _ = overlay.shape
    # 叠放的起始坐标
    y, x = pos[0], pos[1]

    # 以下被注释的代码是没有优化的版本, 便于理解, 与如下没有注释的版本的功能一样
    """     
    # src图像的高和款
    rows,cols,_ = src.shape  
    for i in range(h):
        for j in range(w):
            if x+i >= rows or y+j >= cols:
                continue
            alpha = float(overlay[i][j][3]/255.0) # 读取alpha通道的值
            src[x+i][y+j] = alpha*overlay[i][j][:3]+(1-alpha)*src[x+i][y+j]
    return src """

    alpha = overlay[:, :, 3]/255.0
    alpha = alpha[..., np.newaxis]
    src[x:x+h, y:y+w, :] = alpha * overlay[:, :, :3] + \
        (1-alpha)*src[x:x+h, y:y+w, :]
    return src


def plot_data(img, landmarks, axis=plt):
    """
    Plot image (img), along with normalized facial keypoints (landmarks)
    """
    axis.imshow(np.squeeze(img), cmap='gray')  # plot the image
    landmarks = landmarks * 48 + 48  # undo the normalization
    # Plot the keypoints
    axis.scatter(landmarks[0::2],
                 landmarks[1::2],
                 marker='o',
                 c='c',
                 s=40)
    axis.show()


def plot_keypoints(img_path,
                   face_cascade=cv2.CascadeClassifier(
                       'haarcascade_frontalface_alt.xml'),
                   model_path='my_model.h5'):
    # TODO: write a function that plots keypoints on arbitrary image containing human
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(1, 1, 1, xticks=[], yticks=[])
    ax.imshow(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB))

    if len(faces) == 0:
        plt.title('no faces detected')
    elif len(faces) > 1:
        plt.title('many faces detected')
        for (x, y, w, h) in faces:
            rectangle = cv2.rectangle(
                img, (x, y), (x+w, y+h), (255, 255, 0), 2)
            ax.imshow(cv2.cvtColor(rectangle, cv2.COLOR_BGR2RGB))
    elif len(faces) == 1:
        plt.title('one face detected')
        x, y, w, h = faces[0]
        bgr_crop = img[y:y+h, x:x+w]
        orig_shape_crop = bgr_crop.shape
        gray_crop = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2GRAY)
        resize_gray_crop = cv2.resize(gray_crop, (96, 96)) / 255.
        model = load_model(model_path)
        landmarks = np.squeeze(model.predict(
            np.expand_dims(np.expand_dims(resize_gray_crop, axis=-1), axis=0)))
        ax.scatter(((landmarks[0::2] * 48 + 48)*orig_shape_crop[0]/96)+x,
                   ((landmarks[1::2] * 48 + 48)*orig_shape_crop[1]/96)+y,
                   marker='o', c='c', s=40)
    plt.show()


def plot_loss(hist, name, plt=plt, RMSE_TF=False):
    '''
    RMSE_TF: if True, then RMSE is plotted with original scale 
    '''
    loss = hist['loss']
    val_loss = hist['val_loss']
    if RMSE_TF:
        loss = np.sqrt(np.array(loss))*48
        val_loss = np.sqrt(np.array(val_loss))*48

    plt.figure(figsize=(8, 8))
    plt.plot(loss, "--", linewidth=3, label="train:"+name)
    plt.plot(val_loss, linewidth=3, label="val:"+name)

    plt.legend()
    plt.grid()
    plt.yscale("log")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.show()

def plot_sample(X,y,axs):
    '''
    kaggle picture is 96 by 96
    y is rescaled to range between -1 and 1
    '''
    axs.imshow(X.reshape(96,96),cmap="gray")
    axs.scatter(48*y[0::2]+ 48,48*y[1::2]+ 48)


def plot_predicted_images(X_test, y_test, num_of_images, plt=plt):
    fig = plt.figure(figsize=(7, 7))
    fig.subplots_adjust(hspace=0.13, wspace=0.0001,
                        left=0, right=1, bottom=0, top=1)

    count = 1
    for irow in range(num_of_images):
        ipic = np.random.choice(X_test.shape[0])
        ax = fig.add_subplot(num_of_images/3, 3, count, xticks=[], yticks=[])
        plot_sample(X_test[ipic], y_test[ipic], ax)
        ax.set_title("images " + str(ipic))
        count += 1
    plt.show()
