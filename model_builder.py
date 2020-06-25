from utils import load_data
from utils import plot_data
from utils import plot_loss
from utils import plot_predicted_images
import  kmodel, os

#from data augumention
from ShiftFlipPic import FlipPic
from sklearn.model_selection import train_test_split





os.environ['KERAS_BACKEND'] = 'tensorflow'
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
print(tf.__version__)
config = tf.ConfigProto(log_device_placement=True, allow_soft_placement=True, 
                        gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.7))
                        #allow_growth=True   per_process_gpu_memory_fraction = 0.3
                        #per_process_gpu_memory_fraction = 0.3

sess = tf.Session(config=config)
set_session(sess)


# 加载训练数据
X_train, y_train = load_data()

print("X.shape == {}; X.min == {:.3f}; X.max == {:.3f}".format(
    X_train.shape, X_train.min(), X_train.max()))
print("y.shape == {}; y.min == {:.3f}; y.max == {:.3f}".format(
    y_train.shape, y_train.min(), y_train.max()))

# display one picture
plot_data(X_train[0], y_train[0])

# 创建网络结构
my_model = kmodel.create_model()

# 编译网络模型
my_model = kmodel.compile_model(my_model)

# 训练网络模型
#hist = kmodel.train_model(my_model, X_train, y_train)

modifier = FlipPic() #data argumentation
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

hist = kmodel.train_model(my_model, modifier,
                          train=(X_train,y_train),
                          validation=(X_val,y_val),
                          batch_size=32,epochs=2000,print_every=100)


plot_loss(hist,"Face key points CNN model")

# give a simple predict test
X_test, _ = load_data(test=True)
y_test = my_model.predict(X_test)

#display the predict result
plot_predicted_images(X_test, y_test, 18)

kmodel.save_model(my_model, 'face_keypoints_detection_cnn_model')
