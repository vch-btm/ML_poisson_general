# coding=utf-8

from __future__ import print_function, division
import scipy

#import normalization
#from keras_contrib.layers.normalization import InstanceNormalization
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate, Conv2DTranspose, add, Lambda, concatenate, GlobalAveragePooling2D
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.regularizers import l1, l2, l1_l2
import datetime
import sys
import keras
import time

from keras.models import load_model
#from keras.layers import Dense, Activation, Dropout, Conv2D, Conv2DTranspose, MaxPool2D, Flatten, Reshape, Input, add, subtract, MaxPooling2D, AveragePooling2D, UpSampling2D, average, Concatenate, concatenate, LeakyReLU, Lambda, GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.utils.vis_utils import plot_model
from keras import backend as K
from keras import optimizers, regularizers
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
import tensorflow as tf
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable, ImageGrid
###########
from glob import glob
import os
from shutil import copyfile


#########################################
#########################################

batch_size = 16

# specification of the problem size
img_rows = 16
img_cols = img_rows

# how many samples for training, testing and validation
numDataTrain = 100000
numDataValid = numDataTrain//10
numDataTest = numDataTrain//10

# further sepcifications
numEpochs = 100000


shuffleData = not True

#randomTxt = "F3_"
#randomTxt = "GLS_"
#randomTxt = ""
#randomTxt = "Rand2_"
#randomTxt = "VOR"
randomTxt = "Rand2_"

netType = "gen"

versionNr = 1


#########################################
#########################################

device_name = '/device:GPU:0'
#device_name = '/device:GPU:1'
#device_name = '/device:GPU:2'
#device_name = '/device:GPU:3'


#device_name = tf.test.gpu_device_name()
#if device_name != '/device:GPU:0':
#    raise SystemError('GPU device not found')

print('Found GPU at: {}'.format(device_name))
print("keras version: {}".format(keras.__version__))
print("tensorflow version: {}".format(tf.__version__))

ID = "{}".format(time.strftime("%Y%m%d_%H%M"))
#path = "/home/horakv/ML/"
path = ""

tensorboard = TensorBoard(log_dir="{}logs/{}_{}{}{}_{}".format(path, img_rows, netType, versionNr, randomTxt, ID))


########################################################################################################################


def write_log(callback, names, logs, batch_no):
    for name, value in zip(names, logs):
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = name
        callback.writer.add_summary(summary, batch_no)
        callback.writer.flush()


########################################################################################################################

def build_generator(txt_name):
    def conv(layer_input, filters, f_size, nStrides):
        d = Conv2D(filters, kernel_size=f_size, strides=nStrides, padding='same', kernel_regularizer=l1_l2(0.00001, 0.00001), activity_regularizer=l1_l2(0.00001, 0.00001))(layer_input)
        d = LeakyReLU(alpha=0.2)(d)
        d = InstanceNormalization()(d)
        #d = BatchNormalization()(d)
        return d

    def convT(layer_input, filters, f_size, nStrides):
        u = Conv2DTranspose(filters, kernel_size=f_size, strides=nStrides, padding='same', kernel_regularizer=l1_l2(0.00001, 0.00001), activity_regularizer=l1_l2(0.00001, 0.00001))(layer_input)
        u = LeakyReLU(alpha=0.2)(u)
        u = InstanceNormalization()(u)
        #u = BatchNormalization()(u)
        return u

    d0 = Input(shape=img_shape)  # Image input

    d1a = conv(d0, 64, 11, 1)  # 16
    d1b = conv(d1a, 64, 11, 1)

    d2a = conv(d1b, 64, 5, 2)  # 8
    d2b = conv(d2a, 64, 5, 1)

    d3a = conv(d2b, 128, 3, 2)  # 4
    d3b = conv(d3a, 128, 3, 1)

    d4a = conv(d3b, 512, 3, 2)  # 2
    d4b = conv(d4a, 512, 3, 1)

    u3a = convT(d4b, 128, 3, 2)  # 4
    u3b = convT(d3a, 128, 3, 1)
    u3 = Concatenate()([u3b, d3b])

    #u3 = Dropout(0.3)(u3)

    u2a = convT(u3, 64, 3, 2)  # 8
    u2b = convT(u2a, 64, 3, 1)
    u2 = Concatenate()([u2b, d2b])

    #u2 = Dropout(0.3)(u2)

    u1a = convT(u2, 64, 3, 2)  # 16
    u1b = convT(u1a, 64, 3, 1)
    u1 = Concatenate()([u1b, d1b])

    #u1 = Dropout(0.3)(u1)


    output_img = Conv2D(1, kernel_size=3, strides=1, activation='relu', padding='same')(u1)

    return Model(d0, output_img, name=txt_name)





########################################################################################################################
########################################################################################################################


def train(epochs, batch_size=1, sample_interval=50):
    start_time = datetime.datetime.now()

    for epoch in range(epochs):
        for batch_i, (imgs_A, imgs_B) in enumerate(load_batch(batch_size)):
            gen.train_on_batch(imgs_A, imgs_B, sample_weight=None, class_weight=None)

            elapsed_time = datetime.datetime.now() - start_time

        if epoch % sample_interval == 0:

            loss = gen.evaluate(A_train, B_train, verbose=2)
            v_loss = gen.evaluate(A_valid, B_valid, verbose=2)

            print("[Epoch {}{}]  [loss {}, v_loss {}] time: {} ".format(epoch, epochs, loss, v_loss, elapsed_time))

            write_log(tensorboard, ['loss'], [loss], epoch)
            write_log(tensorboard, ['val_loss'], [v_loss], epoch)
            write_log(tensorboard, ['zzz_diff'], [loss - v_loss], epoch)

            gen.save(os.path.join(path, "{}{}/{}_gen_{}{}.h5".format(path, ID, ID, netType, versionNr)))

            sample_images(epoch)


def sample_images(epoch):
    os.makedirs('{}/imagesAll'.format(ID), exist_ok=True)
    r, c = len(images_to_sample), 5

    #titles = ['Original', 'Translated [1,2]', 'Translated', 'Reconstructed', 'real - fake', 'real - fake',
    #          'Original', 'Translated [1,2]', 'Translated', 'Reconstructed', 'real - fake', 'real - fake']
    fig, axs = plt.subplots(r, c, figsize=(2.75*c, 2*r))

    for i in range(r):
        cnt = 0

        imgs_A = A_test[images_to_sample[i]].reshape((1, img_rows, img_cols, 1))
        imgs_B = B_test[images_to_sample[i]].reshape((1, img_rows, img_cols, 1))

        # Translate images to the other domain
        fake_B = gen.predict(imgs_A)

        diff_B = imgs_B - fake_B

        gen_imgs = np.concatenate([imgs_B, fake_B, fake_B, diff_B, diff_B])

        for j in range(c):
            ax = axs[i, j]

            switcher = {
                1: {'vmin': 1, 'vmax': 2},
                3: {'vmin':-0.05, 'vmax':0.05, 'cmap':plt.get_cmap("seismic")},
                4: {'vmin':-0.5, 'vmax':0.5, 'cmap':plt.get_cmap("seismic")}
            }

            pcm = ax.imshow(gen_imgs[cnt][:, :, 0], **switcher.get(j, {}))

            #ax.set_title(titles[j])
            ax.axis('off')
            fig.colorbar(pcm, ax=ax)

            cnt += 1

    fig.savefig("{}/imagesAll/{}.png".format(ID, epoch), bbox_inches="tight")
    plt.close()


def load_batch(batch_size=1):
    order = np.random.choice(range(total_samples), total_samples, replace=False)

    for i in range(n_batches):
        imgs_A = A_train[order[i * batch_size:(i + 1) * batch_size]].reshape((batch_size, img_rows, img_cols, 1))
        imgs_B = B_train[order[i * batch_size:(i + 1) * batch_size]].reshape((batch_size, img_rows, img_cols, 1))

        yield imgs_A, imgs_B

########################################################################################################################

# where to find the data
pathData = "{}MLdata/{}x{}/".format(path, img_rows, img_cols)

# load data
numDataZ = 150000

if numDataTrain + numDataValid + numDataTest > 150000:
    numDataZ = 1500000

xAllData = np.load("{}{}data{}ZZZ.npy".format(pathData, str(numDataZ), randomTxt)).reshape((-1, img_rows, img_cols, 1))
yAllData = np.load("{}{}solu{}ZZZ.npy".format(pathData, str(numDataZ), randomTxt)).reshape((-1, img_rows, img_cols, 1))

if shuffleData:
    randomize = np.arange(len(xAllData))
    np.random.shuffle(randomize)
    xAllData = xAllData[randomize]
    yAllData = yAllData[randomize]

A_train = xAllData[:numDataTrain]
A_valid = xAllData[numDataTrain:(numDataTrain + numDataValid)]
A_test = xAllData[(numDataTrain + numDataValid):(numDataTrain + numDataValid + numDataTest)]

B_train = yAllData[:numDataTrain]
B_valid = yAllData[numDataTrain:(numDataTrain + numDataValid)]
B_test = yAllData[(numDataTrain + numDataValid):(numDataTrain + numDataValid + numDataTest)]

channels = 1
img_shape = (img_rows, img_cols, 1)

n_batches = int(numDataTrain / batch_size)
total_samples = n_batches * batch_size

#########################

# Build the generator
optimizer = Adam(0.00002, 0.5)
gen = build_generator("generator")
gen.compile(loss='mean_squared_error', optimizer=optimizer)

tensorboard.set_model(gen)

os.makedirs('{}'.format(ID), exist_ok=True)
file_name = os.path.basename(sys.argv[0])
copyfile(file_name, "{}/{}".format(ID, file_name))

gen.summary()
plot_model(gen, to_file='{}/{}_gen.png'.format(ID, ID), show_shapes=True)

images_to_sample = [37, 112, 1337]


train(epochs=2000000, batch_size=batch_size, sample_interval=1)
