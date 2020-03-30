# coding=utf-8

from __future__ import print_function, division
import scipy

import sys
import os

if len(sys.argv) > 1:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(sys.argv[1])

# import normalization
# from keras_contrib.layers.normalization import InstanceNormalization
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate, Conv2DTranspose, add, Lambda, concatenate, GlobalAveragePooling2D, average
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
import datetime
import keras
import time

from keras.models import load_model
# from keras.layers import Dense, Activation, Dropout, Conv2D, Conv2DTranspose, MaxPool2D, Flatten, Reshape, Input, add, subtract, MaxPooling2D, AveragePooling2D, UpSampling2D, average, Concatenate, concatenate, LeakyReLU, Lambda, GlobalAveragePooling2D
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

# from AdamW1 import AdamW1
# from AdamW2 import AdamW2

#########################################
#########################################

batch_size = 16

# specification of the problem size
img_rows = 16
img_cols = img_rows

# how many samples for training, testing and validation
numDataTrain = 100000
numDataValid = numDataTrain // 10
numDataTest = numDataTrain // 10

# further sepcifications
numEpochs = 100000

shuffleData = not True

# randomTxt = "F3"
# randomTxt = "GLS"
# randomTxt = ""
randomTxt = "Rand2"
# randomTxt = "VOR"

netType = "simGAN"

versionNr = 1

use_dis_2 = True

#########################################
#########################################

device_name = '/device:GPU:0'

if len(sys.argv) > 1:
    device_name = '/device:GPU:{}'.format(sys.argv[1])
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    randomTxt = sys.argv[2]
else:
    device_name = tf.test.gpu_device_name()
    if device_name != '/device:GPU:0':
        raise SystemError('GPU device not found')

print('Found GPU at: {}'.format(device_name))
print("keras version: {}".format(keras.__version__))
print("tensorflow version: {}".format(tf.__version__))

ID = "{}_{}".format(time.strftime("%Y%m%d_%H%M"), randomTxt)
# path = "/home/horakv/ML/"
path = ""

tensorboard = TensorBoard(log_dir="{}logs/{}/{}_{}_{}_{}".format(path, randomTxt, ID, img_rows, netType, versionNr))


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
        d = Conv2D(filters, kernel_size=f_size, strides=nStrides, padding='same')(layer_input)
        d = LeakyReLU(alpha=0.2)(d)
        # d = InstanceNormalization()(d)
        # d = BatchNormalization()(d)
        return d

    def convT(layer_input, filters, f_size, nStrides):
        u = Conv2DTranspose(filters, kernel_size=f_size, strides=nStrides, padding='same')(layer_input)
        u = LeakyReLU(alpha=0.2)(u)
        # u = InstanceNormalization()(u)
        # u = BatchNormalization()(u)
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
    u3 = Concatenate()([u3a, u3b, d3a, d3b])

    u2a = convT(u3, 64, 3, 2)  # 8
    u2b = convT(u2a, 64, 3, 1)
    u2 = Concatenate()([u2a, u2b, d2a, d2b])

    u1a = convT(u2, 64, 3, 2)  # 16
    u1b = convT(u1a, 64, 3, 1)
    u1 = Concatenate()([u1a, u1b, d1a, d1b, d0])

    output_img = Conv2D(1, kernel_size=3, strides=1, activation='relu', padding='same')(u1)

    return Model(d0, output_img, name=txt_name)


def build_generator2(txt_name):
    def conv(layer_input, filters, f_size, nStrides, k_init="glorot_uniform"):
        d = Conv2D(filters, kernel_size=f_size, strides=nStrides, padding='same', kernel_initializer=k_init)(layer_input)
        d = LeakyReLU(alpha=0.2)(d)
        # d = InstanceNormalization()(d)
        # d = BatchNormalization()(d)
        return d

    def convT(layer_input, filters, f_size, nStrides, k_init="glorot_uniform"):
        u = Conv2DTranspose(filters, kernel_size=f_size, strides=nStrides, padding='same', kernel_initializer=k_init)(layer_input)
        u = LeakyReLU(alpha=0.2)(u)
        # u = InstanceNormalization()(u)
        # u = BatchNormalization()(u)
        return u

    def block(input, filters, f_size, n_strides):
        # k_init = "glorot_uniform"
        k_init = "he_normal"

        x = conv(input, filters, f_size, 1, k_init)
        x = conv(x, filters, f_size, 1, k_init)
        # x = convT(x, filters, f_size, 2, k_init)
        x = Concatenate()([input, x])

        return x

    d0 = Input(shape=img_shape)

    x = block(d0, 64, 3, 1)
    x = block(x, 64, 3, 1)
    x = block(x, 64, 3, 1)
    x = block(x, 64, 3, 1)
    x = block(x, 64, 3, 1)
    x = block(x, 64, 3, 1)
    x = block(x, 64, 3, 1)
    x = block(x, 64, 3, 1)
    x = block(x, 64, 3, 1)
    x = block(x, 64, 3, 1)
    x = block(x, 64, 3, 1)
    x = block(x, 64, 3, 1)
    x = block(x, 64, 3, 1)
    x = block(x, 64, 3, 1)
    x = block(x, 64, 3, 1)
    x = block(x, 64, 3, 1)
    x = block(x, 64, 3, 1)
    x = block(x, 64, 3, 1)
    x = block(x, 64, 3, 1)
    x = block(x, 64, 3, 1)
    x = block(x, 64, 3, 1)

    x = conv(x, 64, 3, 1)
    x = conv(x, 64, 3, 1)

    x = conv(x, 1, 3, 1)

    # final = Dropout(0.5)(final)
    # final = Flatten()(final)

    return Model(d0, x, name=txt_name)


def build_generator_GLS2(txt_name):
    d0 = Input(shape=img_shape)

    x = Flatten()(d0)

    x = Dense(2048, activation='relu')(x)
    x = Dense(512, activation='relu')(x)
    # x = Dense(128, activation='relu')(x)
    x = Dense(512, activation='relu')(x)
    x = Dense(2048, activation='relu')(x)

    x = Dense(256, activation='relu')(x)

    x = Dropout(0.5)(x)
    x = Reshape((16, 16, 1))(x)

    return Model(d0, x, name=txt_name)


def build_generator3(txt_name):
    def conv(layer_input, filters, f_size, n_strides):
        d = Conv2D(filters, kernel_size=f_size, strides=n_strides, padding='same')(layer_input)
        d = LeakyReLU(alpha=0.2)(d)
        # d = InstanceNormalization()(d)
        # d = BatchNormalization()(d)
        return d

    def block(i0, i1, filters, f_size, n_strides):
        x1 = conv(i1, filters, f_size, n_strides)
        x1 = Concatenate()([x1, i0])

        x2 = conv(x1, filters, f_size, n_strides)
        x2 = Concatenate()([x2, i1])

        return x1, x2

    d0 = Input(shape=img_shape)

    x = conv(d0, 64, 7, 1)

    x1, x2 = block(d0, x, 64, 5, 1)
    x1, x2 = block(x1, x2, 64, 3, 1)
    x1, x2 = block(x1, x2, 64, 3, 1)
    x1, x2 = block(x1, x2, 64, 3, 1)
    x1, x2 = block(x1, x2, 64, 3, 1)
    x1, x2 = block(x1, x2, 64, 3, 1)
    x1, x2 = block(x1, x2, 64, 3, 1)
    x1, x2 = block(x1, x2, 64, 3, 1)

    x = Concatenate()([x1, x2])

    x = conv(x, 128, 3, 1)
    x = conv(x, 128, 3, 1)

    x = conv(x, 1, 3, 1)

    return Model(d0, x, name=txt_name)


def build_discriminator(txt_name):
    def d_layer(layer_input, filters, num_strides=2, f_size=3):
        d = Conv2D(filters, kernel_size=f_size, strides=num_strides, padding='same')(layer_input)
        d = LeakyReLU(alpha=0.2)(d)
        return d

    d0 = Input(shape=img_shape)

    d1 = d_layer(d0, df, 1)
    d2 = d_layer(d1, df * 2)
    d3 = d_layer(d2, df * 4)
    d4 = d_layer(d3, df * 8)

    validity = Conv2D(1, kernel_size=3, strides=2, padding='same')(d4)

    return Model(d0, validity, name=txt_name)


def build_discriminator2(txt_name):
    def d_layer(layer_input, filters, num_strides=1, f_size=3):
        d = Conv2D(filters, kernel_size=f_size, strides=num_strides)(layer_input)
        d = LeakyReLU(alpha=0.2)(d)
        return d

    # d0 = Input(shape=img_shape)
    img_X = Input(shape=img_shape)
    img_Y = Input(shape=img_shape)

    d0 = Concatenate()([img_X, img_Y])

    d = d_layer(d0, df)
    d = d_layer(d, df * 2)
    d = d_layer(d, df * 4)
    d = d_layer(d, df * 8)
    d = d_layer(d, df * 16)
    d = d_layer(d, df * 32)
    d = d_layer(d, df * 64)

    # Concatenate(axis=-1)([img_X, img_Y])

    validity = Conv2D(1, kernel_size=2, strides=1)(d)  # (?, 16, 16, 1)

    return Model([img_X, img_Y], validity, name=txt_name)


########################################################################################################################
########################################################################################################################


def train(epochs, batch_size=1, sample_interval=50, genABepochs=0, actCount=0):
    g_AB.compile(loss='mean_squared_error', optimizer=optimizerG)
    g_AB.fit(A_train, B_train, epochs=genABepochs, batch_size=batch_size, verbose=2)

    best_loss = np.infty

    start_time = datetime.datetime.now()

    valid = np.ones((batch_size,) + disc_patch)  # Adversarial loss ground truths
    fake = np.zeros((batch_size,) + disc_patch)

    for actEpoch in range(epochs):
        epoch = actEpoch + actCount

        for batch_i, (imgs_A, imgs_B) in enumerate(load_batch(batch_size)):
            fake_B = g_AB.predict(imgs_A)  # Translate images to opposite domain

            if use_dis_2:
                dB_loss_real = d_B.train_on_batch([imgs_B, imgs_A], valid)
                dB_loss_fake = d_B.train_on_batch([fake_B, imgs_A], fake)
            else:
                dB_loss_real = d_B.train_on_batch(imgs_B, valid)
                dB_loss_fake = d_B.train_on_batch(fake_B, fake)

            dB_loss = 0.5 * np.add(dB_loss_real, dB_loss_fake)
            d_loss = dB_loss

            if use_dis_2:
                c_loss = combined.train_on_batch([imgs_A, imgs_B], [valid, imgs_B])
            else:
                c_loss = combined.train_on_batch([imgs_A], [valid, imgs_B])

            # elapsed_time = datetime.datetime.now() - start_time

        if epoch % sample_interval == 0:
            loss = g_AB.evaluate(x=A_train, y=B_train, verbose=2)
            v_loss = g_AB.evaluate(x=A_valid, y=B_valid, verbose=2)

            print(
                "{}: {} G: {} {} {}".format(randomTxt, epoch, loss, v_loss,
                                            # c_loss[0], d_loss[0], 100 * d_loss[1], elapsed_time,
                                            time.ctime(time.time())))

            if v_loss < best_loss:
                best_loss = v_loss
                d_B.save("{}{}/{}_d_B_{}{}_best.h5".format(path, subpathID, ID, netType, versionNr))
                g_AB.save("{}{}/{}_g_AB_{}{}_best.h5".format(path, subpathID, ID, netType, versionNr))
                combined.save("{}{}/{}_comb_{}{}_best.h5".format(path, subpathID, ID, netType, versionNr))

            write_log(tensorboard, ['loss'], [loss], epoch)
            write_log(tensorboard, ['val_loss'], [v_loss], epoch)
            write_log(tensorboard, ['zzz_diff'], [loss - v_loss], epoch)
            write_log(tensorboard, ['dB_loss'], [dB_loss[0]], epoch)
            write_log(tensorboard, ['gLoss'], [c_loss[0]], epoch)

            d_B.save(os.path.join(path, "{}{}/{}_d_B_{}{}.h5".format(path, subpathID, ID, netType, versionNr)))
            g_AB.save(os.path.join(path, "{}{}/{}_g_AB_{}{}.h5".format(path, subpathID, ID, netType, versionNr)))
            combined.save(os.path.join(path, "{}{}/{}_comb_{}{}.h5".format(path, subpathID, ID, netType, versionNr)))

            sample_images(epoch)


def sample_images(epoch):
    os.makedirs('{}/imagesAll'.format(subpathID), exist_ok=True)
    r, c = len(images_to_sample), 5

    v_min = 1
    v_max = 2

    if randomTxt == "GLS":
        v_min = 0
        v_max = 1

    # titles = ['Original', 'Translated [1,2]', 'Translated', 'real - fake', 'real - fake']
    fig, axs = plt.subplots(r, c, figsize=(2.75 * c, 2 * r))

    for i in range(r):
        cnt = 0

        img_A = A_test[images_to_sample[i]].reshape((1, img_rows, img_cols, 1))
        img_B = B_test[images_to_sample[i]].reshape((1, img_rows, img_cols, 1))
        fake_B = g_AB.predict(img_A)

        gen_imgs = np.concatenate([img_B, fake_B, fake_B, img_B - fake_B, img_B - fake_B])

        for j in range(c):
            ax = axs[i, j]

            switcher = {
                1: {'vmin': v_min, 'vmax': v_max},
                3: {'vmin': -0.05, 'vmax': 0.05, 'cmap': plt.get_cmap("seismic")},
                4: {'vmin': -0.5, 'vmax': 0.5, 'cmap': plt.get_cmap("seismic")}
            }

            pcm = ax.imshow(gen_imgs[cnt][:, :, 0], **switcher.get(j, {}))

            # ax.set_title(titles[j])
            ax.axis('off')
            fig.colorbar(pcm, ax=ax)

            cnt += 1

    fig.savefig("{}/imagesAll/{}.png".format(subpathID, epoch), bbox_inches="tight")
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

xAllData = np.load("{}{}data{}.npy".format(pathData, str(numDataZ), randomTxt)).reshape((-1, img_rows, img_cols, 1))
yAllData = np.load("{}{}solu{}.npy".format(pathData, str(numDataZ), randomTxt)).reshape((-1, img_rows, img_cols, 1))

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

patch = int(img_rows / 2 ** 4)  # Calculate output shape of D (PatchGAN)
disc_patch = (patch, patch, 1)

df = 32

optimizerC = Adam(0.0001)  # , 0.5)
optimizerD = Adam(0.0001)  # , 0.5)
optimizerG = Adam(0.00001, 0.5)

lossWeights = [1, 0.1]

num_startEpochs = 0

if use_dis_2:
    d_B = build_discriminator2("d_B")
else:
    d_B = build_discriminator("d_B")

d_B.compile(loss='mse', optimizer=optimizerD, metrics=['accuracy'])
g_AB = build_generator3("g_AB")

if randomTxt == "Rand2":
    g_AB = build_generator2("g_AB")
    num_startEpochs = 0
    optimizerD = Adam(0.00001, 0.75)
    optimizerC = Adam(0.00001, 0.75)
    optimizerG = Adam(0.0002)  # , 0.5)
    lossWeights = [1, 25]

if randomTxt == "F3":
    g_AB = build_generator2("g_AB")
    optimizerD = Adam(0.00002)  # , 0.5)
    optimizerC = Adam(0.00002)  # , 0.5)
    optimizerG = Adam(0.00002)  # , 0.5)
    lossWeights = [1, 0.5]

if randomTxt == "GLS":
    g_AB = build_generator2("g_AB")
    optimizerD = Adam(0.00002, 0.5)
    optimizerC = Adam(0.00002, 0.5)
    optimizerG = Adam(0.00002, 0.5)
    lossWeights = [1, 10]
    # g_AB = build_generator_GLS("g_AB")

if randomTxt == "RandSQ2":
    g_AB = build_generator2("g_AB")
    optimizerD = Adam(0.0002)  # , 0.5)
    optimizerC = Adam(0.0002)  # , 0.5)
    optimizerG = Adam(0.0002)  # , 0.5)
    lossWeights = [1, 10]
    # g_AB = build_generator_RandSQ2("g_AB")

if randomTxt == "VOR":
    g_AB = build_generator2("g_AB")
    optimizerD = Adam(0.00002, 0.5)
    optimizerC = Adam(0.00002, 0.5)
    optimizerG = Adam(0.00002, 0.5)
    lossWeights = [1, 0.5]

img_A = Input(shape=img_shape)
img_B = Input(shape=img_shape)
fake_B = g_AB(img_A)  # Translate images to the other domain

d_B.trainable = False

if use_dis_2:
    valid_B = d_B([img_A, fake_B])
    combined = Model(inputs=[img_A, img_B], outputs=[valid_B, fake_B], name='combined')
else:
    valid_B = d_B(fake_B)
    combined = Model(inputs=img_A, outputs=[valid_B, fake_B], name='combined')

tensorboard.set_model(combined)
combined.compile(loss=['mse', 'mse'], loss_weights=lossWeights, optimizer=optimizerC)

subpathID = "{}/{}".format(randomTxt, ID)

os.makedirs('{}'.format(subpathID), exist_ok=True)

file_name = os.path.basename(sys.argv[0])
copyfile(file_name, "{}/{}".format(subpathID, file_name))

d_B.summary()
g_AB.summary()
combined.summary()
plot_model(d_B, to_file='{}/{}_d_B.png'.format(subpathID, ID), show_shapes=True)
plot_model(g_AB, to_file='{}/{}_g_AB.png'.format(subpathID, ID), show_shapes=True)
plot_model(combined, to_file='{}/{}_combined.png'.format(subpathID, ID), show_shapes=True)

images_to_sample = [37, 112, 1337]

train(epochs=2000000, batch_size=batch_size, sample_interval=1, genABepochs=num_startEpochs, actCount=0)
