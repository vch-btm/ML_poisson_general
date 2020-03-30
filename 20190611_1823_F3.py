# coding=utf-8

from __future__ import print_function, division
import scipy

import sys
import os

if len(sys.argv) > 1:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(sys.argv[1]);

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
from scipy.special import erf

from keras.layers import Layer
from keras import backend as K


class Swish(Layer):
    def __init__(self, beta, **kwargs):
        super(Swish, self).__init__(**kwargs)
        self.beta = K.cast_to_floatx(beta)

    def call(self, inputs):
        return K.sigmoid(self.beta * inputs) * inputs

    def get_config(self):
        config = {'beta': float(self.beta)}
        base_config = super(Swish, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


class Brish(Layer):
    def __init__(self, c1, c2, sigma, **kwargs):
        super(Brish, self).__init__(**kwargs)
        self.c1 = K.cast_to_floatx(c1)
        self.c2 = K.cast_to_floatx(c2)
        self.sigma = K.cast_to_floatx(sigma)

        self.cp = K.cast_to_floatx((self.c1 + self.c2) / 2)
        self.cm = K.cast_to_floatx((self.c1 - self.c2) / 2)
        self.cms = K.cast_to_floatx((self.c1 - self.c2) / np.sqrt(2 * np.pi) * self.sigma)
        self.s1 = K.cast_to_floatx(np.sqrt(2) * self.sigma)
        self.s2 = K.cast_to_floatx(2 * self.sigma ** 2)

    def call(self, inputs):
        return self.cm * inputs * tf.math.erf(inputs / self.s1) + self.cms * K.exp(- K.square(inputs) / self.s2) + self.cp * inputs

    def get_config(self):
        config = {'c1': float(self.c1), 'c2': float(self.c2), 'sigma': float(self.sigma)}
        base_config = super(Brish, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


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

netType = "cycGAN"

versionNr = 1

use_dis_2 = True

#########################################
#########################################

device_name = ""

# device_name = '/device:GPU:0'

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
        # d = LeakyReLU(alpha=0.2)(d)
        d = Swish(beta=swish_beta)(d)
        # d = Brish(c1=brish_c1, c2=brish_c2, sigma=brish_sigma)(d)
        # d = InstanceNormalization()(d)
        # d = BatchNormalization()(d)
        return d

    def convT(layer_input, filters, f_size, nStrides, k_init="glorot_uniform"):
        d = Conv2DTranspose(filters, kernel_size=f_size, strides=nStrides, padding='same', kernel_initializer=k_init)(layer_input)
        d = Swish(beta=swish_beta)(d)
        # d = Brish(c1=brish_c1, c2=brish_c2, sigma=brish_sigma)(d)
        # d = LeakyReLU(alpha=0.2)(d)
        # d = InstanceNormalization()(d)
        # d = BatchNormalization()(d)
        return d

    def block(input, filters, f_size, n_strides):
        k_init = "glorot_uniform"
        # k_init = "he_normal"

        x = conv(input, filters, f_size, 1, k_init)
        x = conv(x, filters, f_size, 1, k_init)
        # x = convT(x, filters, f_size, 2, k_init)
        x = Concatenate()([input, x])
        return x

    d0 = Input(shape=img_shape)

    x = block(d0, 16, 7, 1)
    x = block(x, 16, 5, 1)
    x = block(x, 16, 3, 1)
    x = block(x, 32, 3, 1)
    x = block(x, 32, 3, 1)
    x = block(x, 64, 3, 1)
    x = block(x, 64, 3, 1)
    x = block(x, 64, 3, 1)
    # x = block(x, 64, 3, 1)
    # x = block(x, 64, 3, 1)

    x = block(x, 128, 3, 1)
    x = block(x, 128, 3, 1)
    x = block(x, 128, 3, 1)

    # x = conv(x, 64, 3, 1)
    # x = conv(x, 64, 3, 1)

    x = conv(x, 1, 3, 1)

    return Model(d0, x, name=txt_name)


def build_generator3(txt_name):
    def conv(layer_input, filters, f_size, nStrides, k_init="glorot_uniform"):
        d = Conv2D(filters, kernel_size=f_size, strides=nStrides, padding='same', kernel_initializer=k_init)(layer_input)
        # d = LeakyReLU(alpha=0.2)(d)
        d = Swish(beta=swish_beta)(d)
        # d = Brish(c1=brish_c1, c2=brish_c2, sigma=brish_sigma)(d)
        # d = InstanceNormalization()(d)
        # d = BatchNormalization()(d)
        return d

    def block(i0, i1, filters, f_size, n_strides):
        k_init = "glorot_uniform"
        # k_init = "he_normal"

        x1 = conv(i1, filters, f_size, n_strides, k_init)
        x1 = Concatenate()([x1, i0])

        x2 = conv(x1, filters, f_size, n_strides, k_init)
        x2 = Concatenate()([x2, i1])

        return x1, x2

    d0 = Input(shape=img_shape)

    x = conv(d0, 64, 7, 1)

    x1, x2 = block(d0, x, 64, 5, 1)
    x1, x2 = block(x1, x2, 64, 3, 1)
    x1, x2 = block(x1, x2, 64, 3, 1)
    # x1, x2 = block(x1, x2, 64, 3, 1)
    # x1, x2 = block(x1, x2, 64, 3, 1)
    # x1, x2 = block(x1, x2, 64, 3, 1)
    # x1, x2 = block(x1, x2, 64, 3, 1)
    x1, x2 = block(x1, x2, 128, 3, 1)
    x1, x2 = block(x1, x2, 128, 3, 1)

    x = Concatenate()([x1, x2])

    # x = conv(x, 128, 3, 1)
    # x = conv(x, 128, 3, 1)

    x = conv(x, 1, 3, 1)

    return Model(d0, x, name=txt_name)


def build_generator4(txt_name):
    def conv(layer_input, filters, f_size, nStrides, k_init="glorot_uniform"):
        d = Conv2D(filters, kernel_size=f_size, strides=nStrides, padding='same', kernel_initializer=k_init)(layer_input)
        # d = LeakyReLU(alpha=0.2)(d)
        d = Swish(beta=swish_beta)(d)
        # d = Brish(c1=brish_c1, c2=brish_c2, sigma=brish_sigma)(d)
        # d = InstanceNormalization()(d)
        # d = BatchNormalization()(d)
        return d

    def block(i0, i1, filters, f_size, n_strides):
        k_init = "glorot_uniform"
        # k_init = "he_normal"

        x1 = conv(i1, filters, f_size, n_strides, k_init)
        x1 = Concatenate()([x1, i0])

        x2 = conv(x1, filters, f_size, n_strides, k_init)
        x2 = Concatenate()([x2, i1])

        return x1, x2

    global use_boundaries
    use_boundaries = True

    img_X = Input(shape=img_shape)
    bnd_Y = Input(shape=img_shape)

    x0 = Concatenate()([img_X, bnd_Y])
    x = conv(x0, 64, 7, 1)

    x1, x2 = block(x0, x, 64, 5, 1)
    x1, x2 = block(x1, x2, 64, 3, 1)
    x1, x2 = block(x1, x2, 64, 3, 1)
    x1, x2 = block(x1, x2, 64, 3, 1)
    x1, x2 = block(x1, x2, 64, 3, 1)
    #x1, x2 = block(x1, x2, 64, 3, 1)
    #x1, x2 = block(x1, x2, 64, 3, 1)
    x1, x2 = block(x1, x2, 128, 3, 1)
    x1, x2 = block(x1, x2, 128, 3, 1)
    #x1, x2 = block(x1, x2, 128, 3, 1)
    #x1, x2 = block(x1, x2, 128, 3, 1)

    x = Concatenate()([x1, x2, bnd_Y])

    # x = conv(x, 128, 3, 1)
    # x = conv(x, 128, 3, 1)

    x = conv(x, 1, 3, 1)

    return Model([img_X, bnd_Y], x, name=txt_name)


# imgs_A_bounds =  imgs_A[imgs_A.ndim * (slice(1, -1),)] = 0

def get_boundaries(arr):
    arr2 = np.copy(arr)

    if arr2.ndim == 2:
        arr2[slice(1, -1), slice(1, -1)] = 0.0
    else:
        arr2[:, slice(1, -1), slice(1, -1), :] = 0.0
    return arr2


def build_discriminator(txt_name):
    def d_layer(layer_input, filters, f_size=4):
        d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
        # d = LeakyReLU(alpha=0.2)(d)
        d = Swish(beta=swish_beta)(d)
        # d = Brish(c1=brish_c1, c2=brish_c2, sigma=brish_sigma)(d)
        return d

    d0 = Input(shape=img_shape)

    d1 = d_layer(d0, df)
    d2 = d_layer(d1, df * 2)
    d3 = d_layer(d2, df * 4)
    d4 = d_layer(d3, df * 8)

    validity = Conv2D(1, kernel_size=5, strides=1, padding='same')(d4)

    return Model(d0, validity, name=txt_name)


def build_discriminator2(txt_name):
    def d_layer(layer_input, filters, f_size=4):
        d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
        # d = LeakyReLU(alpha=0.2)(d)
        d = Swish(beta=swish_beta)(d)
        # d = Brish(c1=1.0, c2=0.3, sigma=0.5)(d)
        return d

    # d0 = Input(shape=img_shape)
    img_X = Input(shape=img_shape)
    img_Y = Input(shape=img_shape)

    d0 = Concatenate()([img_X, img_Y])

    d1 = d_layer(d0, df)
    d2 = d_layer(d1, df * 2)
    d3 = d_layer(d2, df * 4)
    d4 = d_layer(d3, df * 8)

    # Concatenate(axis=-1)([img_X, img_Y])

    validity = Conv2D(1, kernel_size=3, strides=1, padding='same')(d4)  # (?, 16, 16, 1)

    return Model([img_X, img_Y], validity, name=txt_name)


########################################################################################################################
########################################################################################################################


def train(epochs, batch_size=1, sample_interval=50, genABepochs=0, genBAepochs=0, actCount=0):
    g_AB.compile(loss='mean_squared_error', optimizer=optimizerG)
    # g_AB.fit(A_train, B_train, epochs=genABepochs, batch_size=batch_size, verbose=2)

    g_BA.compile(loss='mean_squared_error', optimizer=optimizerG)
    # g_BA.fit(B_train, A_train, epochs=genBAepochs, batch_size=batch_size, verbose=2)

    best_loss = np.infty

    start_time = datetime.datetime.now()

    valid = np.ones((batch_size,) + disc_patch)  # Adversarial loss ground truths
    fake = np.zeros((batch_size,) + disc_patch)

    for actEpoch in range(epochs):
        epoch = actEpoch + actCount

        for batch_i, (imgs_A, imgs_B) in enumerate(load_batch(batch_size)):

            if use_boundaries:
                bnds_A = get_boundaries(imgs_A)
                bnds_B = get_boundaries(imgs_B)

                fake_B = g_AB.predict([imgs_A, bnds_B])
                fake_A = g_BA.predict([imgs_B, bnds_A])
            else:
                fake_B = g_AB.predict(imgs_A)
                fake_A = g_BA.predict(imgs_B)

            if use_dis_2:
                dA_loss_real = d_A.train_on_batch([imgs_A, imgs_B], valid)
                dA_loss_fake = d_A.train_on_batch([fake_A, imgs_B], fake)

                dB_loss_real = d_B.train_on_batch([imgs_B, imgs_A], valid)
                dB_loss_fake = d_B.train_on_batch([fake_B, imgs_A], fake)
            else:
                dA_loss_real = d_A.train_on_batch(imgs_A, valid)
                dA_loss_fake = d_A.train_on_batch(fake_A, fake)

                dB_loss_real = d_B.train_on_batch(imgs_B, valid)
                dB_loss_fake = d_B.train_on_batch(fake_B, fake)

            dA_loss = 0.5 * np.add(dA_loss_real, dA_loss_fake)
            dB_loss = 0.5 * np.add(dB_loss_real, dB_loss_fake)
            d_loss = 0.5 * np.add(dA_loss, dB_loss)  # Total disciminator loss

            if use_boundaries:
                g_loss = combined.train_on_batch([imgs_A, imgs_B, bnds_A, bnds_B], [valid, valid, imgs_A, imgs_B, imgs_A, imgs_B, imgs_A, imgs_B])
            else:
                g_loss = combined.train_on_batch([imgs_A, imgs_B], [valid, valid, imgs_A, imgs_B, imgs_A, imgs_B, imgs_A, imgs_B])

            elapsed_time = datetime.datetime.now() - start_time

        if epoch % sample_interval == 0:

            # print("[Epoch {}{}] [D loss: {}, acc: {}] [G loss: {}, adv: {}, recon: {}, id: {}] time: {} ".format(epoch, epochs, d_loss[0], 100 * d_loss[1], g_loss[0], np.mean(g_loss[1:3]), np.mean(g_loss[3:5]), np.mean(g_loss[5:6]), elapsed_time))

            if use_boundaries:
                loss = g_AB.evaluate(x=[A_train, B_train_bnd], y=B_train, verbose=2)
                v_loss = g_AB.evaluate(x=[A_valid, B_valid_bnd], y=B_valid, verbose=2)

                loss2 = g_BA.evaluate(x=[B_train, A_train_bnd], y=A_train, verbose=2)
                v_loss2 = g_BA.evaluate(x=[B_valid, A_valid_bnd], y=A_valid, verbose=2)
            else:
                loss = g_AB.evaluate(x=A_train, y=B_train, verbose=2)
                v_loss = g_AB.evaluate(x=A_valid, y=B_valid, verbose=2)

                loss2 = g_BA.evaluate(x=B_train, y=A_train, verbose=2)
                v_loss2 = g_BA.evaluate(x=B_valid, y=A_valid, verbose=2)

            print("{}: {} G: {} {} {} {}".format(randomTxt, epoch, loss, v_loss, loss - v_loss, time.ctime(time.time())))

            write_log(tensorboard, ['loss'], [loss], epoch)
            write_log(tensorboard, ['val_loss'], [v_loss], epoch)
            write_log(tensorboard, ['zzz_diff'], [loss - v_loss], epoch)

            write_log(tensorboard, ['loss2'], [loss2], epoch)
            write_log(tensorboard, ['val_loss2'], [v_loss2], epoch)
            write_log(tensorboard, ['zzz_diff2'], [loss2 - v_loss2], epoch)

            write_log(tensorboard, ['dA_loss'], [dA_loss[0]], epoch)
            write_log(tensorboard, ['dB_loss'], [dB_loss[0]], epoch)
            write_log(tensorboard, ['gLoss'], [g_loss[0]], epoch)

            d_A.save(os.path.join(path, "{}{}/{}_d_A_{}{}.h5".format(path, subpathID, ID, netType, versionNr)))
            d_B.save(os.path.join(path, "{}{}/{}_d_B_{}{}.h5".format(path, subpathID, ID, netType, versionNr)))
            g_AB.save(os.path.join(path, "{}{}/{}_g_AB_{}{}.h5".format(path, subpathID, ID, netType, versionNr)))
            g_BA.save(os.path.join(path, "{}{}/{}_g_BA_{}{}.h5".format(path, subpathID, ID, netType, versionNr)))
            combined.save(os.path.join(path, "{}{}/{}_comb_{}{}.h5".format(path, subpathID, ID, netType, versionNr)))

            if v_loss < best_loss:
                best_loss = v_loss
                d_A.save("{}{}/{}_d_A_{}{}_best.h5".format(path, subpathID, ID, netType, versionNr))
                d_B.save("{}{}/{}_d_B_{}{}_best.h5".format(path, subpathID, ID, netType, versionNr))
                g_AB.save("{}{}/{}_g_AB_{}{}_best.h5".format(path, subpathID, ID, netType, versionNr))
                g_BA.save("{}{}/{}_g_BA_{}{}_best.h5".format(path, subpathID, ID, netType, versionNr))
                combined.save("{}{}/{}_comb_{}{}_best.h5".format(path, subpathID, ID, netType, versionNr))

            sample_images(epoch)


def sample_images(epoch):
    os.makedirs('{}/imagesAll'.format(subpathID), exist_ok=True)
    r, c = len(images_to_sample), 12

    v_min = 1
    v_max = 2

    if randomTxt == "GLS":
        v_min = 0
        v_max = 1

    # titles = ['Original', 'Translated [1,2]', 'Translated', 'Reconstructed', 'real - fake', 'real - fake',
    #          'Original', 'Translated [1,2]', 'Translated', 'Reconstructed', 'real - fake', 'real - fake']
    fig, axs = plt.subplots(r, c, figsize=(2.75 * c, 2 * r))

    for i in range(r):
        cnt = 0

        img_A = A_test[images_to_sample[i]].reshape((1, img_rows, img_cols, 1))
        img_B = B_test[images_to_sample[i]].reshape((1, img_rows, img_cols, 1))

        if use_boundaries:
            # Translate images to the other domain
            fake_B = g_AB.predict([img_A, get_boundaries(img_B)])
            fake_A = g_BA.predict([img_B, get_boundaries(img_A)])

            # Translate back to original domain
            reconstr_A = g_BA.predict([fake_B, get_boundaries(img_A)])
            reconstr_B = g_AB.predict([fake_A, get_boundaries(img_B)])
        else:
            fake_B = g_AB.predict(img_A)
            fake_A = g_BA.predict(img_B)

            reconstr_A = g_BA.predict(fake_B)
            reconstr_B = g_AB.predict(fake_A)

        gen_imgs = np.concatenate(
            [img_A, fake_A, fake_A, reconstr_A, img_A - fake_A, img_A - fake_A,
             img_B, fake_B, fake_B, reconstr_B, img_B - fake_B, img_B - fake_B])

        for j in range(c):
            ax = axs[i, j]

            switcher = {
                1: {'vmin': np.min(img_A), 'vmax': np.max(img_A)},
                4: {'vmin': -0.05, 'vmax': 0.05, 'cmap': plt.get_cmap("seismic")},
                5: {'vmin': -0.5, 'vmax': 0.5, 'cmap': plt.get_cmap("seismic")},
                6: {'vmin': v_min, 'vmax': v_max},
                7: {'vmin': v_min, 'vmax': v_max},
                10: {'vmin': -0.05, 'vmax': 0.05, 'cmap': plt.get_cmap("seismic")},
                11: {'vmin': -0.5, 'vmax': 0.5, 'cmap': plt.get_cmap("seismic")}
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

# Loss weights
lambda_cycle = 10.0  # Cycle-consistency loss
lambda_id = 0.1 * lambda_cycle  # Identity loss

optimizerC = Adam(0.0001)  # , 0.5)
optimizerD = Adam(0.0001)  # , 0.5)
optimizerG = Adam(0.00001, 0.5)

swish_beta = 1.0
brish_c1 = 1.0
brish_c2 = 0.3
brish_sigma = 0.25

# Build the generators
g_AB = build_generator2("g_AB")
g_BA = build_generator2("g_BA")

lossWeights = [1, 1, 10, 10, 0.1, 0.1, 25, 25]

num_startEpochs = 0
use_boundaries = False

if randomTxt == "Rand2":
    g_AB = build_generator4("g_AB")
    g_BA = build_generator4("g_BA")
    num_startEpochs = 0
    optimizerD = Adam(0.00002, 0.5)
    optimizerC = Adam(0.00002, 0.5)
    optimizerG = Adam(0.0002)  # , 0.5)
    #lossWeights = [1, 1, 0.25, 0.25, 0.001, 0.001, 20, 20]
    lossWeights = [1, 1, 0.25, 0.25, 0.001, 0.001, 50, 50]
    swish_beta = 1

if randomTxt == "F3":
    g_AB = build_generator3("g_AB")
    g_BA = build_generator3("g_BA")
    num_startEpochs = 0
    optimizerD = Adam(0.00002, 0.5)
    optimizerC = Adam(0.00002, 0.5)
    optimizerG = Adam(0.0002)  # , 0.5)
    # lossWeights = [1, 1, 0.25, 0.25, 0.001, 0.001, 20, 20]
    lossWeights = [1, 1, 0.25, 0.25, 0.001, 0.001, 50, 50]
    swish_beta = 1

if randomTxt == "GLS":
    g_AB = build_generator4("g_AB")
    g_BA = build_generator4("g_BA")
    num_startEpochs = 0
    optimizerD = Adam(0.00002, 0.5)
    optimizerC = Adam(0.00002, 0.5)
    optimizerG = Adam(0.0002)  # , 0.5)
    #lossWeights = [1, 1, 0.25, 0.25, 0.001, 0.001, 20, 20]
    lossWeights = [1, 1, 0.25, 0.25, 0.001, 0.001, 50, 50]
    swish_beta = 1

if randomTxt == "RandSQ2":
    g_AB = build_generator2("g_AB")
    g_BA = build_generator2("g_BA")
    optimizerD = Adam(0.0002)  # , 0.5)
    optimizerC = Adam(0.0002)  # , 0.5)
    optimizerG = Adam(0.0002)  # , 0.5)
    lossWeights = [1, 1, 10, 10, 0.1, 0.1, 25, 25]
    # g_AB = build_generator_RandSQ2("g_AB")

if randomTxt == "VOR":
    g_AB = build_generator2("g_AB")
    g_BA = build_generator2("g_BA")
    optimizerD = Adam(0.00002, 0.5)
    optimizerC = Adam(0.00002, 0.5)
    optimizerG = Adam(0.00002, 0.5)
    lossWeights = [1, 1, 10, 10, 0.1, 0.1, 25, 25]

# Build and compile the discriminators
if use_dis_2:
    d_A = build_discriminator2("d_A")
    d_B = build_discriminator2("d_B")
else:
    d_A = build_discriminator("d_A")
    d_B = build_discriminator("d_B")
d_A.compile(loss='mse', optimizer=optimizerD, metrics=['accuracy'])
d_B.compile(loss='mse', optimizer=optimizerD, metrics=['accuracy'])

# Input images from both domains
img_A = Input(shape=img_shape)
img_B = Input(shape=img_shape)

if use_boundaries:
    A_train_bnd = get_boundaries(A_train)
    B_train_bnd = get_boundaries(B_train)

    A_valid_bnd = get_boundaries(A_valid)
    B_valid_bnd = get_boundaries(B_valid)

    bnd_A = Input(shape=img_shape)
    bnd_B = Input(shape=img_shape)

    fake_B = g_AB([img_A, bnd_B])  # Translate images to the other domain
    fake_A = g_BA([img_B, bnd_A])
    reconstr_A = g_BA([fake_B, bnd_A])  # Translate images back to original domain
    reconstr_B = g_AB([fake_A, bnd_B])
    img_A_id = g_BA([img_A, bnd_B])  # Identity mapping of images
    img_B_id = g_AB([img_B, bnd_A])
else:
    fake_B = g_AB(img_A)  # Translate images to the other domain
    fake_A = g_BA(img_B)
    reconstr_A = g_BA(fake_B)  # Translate images back to original domain
    reconstr_B = g_AB(fake_A)
    img_A_id = g_BA(img_A)  # Identity mapping of images
    img_B_id = g_AB(img_B)

d_A.trainable = False  # For the combined model we will only train the generators
d_B.trainable = False

if use_dis_2:
    valid_A = d_B([img_B, fake_A])
    valid_B = d_B([img_A, fake_B])
else:
    valid_A = d_A(fake_A)  # Discriminators determines validity of translated images
    valid_B = d_B(fake_B)

subpathID = "{}/{}".format(randomTxt, ID)

os.makedirs('{}'.format(subpathID), exist_ok=True)

file_name = os.path.basename(sys.argv[0])
copyfile(file_name, "{}/{}".format(subpathID, file_name))

images_to_sample = [37, 112, 1337]

if use_boundaries:
    combined = Model(inputs=[img_A, img_B, bnd_A, bnd_B], outputs=[valid_A, valid_B, reconstr_A, reconstr_B, img_A_id, img_B_id, fake_A, fake_B], name='combined')
else:
    combined = Model(inputs=[img_A, img_B], outputs=[valid_A, valid_B, reconstr_A, reconstr_B, img_A_id, img_B_id, fake_A, fake_B], name='combined')

combined.compile(loss=['mse', 'mse', 'mae', 'mae', 'mae', 'mae', 'mse', 'mse'], loss_weights=lossWeights, optimizer=optimizerC)

tensorboard.set_model(combined)
d_A.summary()
d_B.summary()
g_AB.summary()
g_BA.summary()
combined.summary()
plot_model(d_A, to_file='{}/{}_d_A.png'.format(subpathID, ID), show_shapes=True)
plot_model(d_B, to_file='{}/{}_d_B.png'.format(subpathID, ID), show_shapes=True)
plot_model(g_AB, to_file='{}/{}_g_AB.png'.format(subpathID, ID), show_shapes=True)
plot_model(g_BA, to_file='{}/{}_g_BA.png'.format(subpathID, ID), show_shapes=True)
plot_model(combined, to_file='{}/{}_combined.png'.format(subpathID, ID), show_shapes=True)

train(epochs=2000000, batch_size=batch_size, sample_interval=1, genABepochs=num_startEpochs, genBAepochs=num_startEpochs, actCount=0)
