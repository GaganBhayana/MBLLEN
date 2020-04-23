from keras.layers import Input, Conv2D, Conv2DTranspose, Concatenate, MaxPooling2D, UpSampling2D, Add
from keras.applications.vgg19 import VGG19
from keras.models import Model
from DWT import DWT_Pooling, IWT_UpSampling


def down_block(input_layer, filters, kernel_size=(3,3), activation="relu"):
    output = Conv2D(filters, (3,3), padding="same", activation=activation, data_format='channels_last')(input_layer)
    output = Conv2D(filters, kernel_size, padding="valid", activation=activation, data_format='channels_last')(output)
    output = Conv2D(filters*2, kernel_size, padding="valid", activation=activation, data_format='channels_last')(output)
    output = Conv2D(filters*4, kernel_size, padding="valid", activation=activation, data_format='channels_last')(output)
    return output, DWT_Pooling()(output)

def up_block(input_layer, residual_layer, filters, kernel_size=(3,3),activation="relu"):
    output = IWT_UpSampling()(input_layer)
    output = Add()([residual_layer,output])
    output = Conv2DTranspose(filters*2, kernel_size, activation='relu', padding='valid', data_format='channels_last')(output)
    output = Conv2DTranspose(filters, kernel_size, activation='relu', padding='valid', data_format='channels_last')(output)
    output = Conv2DTranspose(3, kernel_size, activation='relu', padding='valid', data_format='channels_last')(output)
    return output

def build_vgg():
    vgg_model = VGG19(include_top=False, weights='imagenet')
    vgg_model.trainable = False
    return Model(inputs=vgg_model.input, outputs=vgg_model.get_layer('block3_conv4').output)

def build_mbllen(input_shape):

    def EM(input, kernel_size, channel):
        down1, pool1 = down_block(input, channel, (kernel_size, kernel_size))
        res = up_block(pool1, down1, channel, (kernel_size, kernel_size))
        return res

    inputs = Input(shape=input_shape)
    FEM = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_last')(inputs)
    EM_com = EM(FEM, 5, 8)

    for j in range(3):
        for i in range(0, 3):
            FEM = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_last')(FEM)
            EM1 = EM(FEM, 5, 8)
            EM_com = Concatenate(axis=3)([EM_com, EM1])

    outputs = Conv2D(3, (1, 1), activation='relu', padding='same', data_format='channels_last')(EM_com)
    return Model(inputs, outputs)
