""" Denoising Auto-Encoder """
from keras.models import Sequential, Model
from keras.layers.convolutional import Conv1D
from keras.layers import Input, Dense, Activation, Reshape, Flatten, Conv1D, MaxPooling1D, Dropout


MODEL_CONV_FILTERS = 8
MODEL_CONV_KERNEL_SIZE = 4
MODEL_CONV_STRIDES = 1
MODEL_CONV_PADDING = 'same'


def build_model(input_shape, appliances):
    seq_length = input_shape[0]

    # build it!
    x = Input(shape=input_shape)
    # conv
    conv_1 = Conv1D(filters=MODEL_CONV_FILTERS, kernel_size=MODEL_CONV_KERNEL_SIZE, padding=MODEL_CONV_PADDING,activation='linear')(x)
    conv_2 = Conv1D(filters=64, kernel_size=12, padding=MODEL_CONV_PADDING, activation='linear')(conv_1)
    conv_3 = Conv1D(filters=128, kernel_size=7, padding=MODEL_CONV_PADDING, activation='linear')(conv_2)

    # reshape
    conv_3 = Flatten()(conv_3 )
    # dense
    dense_1 = Dense(units=seq_length*MODEL_CONV_FILTERS, activation='linear')(conv_3)
    dense_2 = Reshape(target_shape=(seq_length, MODEL_CONV_FILTERS))(dense_1)
    # Initialization
    outputs_disaggregation = []
    for appliance in appliances:
	    outputs_disaggregation.append(Conv1D(filters=1,
                     kernel_size=MODEL_CONV_KERNEL_SIZE,
                     strides=MODEL_CONV_STRIDES,
                     padding=MODEL_CONV_PADDING, activation='linear')(dense_2))
    # compile it!
    model = Model(inputs=x, outputs=outputs_disaggregation)
    model.compile(loss='mean_squared_error',
                  optimizer='adam',
                  metrics=['mse', 'mae'])

    return model
