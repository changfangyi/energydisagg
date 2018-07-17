""" Denoising Auto-Encoder """
from keras.models import Sequential, Model
from keras.layers.convolutional import Conv1D
from keras.layers import Input, Dense, Activation, Reshape, Flatten, Conv1D, MaxPooling1D, Dropout, Subtract


MODEL_CONV_FILTERS = 16
MODEL_CONV_KERNEL_SIZE = 2
MODEL_CONV_STRIDES = 1
MODEL_CONV_PADDING = 'same'


def build_model(input_shape, appliances):
    seq_length = input_shape[0]

    # build it!
    x = Input(shape=input_shape)
    # conv
    conv_1 = Conv1D(filters=MODEL_CONV_FILTERS, kernel_size=MODEL_CONV_KERNEL_SIZE, padding=MODEL_CONV_PADDING,activation='linear')(x)
    conv_2 = Conv1D(filters=MODEL_CONV_FILTERS, kernel_size=MODEL_CONV_KERNEL_SIZE, padding=MODEL_CONV_PADDING, activation='linear')(conv_1)
    # reshape
    conv_2 = Flatten()(conv_2)
    dense_1 = Dense(units=(seq_length)*MODEL_CONV_FILTERS, activation='relu')(conv_2)
    dense_2 = Dense(units=128, activation='relu')(dense_1)
    dense_3 = Dense(units=64, activation='relu')(dense_2)
    dense_4 = Dense(units=128, activation='relu')(dense_3)
    dense_5 = Dense(units=seq_length*MODEL_CONV_FILTERS, activation='relu')(dense_4)
    dense_5= Reshape(target_shape=(seq_length, MODEL_CONV_FILTERS))(dense_5)
    conv_3 = Conv1D(filters=1, kernel_size=MODEL_CONV_KERNEL_SIZE, strides=MODEL_CONV_STRIDES,
                     	padding=MODEL_CONV_PADDING, activation='linear')(dense_5)

    # Initialization
    outputs_disaggregation = []
    for appliance in appliances:
	if appliance == 'bottle warmer':
	   conv_4 = Conv1D(filters=MODEL_CONV_FILTERS, kernel_size=MODEL_CONV_KERNEL_SIZE, padding=MODEL_CONV_PADDING, activation='linear')(Subtract()([x,conv_3]))
    	   #dense_4 = Reshape(target_shape=(seq_length, 16))(conv_4)
	   outputs_disaggregation.append(Conv1D(filters=1, kernel_size=MODEL_CONV_KERNEL_SIZE, strides=MODEL_CONV_STRIDES,
                     	padding=MODEL_CONV_PADDING, activation='linear')(conv_4))
		
	else:
	     outputs_disaggregation.append(conv_3)
    # compile it!
    model = Model(inputs=x, outputs=outputs_disaggregation)
    model.compile(loss='mean_squared_error',
                  optimizer='adam',
                  metrics=['mse', 'mae'])

    return model
