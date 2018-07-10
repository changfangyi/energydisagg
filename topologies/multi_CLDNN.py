from keras.models import Sequential, Model
from keras.layers import Input, Dense, Activation, Reshape, Flatten, Conv1D, MaxPooling1D, Dropout, LSTM, TimeDistributed, Bidirectional
from keras.optimizers import RMSprop

MODEL_CONV_FILTERS = 8
MODEL_CONV_KERNEL_SIZE = 4
MODEL_CONV_STRIDES = 1
MODEL_CONV_PADDING = 'same'

def build_model(input_shape, appliances):
    seq_length = input_shape[0]
    # conv
    x = Input(shape=input_shape)
    conv_1 = Conv1D(filters=MODEL_CONV_FILTERS, kernel_size=MODEL_CONV_KERNEL_SIZE, padding=MODEL_CONV_PADDING, activation='linear')(x)
    drop_1 = Dropout(0.12)(conv_1)
    conv_2 = Conv1D(filters=64, kernel_size=4, padding=MODEL_CONV_PADDING, activation='linear')(drop_1)
    drop_2 = Dropout(0.14)(conv_2)
    conv_3 = Conv1D(filters=128, kernel_size=4, padding=MODEL_CONV_PADDING, activation='linear')(drop_2)
    drop_3 = Dropout(0.18)(conv_3)
    conv_4 = Conv1D(filters=128, kernel_size=4, padding=MODEL_CONV_PADDING, activation='linear')(drop_3)
    drop_4 = Dropout(0.2)(conv_4)
    # reshape
    flat_4 = Flatten()(drop_4)
    dense_5 = Dense(seq_length*4, activation='relu')(flat_4)
    drop_5 = Dropout(0.16)(dense_5)
    dense_6 = Dense(128, activation='relu')(drop_5)
    drop_6 = Dropout(0.14)(dense_6)
    dense_7 = Dense(seq_length*4, activation='relu')(drop_6)
    drop_7 = Dropout(0.12)(dense_7)
    reshape_8 = Reshape(target_shape=(seq_length, 4))(drop_7)
    # Initialization
    outputs_disaggregation = []
    for appliance_name in appliances:
        biLSTM_1 = Bidirectional(LSTM(128, return_sequences=True))(reshape_8)
        outputs_disaggregation.append(TimeDistributed(Dense(1, activation='linear'), name=appliance_name.replace(" ", "_"))(biLSTM_1))

    model = Model(inputs=x, outputs=outputs_disaggregation)
    optimizer = RMSprop(lr=0.001, clipnorm=4)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])
    return model
