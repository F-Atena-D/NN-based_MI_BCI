

import numpy as np
import scipy.io
import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.layers import Dense, Activation, Conv2D, BatchNormalization, AveragePooling2D, Dropout, Flatten, DepthwiseConv2D, SeparableConv2D, Input
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam


# Assuming your EEG data is in 'eeg_data' with shape (144, 22, 750, 1)
# Your labels: 72 samples of class 0 (right hand), 72 samples of class 1 (left hand)

# Encode labels
labels = np.array([0] * 72 + [1] * 72)

# Load the EEG data from the .mat file
d = 3
mat = scipy.io.loadmat(f'DLR{d}.mat')
eeg_data = mat['EDLR']  # Replace 'data' with the actual key in your .mat file

eegdata = eeg_data.reshape(144, 22, 750, 1)

# Split the data
X_train, X_temp, y_train, y_temp = train_test_split(eegdata, labels, test_size=0.3, random_state=42, stratify=labels)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# Convert labels to categorical format
y_train_categorical = to_categorical(y_train)
y_val_categorical = to_categorical(y_val)
y_test_categorical = to_categorical(y_test)

def EEGNet(nb_classes, Chans=22, Samples=750, dropoutRate=0.5, kernLength=64, F1=8, D=2, F2=16, norm_rate=0.25):
    """ EEGNet implementation for EEG-based signal classification.
    
    Arguments:
    nb_classes -- Number of classes for classification
    Chans -- Number of EEG channels
    Samples -- Number of EEG sample points
    dropoutRate -- Dropout rate for dropout layers
    kernLength -- Length of the temporal convolutional kernel
    F1 -- Number of temporal filters
    D -- Number of spatial filters per temporal filter
    F2 -- Number of pointwise filters
    norm_rate -- Normalization rate for BatchNormalization
    """
    input1 = Input(shape=(Chans, Samples, 1))

    ##################################################################
    block1 = Conv2D(F1, (1, kernLength), padding='same', input_shape=(Chans, Samples, 1), use_bias=False)(input1)
    block1 = BatchNormalization()(block1)
    block1 = DepthwiseConv2D((Chans, 1), use_bias=False, depth_multiplier=D, depthwise_constraint=tf.keras.constraints.max_norm(1.))(block1)
    block1 = BatchNormalization()(block1)
    block1 = Activation('elu')(block1)
    block1 = AveragePooling2D((1, 4))(block1)
    block1 = Dropout(dropoutRate)(block1)

    ##################################################################
    block2 = SeparableConv2D(F2, (1, 16), use_bias=False, padding='same')(block1)
    block2 = BatchNormalization()(block2)
    block2 = Activation('elu')(block2)
    block2 = AveragePooling2D((1, 8))(block2)
    block2 = Dropout(dropoutRate)(block2)

    ##################################################################
    flatten = Flatten(name='flatten')(block2)
    
    dense = Dense(nb_classes, name='dense', kernel_constraint=tf.keras.constraints.max_norm(norm_rate))(flatten)
    softmax = Activation('softmax', name='softmax')(dense)
    
    return Model(inputs=input1, outputs=softmax)

model = EEGNet(nb_classes=2) # 2 classes for binary classification
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train_categorical, validation_data=(X_val, y_val_categorical), epochs=10, batch_size=16)

test_loss, test_accuracy = model.evaluate(X_test, y_test_categorical)
print("Test Accuracy: ", test_accuracy)