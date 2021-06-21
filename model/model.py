import pandas as pd
import os
import tensorflow
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import train_test_split


print('Data PATH를 입력해주세요')
path = input()
os.chdir(path)


data = pd.read_csv('dataset.csv')

data_y = data['label']
data_x = data.drop(['label'], axis = 1)

del data

# Data normalization
data_x = data_x/255.0
data_x = data_x.values.reshape(-1,28,28,1)
data_y = to_categorical(data_y, num_classes = 14)

# Train, Validation Split
x_train, x_val, y_train, y_val = train_test_split(data_x, data_y, test_size = .1 , random_state = 1234, stratify = data_y)

# Image augmentation(Operator쪽 데이터가 부족하니 증강시켜보자)
augmentation = ImageDataGenerator(featurewise_center = False,
                                 samplewise_center = False,
                                 featurewise_std_normalization = False,
                                 samplewise_std_normalization = False,
                                 zca_whitening = False,
                                 rotation_range = 10,
                                 zoom_range = 0.1,
                                 width_shift_range = 0.1,
                                 height_shift_range = 0.1,
                                 horizontal_flip = False,
                                 vertical_flip = False)


augmentation.fit(x_train)

# CNN Modeling
model = tensorflow.keras.Sequential([
    tensorflow.keras.layers.Conv2D(kernel_size=(3,3), filters=64, input_shape=(28,28,1), padding='same', activation='relu'),
    tensorflow.keras.layers.Conv2D(kernel_size=(3,3), filters=64, padding='same', activation='relu'),
    tensorflow.keras.layers.MaxPool2D(pool_size=(2,2)),
    tensorflow.keras.layers.Dropout(0.2),

    tensorflow.keras.layers.Conv2D(kernel_size=(3,3), filters=64, padding='same', activation='relu'),
    tensorflow.keras.layers.Conv2D(kernel_size=(3,3), filters=64, padding='same', activation='relu'),
    tensorflow.keras.layers.MaxPool2D(pool_size=(2,2)),
    tensorflow.keras.layers.Dropout(0.2),

    tensorflow.keras.layers.Flatten(),
    tensorflow.keras.layers.Dense(units=256, activation='relu'),
    tensorflow.keras.layers.Dense(units=64, activation='relu'),
    tensorflow.keras.layers.Dropout(0.2),
    tensorflow.keras.layers.Dense(units = 14, activation='softmax')
])
optimizer = tensorflow.keras.optimizers.RMSprop(lr = 0.001, rho = 0.9, epsilon = 1e-08, decay = 0.0 )
model.compile(optimizer = optimizer, 
              loss = "categorical_crossentropy", 
              metrics = ["accuracy"])


call_back = ReduceLROnPlateau(monitor = 'val_accuracy', patience = 3, verbose = 1, factor = 0.5, min_lr = 0.0001)
# monitor : 기준값, patience = 최적의 학습? verbose : 화면체크, factor : lr감소값, min_lr : 최소 lr값

print('학습을 시작합니다.')

history = model.fit_generator(augmentation.flow(x_train, y_train, batch_size=64),
                              epochs = 20, 
                              validation_data = (x_val, y_val),
                              verbose = 1,
                              steps_per_epoch=x_train.shape[0] // 64,  
                              callbacks=[call_back])

# Model Evaluation & Save
model.evaluate(x_val, y_val)
model.save('model.h5')
print(path,'에 모델이 저장되었습니다.')

