#keras:
from keras.models import Sequential
from keras import optimizers
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D

from plant_disease_data_process import split_data

# Parameters:
BATCH_SIZE = 16
EPOCHS = 8
#EPOCHS = 20
LEARNING_RATE = 0.001
SPECIES_CLASSES = 14 #The number of total plant species classes
DISEASE_CLASSES = 61 #The number of total dissease classes

#Defining deep_cnn (VGG) model
def deep_cnn():
    model = Sequential()
    
    model.add(Conv2D(input_shape = (X_data.shape[1], X_data.shape[2], X_data.shape[3]), 
                      filters = 64, kernel_size = (3,3), strides = (1,1), padding = 'same'))
    model.add(Activation('relu'))
    model.add(Conv2D(filters = 64, kernel_size = (3,3), strides = (1,1), padding = 'same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Conv2D(filters = 64, kernel_size = (3,3), strides = (1,1), padding = 'same'))
    model.add(Activation('relu'))
    model.add(Conv2D(filters = 64, kernel_size = (3,3), strides = (1,1), padding = 'same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Conv2D(filters = 64, kernel_size = (3,3), strides = (1,1), padding = 'same'))
    model.add(Activation('relu'))
    model.add(Conv2D(filters = 64, kernel_size = (3,3), strides = (1,1), padding = 'same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Flatten())
    model.add(Dense(200, activation = 'relu'))
    model.add(Dense(100, activation = 'relu'))
    model.add(Dense(20, activation = 'relu'))
    model.add(Dense(DISEASE_CLASSES, activation = 'softmax'))   
    adam = optimizers.Adam(lr = LEARNING_RATE)
    model.compile(loss = 'categorical_crossentropy', optimizer = adam, metrics = ['accuracy'])
    
    return model 


#Train the deep cnn model
def TrainModel_deepcnn(data=None, epochs=EPOCHS, batch=BATCH_SIZE):
    start_time = time.time()
    model = deep_cnn()
    
    x_train, x_test, y_train, y_test = split_data()
    print('Start training now...')
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch,
              validation_data=(x_test, y_test), verbose=1)
    print("Training totally took {0} seconds.".format(time.time() - start_time))
    return model, history
