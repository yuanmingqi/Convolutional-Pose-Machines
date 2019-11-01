import tensorflow as tf
import numpy as np
import time
import sys

class CPM:
    def __init__(self, 
                 inputs_width,
                 inputs_height,
                 inputs_channel,
                 keypoints_num,
                 stage_num,
                 padding,
                 warm_up_file,
                 batch_size,
                 repeat,
                 epochs,
                 lr,
                 lr_decay_rate,
                 save_path,
                 training_data_file):
        self.inputs_width = inputs_width
        self.inputs_height = inputs_height
        self.inputs_channel = inputs_channel
        self.keypoints_num = keypoints_num
        self.stage_num = stage_num
        self.padding = padding

        self.warm_up_file = warm_up_file
        self.training_data_file = training_data_file
        self.batch_size = batch_size
        self.repeat = repeat
        self.lr = lr
        self.lr_decay_rate = lr_decay_rate
        self.epochs = epochs
        self.save_path = save_path
        
    def stage_1(self, inputs):
        conv_1 = tf.keras.layers.Conv2D(filters=128, kernel_size=(9, 9), activation='relu', padding=self.padding)(inputs)
        pooling_1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding=self.padding)(conv_1)
        bn_1 = tf.keras.layers.BatchNormalization()(pooling_1)

        conv_2 = tf.keras.layers.Conv2D(filters=128, kernel_size=(9, 9), activation='relu', padding=self.padding)(bn_1)
        pooling_2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding=self.padding)(conv_2)
        bn_2 = tf.keras.layers.BatchNormalization()(pooling_2)

        conv_3 = tf.keras.layers.Conv2D(filters=256, kernel_size=(9, 9), activation='relu', padding=self.padding)(bn_2)
        pooling_3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding=self.padding)(conv_3)
        bn_3 = tf.keras.layers.BatchNormalization()(pooling_3)

        conv_4 = tf.keras.layers.Conv2D(filters=256, kernel_size=(5, 5), activation='relu', padding=self.padding)(bn_3)
        bn_4 = tf.keras.layers.BatchNormalization()(conv_4)

        conv_5 = tf.keras.layers.Conv2D(filters=256, kernel_size=(9, 9), activation='relu', padding=self.padding)(bn_4)
        bn_5 = tf.keras.layers.BatchNormalization()(conv_5)

        conv_6 = tf.keras.layers.Conv2D(filters=256, kernel_size=(1, 1), activation=None, padding=self.padding)(bn_5)
        bn_6 = tf.keras.layers.BatchNormalization()(conv_6)
        
        conv_7 = tf.keras.layers.Conv2D(filters=self.keypoints_num, kernel_size=(1, 1), activation=None, padding=self.padding)(bn_6)
        bn_7 = tf.keras.layers.BatchNormalization()(conv_7)
       
        score_map = bn_7

        return score_map

    def stage_t(self, feature_org, feature_before):
        inputs = tf.keras.layers.concatenate([feature_before, feature_org])

        conv_1 = tf.keras.layers.Conv2D(filters=64, kernel_size=(11, 11), activation='relu', padding=self.padding)(inputs)
        bn_1 = tf.keras.layers.BatchNormalization()(conv_1)

        conv_2 = tf.keras.layers.Conv2D(filters=64, kernel_size=(11, 11), activation='relu', padding=self.padding)(bn_1)
        bn_2 = tf.keras.layers.BatchNormalization()(conv_2)

        conv_3 = tf.keras.layers.Conv2D(filters=64, kernel_size=(11, 11), activation='relu', padding=self.padding)(bn_2)
        bn_3 = tf.keras.layers.BatchNormalization()(conv_3)

        conv_4 = tf.keras.layers.Conv2D(filters=64, kernel_size=(1, 1), activation=None, padding=self.padding)(bn_3)
        bn_4 = tf.keras.layers.BatchNormalization()(conv_4)

        conv_5 = tf.keras.layers.Conv2D(filters=self.keypoints_num, kernel_size=(1, 1), activation=None, padding=self.padding)(bn_4)
        bn_5 = tf.keras.layers.BatchNormalization()(conv_5)

        score_map = bn_5

        return score_map

    def original_feature_extractor(self, inputs):
        conv_1 = tf.keras.layers.Conv2D(filters=128, kernel_size=(9, 9), activation='relu', padding=self.padding)(inputs)
        pooling_1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding=self.padding)(conv_1)
        bn_1 = tf.keras.layers.BatchNormalization()(pooling_1)

        conv_2 = tf.keras.layers.Conv2D(filters=128, kernel_size=(9, 9), activation='relu', padding=self.padding)(bn_1)
        pooling_2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding=self.padding)(conv_2)
        bn_2 = tf.keras.layers.BatchNormalization()(pooling_2)

        conv_3 = tf.keras.layers.Conv2D(filters=256, kernel_size=(9, 9), activation='relu', padding=self.padding)(bn_2)
        pooling_3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding=self.padding)(conv_3)
        bn_3 = tf.keras.layers.BatchNormalization()(pooling_3)

        conv_4 = tf.keras.layers.Conv2D(filters=self.keypoints_num, kernel_size=(5, 5), activation=None, padding=self.padding)(bn_3)
        bn_4 = tf.keras.layers.BatchNormalization()(conv_4)
        
        return bn_4

    def forward_model(self, inputs):
        feature_org = self.original_feature_extractor(inputs)
        score_map = []

        if self.stage_num < 2:
            raise ValueError("The number of stages must be bigger than 1")
        else:
            for i in range(self.stage_num):
                if i == 0:
                    score_map.append(self.stage_1(inputs))
                else:
                    score_map.append(self.stage_t(feature_org=feature_org, feature_before=score_map[i-1]))

        return score_map

    def loss(self, y_true, y_pred):
        total_loss = 0
        for i in range(self.stage_num):
            total_loss = total_loss + tf.keras.losses.MSE(y_true=y_true, y_pred=y_pred[i])

        return total_loss

    def build_model(self):
        print("[INFO]: Building model!")
        start = time.clock()

        try:
            model = tf.keras.models.load_model(self.warm_up_file)
        except:
            inputs = tf.keras.layers.Input(shape=[self.inputs_height, self.inputs_width, self.inputs_channel])
            outputs = self.forward_model(inputs=inputs)
            model = tf.keras.Model(inputs=inputs, outputs=outputs)

        end = time.clock()
        t = end - start

        print("[INFO]: Model building accomplished!")
        print("[INFO]: Time cost %f" % (t))

        model.summary()

        return model

    def create_tfdata_datasets(self, npz_file_path):
        data = np.load(npz_file_path)
        x_train = data['x_train'] / 255.0
        y_train = data['y_train']
        x_test = data['x_test']
        y_test = data['y_test']

        x_train = x_train.astype('float32')

        datasets = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        datasets = datasets.repeat(self.repeat).batch(self.batch_size)

        return x_train.shape[0], datasets

    def train_op(self, model, images, labels):
        optimizer = tf.keras.optimizers.Adam(
            lr=self.lr,
            decay=self.lr_decay_rate)

        with tf.GradientTape() as tape:
            predictions = model(images)
            loss = self.loss(predictions=predictions, labels=labels)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            return tf.reduce_sum(loss)

    def train(self):
        model = self.build_model()
        samples_num, training_data = self.create_tfdata_datasets(npz_file_path=self.training_data_file)

        print("[INFO]: Train on %d samples, batch size %d, learning rate %f" % (samples_num, self.batch_size, self.lr))
        for i in range(self.epochs):
            j = 0
            print("[INFO]: Epoch: %d" % (i))
            t_start = time.clock()
            t_end = 0.0
            for images, labels in training_data:
                self.train_op(model=model, images=images, labels=labels)
                j = j + 1
                if j % 100 == 0:
                    t_end = time.clock()
                    t = t_end - t_start
                    loss = self.train_op(model=model, images=images, labels=labels)
                    print("[INFO]: Steps %d, MSE loss %f, Time %f s" % (j, loss, t))
                    try:
                        model.save(self.save_path + "snapshots-" + str(j) + '.h5')
                    except:
                        raise FileNotFoundError("Please recheck your save path, which should be ended with '/'.")

                t_start = t_end

        print("[INFO]: Training accomplished!")
    
    def keras_train(self):
        model = self.build_model()
        samples_num, training_data = self.create_tfdata_datasets_for_keras_train(npz_file_path=self.training_data_file)

        print("[INFO]: Train on %d samples, repeat %d, batch size %d, learning rate %f" % (samples_num, self.repeat, self.batch_size, self.lr))
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(lr=self.lr, decay=self.lr_decay_rate),
            loss=tf.keras.losses.MSE,
            metric=['mse']
            )
        
        model.fit(training_data, epochs=self.epochs)
        model.save('./snapshots/model.h5')
        print("[INFO]: Training accomplished!")

    def create_tfdata_datasets_for_keras_train(self, npz_file_path):
        data = np.load(npz_file_path)
        x_train = data['x_train'] / 255.0
        y_train = data['y_train']
        x_test = data['x_test']
        y_test = data['y_test']

        x_train = x_train.astype('float32')

        new_y_train = []
        for i in range(self.stage_num):
            new_y_train.append(y_train)
        
        new_y_train = tuple(new_y_train)

        datasets = tf.data.Dataset.from_tensor_slices((x_train, new_y_train))
        datasets = datasets.repeat(self.repeat).batch(self.batch_size)

        return x_train.shape[0], datasets