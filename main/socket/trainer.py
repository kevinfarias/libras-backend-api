import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow.keras.utils import to_categorical, plot_model 
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras import backend
from sklearn.metrics import classification_report

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
import sys
from cnn import Convolucao
from utils import DateUtils
import time

class Trainer:
    def __init__(self, fileName, EPOCHS = 30, CLASS = 21):
        self.fileName = fileName
        self.EPOCHS = EPOCHS
        self.CLASS = CLASS
    
    def run(self, imageWidth = 64, imageHeight = 64):
        fileDate = DateUtils.getNowInStr()
        fileName = self.fileName+fileDate

        EPOCHS = self.EPOCHS
        CLASS = self.CLASS

        print('[INFO] [INICIO]: ' + DateUtils.getNowInStr() + '\n')

        print('[INFO] Download dataset usando tensorflow.keras.preprocessing.image.ImageDataGenerator')

        train_datagen = ImageDataGenerator(
                rescale=1./255,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True, 
                validation_split=0.25)

        test_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.25)

        training_set = train_datagen.flow_from_directory(
                '../../dataset/training',
                target_size=(imageWidth, imageHeight),
                color_mode = 'rgb',
                batch_size=32,
                shuffle=False,
                class_mode='categorical')

        test_set = test_datagen.flow_from_directory(
                '../../dataset/test',
                target_size=(imageWidth, imageHeight),
                color_mode = 'rgb',
                batch_size=32,
                shuffle=False,
                class_mode='categorical')

        # inicializar e otimizar modelo
        print("[INFO] Inicializando e otimizando a CNN...")
        start = time.time()

        early_stopping_monitor = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15)

        model = Convolucao.build(imageWidth, imageHeight, 3, CLASS)
        model.compile(optimizer=SGD(0.01), loss="categorical_crossentropy",
                      metrics=["accuracy"])

        # treinar a CNN
        print("[INFO] Treinando a CNN...")
        classifier = model.fit_generator(
                training_set,
                steps_per_epoch=(training_set.n // training_set.batch_size),
                epochs=EPOCHS,
                validation_data = test_set,
                validation_steps= (test_set.n // test_set.batch_size),
                shuffle = False,
                verbose=2,
                callbacks = [early_stopping_monitor]
            )

        # atualizo valor da epoca caso o treinamento tenha finalizado antes do valor de epoca que foi iniciado
        EPOCHS = len(classifier.history["loss"])

        print("[INFO] Salvando modelo treinado ...")

        #para todos arquivos ficarem com a mesma data e hora. Armazeno na variavel
        model.save(f'./models/{fileName}.h5')
        print('[INFO] modelo: ./models/{fileName}.h5 salvo!')

        end = time.time()

        print("[INFO] Tempo de execução da CNN: %.1f min" %(DateUtils.getTimeDiffInMin(start,end)))

        print('[INFO] Summary: ')
        model.summary()

        print("\n[INFO] Avaliando a CNN...")
        score = model.evaluate_generator(generator=test_set, steps=(test_set.n // test_set.batch_size), verbose=1)
        print('[INFO] Accuracy: %.2f%%' % (score[1]*100), '| Loss: %.5f' % (score[0]))

        print("[INFO] Sumarizando loss e accuracy para os datasets 'train' e 'test'")

        plt.style.use("ggplot")
        plt.figure()
        plt.plot(np.arange(0,EPOCHS), classifier.history["loss"], label="train_loss")
        plt.plot(np.arange(0,EPOCHS), classifier.history["val_loss"], label="val_loss")
        plt.plot(np.arange(0,EPOCHS), classifier.history["acc"], label="train_acc")
        plt.plot(np.arange(0,EPOCHS), classifier.history["val_acc"], label="val_acc")
        plt.title("Training Loss and Accuracy")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Accuracy")
        plt.legend()
        plt.savefig(f'./models/graphics/{fileName}.png', bbox_inches='tight')

        print('[INFO] Gerando imagem do modelo de camadas da CNN')
        plot_model(model, to_file=f'./models/image/{fileName}.png', show_shapes = True)

        print('\n[INFO] [FIM]: ' + DateUtils.getNowInStr())
        print('\n\n')

if sys.stdin:
    trainer = Trainer('new_cnn_model')
    trainer.run(256, 256)