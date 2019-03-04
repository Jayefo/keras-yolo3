import os
import numpy as np
import keras.backend as K
from keras.layers import Input, Lambda
from keras.models import Model
from yolo3.model import preprocess_true_boxes, yolo_body, yolo_loss
from yolo3.utils import get_random_data
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

def getClassNameList(classFilePath):
    with open(classFilePath) as file:
        className_list = [k.strip() for k in file.readlines() if k.strip() != '']
    return className_list


def getAnchorList(anchorFilePath):
    with open(anchorFilePath) as file:
        anchor_list = [float(k) for k in file.read().split(',')]
    return np.array(anchor_list).reshape(-1, 2)


def main():
    classeFilePath = 'model_data/voc_classes.txt'
    anchorFilePath = 'model_data/yolo_anchors.txt'
    className_list = getClassNameList(classeFilePath)
    anchor_list = getAnchorList(anchorFilePath)
    input_shape = (416,416) # multiple of 32, height and width
    model = create_model(input_shape, anchor_list, len(className_list))
    annotationFilePath = 'dataset_train.txt'
    train(model, annotationFilePath, input_shape, anchor_list, len(className_list))


def create_model(input_shape,
                 anchor_list,
                 num_classes,
                 load_pretrained=True,
                 freeze_body=False,
                 weights_path='saved_model/trained_weights.h5'):
    K.clear_session() # get a new session
    image_input = Input(shape=(None, None, 3))
    height, width = input_shape
    num_anchors = len(anchor_list)
    y_true = [Input(shape=(height // k,
                           width // k,
                           num_anchors // 3,
                           num_classes + 5)) for k in [32, 16, 8]]
    model_body = yolo_body(image_input, num_anchors//3, num_classes)
    print('Create YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    if load_pretrained and os.path.exists(weights_path):
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))
        if freeze_body:
            num = len(model_body.layers)-7
            for i in range(num):
                model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

    model_loss = Lambda(yolo_loss,
                        output_shape=(1,),
                        name='yolo_loss',
                        arguments={'anchors': anchor_list,
                                   'num_classes': num_classes,
                                   'ignore_thresh': 0.5})(
                                        [*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)
    return model


def train(model,
          annotationFilePath,
          input_shape,
          anchor_list,
          num_classes,
          logDirPath='saved_model/'):
    model.compile(optimizer='adam',
                  loss={'yolo_loss': lambda y_true, y_pred: y_pred})
    batch_size = 2 * num_classes
    val_split = 0.05
    with open(annotationFilePath) as file:
        lines = file.readlines()
    np.random.shuffle(lines)
    num_val = int(len(lines)*val_split)
    num_train = len(lines) - num_val
    print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
    if not os.path.isdir(logDirPath):
        os.makedirs(logDirPath)
    logging = TensorBoard(log_dir=logDirPath)
    checkpoint = ModelCheckpoint(logDirPath + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
        monitor='val_loss', save_weights_only=True, save_best_only=True, period=3)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)
    model.fit_generator(
        data_generator(lines[:num_train], batch_size, input_shape, anchor_list, num_classes),
        steps_per_epoch=max(1, num_train // batch_size),
        validation_data=data_generator(lines[num_train:], batch_size, input_shape, anchor_list, num_classes),
        validation_steps=max(1, num_val // batch_size),
        epochs=200,
        initial_epoch=0,
        callbacks=[logging, checkpoint, reduce_lr, early_stopping])
    # when model training finished, save model
    
    model_savedPath = 'saved_model/trained_weights.h5'
    model.save_weights(model_savedPath)


def data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes):
    n = len(annotation_lines)
    np.random.shuffle(annotation_lines)
    i = 0
    while True:
        image_data = []
        box_data = []
        for b in range(batch_size):
            i %= n
            image, box = get_random_data(annotation_lines[i], input_shape, random=True)
            image_data.append(image)
            box_data.append(box)
            i += 1
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
        yield [image_data, *y_true], np.zeros(batch_size)


if __name__ == '__main__':
    main()
