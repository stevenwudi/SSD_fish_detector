"""
This is a SSD512 fish detector
modify by DI WU: stevenwudi@gmail.com  2017-05-25
"""
import keras
import pickle
import random
import numpy as np
from SSD.ssd_512_v2 import SSD512v2
from SSD.ssd_utils import BBoxUtility
from SSD.ssd_training import Generator, MultiboxLoss


def gt_classification_convert(gt, NUM_CLASSES=8):
    for key in gt.keys():
        if len(gt[key]) != 0:
            anno_list = []
            for l in gt[key]:
                anno_list.append(l)
            gt[key] = np.asarray(anno_list)
        else:
            gt[key] = np.ndarray(shape=(0, 4 + NUM_CLASSES -1))

    return gt


def main():
    # random seed, it's very important for the experiment...
    random.seed(1000)
    NUM_CLASSES = 7 + 1
    ### load model ###
    input_shape = (512, 512, 3)
    model = SSD512v2(input_shape, num_classes=NUM_CLASSES)

    priors = pickle.load(open('./SSD/prior_boxes_ssd512.pkl', 'rb'))
    bbox_util = BBoxUtility(NUM_CLASSES, priors)

    model.summary()
    #model.load_weights('./SSD/weights_SSD300.hdf5', by_name=True)
    model.load_weights('./checkpoints_classification/weights_512.17-0.82.hdf5', by_name=True)
    gt = pickle.load(open('./data/gt.pkl', 'rb'))
    gt = gt_classification_convert(gt, NUM_CLASSES)

    keys = sorted(gt.keys())
    random.shuffle(keys)

    num_train = int(round(0.9 * len(keys)))
    train_keys = keys[:num_train]
    val_keys = keys[num_train:]

    gen = Generator(gt=gt, bbox_util=bbox_util, batch_size=1, path_prefix='',
                    train_keys=train_keys, val_keys=val_keys,
                    image_size=(input_shape[0], input_shape[1]), do_crop=False)

    freeze = ['input_1', 'conv1_1', 'conv1_2', 'pool1',
              'conv2_1', 'conv2_2', 'pool2']
              #    'conv3_1', 'conv3_2', 'conv3_3', 'pool3']  # ,
    #           'conv4_1', 'conv4_2', 'conv4_3', 'pool4']

    for L in model.layers:
        if L.name in freeze:
            L.trainable = False

    def schedule(epoch, decay=0.9):
        return base_lr * decay ** (epoch)

    callbacks = [keras.callbacks.ModelCheckpoint('./checkpoints_classification/weights_512.{epoch:02d}-{val_loss:.2f}.hdf5',
                                                 verbose=1,
                                                 save_weights_only=True),
                 keras.callbacks.LearningRateScheduler(schedule)]

    base_lr = 1e-5
    optim = keras.optimizers.Adam(lr=base_lr)
    model.compile(optimizer=optim,
                  loss=MultiboxLoss(NUM_CLASSES, neg_pos_ratio=2.0).compute_loss)

    nb_epoch = 50
    history = model.fit_generator(generator=gen.generate(True),
                                  steps_per_epoch=gen.train_batches,
                                  epochs=nb_epoch, verbose=1,
                                  callbacks=callbacks,
                                  validation_data=gen.generate(False),
                                  validation_steps=gen.val_batches,
                                  workers=1)


if __name__ == "__main__":
    main()
