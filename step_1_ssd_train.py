"""
This is a SSD fish detector
"""
import keras
import pickle
import random
import numpy as np
from SSD.ssd_v2 import SSD300v2
from SSD.ssd_training import MultiboxLoss, Generator
from SSD.ssd_utils import BBoxUtility


def gt_binary_convert(gt):
    for key in gt.keys():
        if len(gt[key]) != 0:
            anno_list = []
            for l in gt[key]:
                gt_annot = np.ones(4 + 1)
                gt_annot[:4] = l[:4]
                anno_list.append(gt_annot)
            gt[key] = np.asarray(anno_list)
        else:
            gt[key] = np.ndarray(shape=(0, 5))

    return gt


def main():
    # random seed, it's very important for the experiment...
    random.seed(1000)
    NUM_CLASSES = 2
    input_shape = (300, 300, 3)
    priors = pickle.load(open('./SSD/prior_boxes_ssd300.pkl', 'rb'))
    bbox_util = BBoxUtility(NUM_CLASSES, priors)

    gt = pickle.load(open('./data/gt.pkl', 'rb'))
    gt = gt_binary_convert(gt)
    keys = sorted(gt.keys())
    random.shuffle(keys)

    num_train = int(round(0.9 * len(keys)))
    train_keys = keys[:num_train]
    val_keys = keys[num_train:]
    num_val = len(val_keys)

    gen = Generator(gt=gt, bbox_util=bbox_util, batch_size=1, path_prefix='',
                    train_keys=train_keys, val_keys=val_keys,
                    image_size=(input_shape[0], input_shape[1]), do_crop=False)

    model = SSD300v2(input_shape, num_classes=NUM_CLASSES)
    model.load_weights('./SSD/weights_SSD300.hdf5', by_name=True)

    freeze = ['input_1', 'conv1_1', 'conv1_2', 'pool1',
              'conv2_1', 'conv2_2', 'pool2',
              'conv3_1', 'conv3_2', 'conv3_3', 'pool3']  # ,
    #           'conv4_1', 'conv4_2', 'conv4_3', 'pool4']

    for L in model.layers:
        if L.name in freeze:
            L.trainable = False

    def schedule(epoch, decay=0.9):
        return base_lr * decay**(epoch)

    callbacks = [keras.callbacks.ModelCheckpoint('./checkpoints/weights.{epoch:02d}-{val_loss:.2f}.hdf5',
                                                 verbose=1,
                                                 save_weights_only=True),
                 keras.callbacks.LearningRateScheduler(schedule)]

    base_lr = 1e-4
    optim = keras.optimizers.Adam(lr=base_lr)
    model.compile(optimizer=optim,
                  loss=MultiboxLoss(NUM_CLASSES, neg_pos_ratio=2.0).compute_loss)

    nb_epoch = 30
    history = model.fit_generator(generator=gen.generate(True),
                                  steps_per_epoch=gen.train_batches,
                                  epochs=nb_epoch, verbose=1,
                                  callbacks=callbacks,
                                  validation_data=gen.generate(False),
                                  validation_steps=gen.val_batches,
                                  workers=1)

if __name__ == "__main__":
    main()


# Epoch 19/20
# 3011/3012 [============================>.] - ETA: 0s - loss: 1.2628Epoch 00018: saving model to ./checkpoints/weights.18-1.91.hdf5
# 3012/3012 [==============================] - 518s - loss: 1.2625 - val_loss: 1.9066
# Epoch 20/20
# 3011/3012 [============================>.] - ETA: 0s - loss: 1.2170Epoch 00019: saving model to ./checkpoints/weights.19-1.87.hdf5
# 3012/3012 [==============================] - 518s - loss: 1.2170 - val_loss: 1.8701
