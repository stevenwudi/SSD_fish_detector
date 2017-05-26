"""
This is a SSD fish detector
"""
import keras
from keras.backend.tensorflow_backend import set_session
from keras.applications.imagenet_utils import preprocess_input

from keras.models import Model
from keras.preprocessing import image
import matplotlib.pyplot as plt
import pickle
import random
import numpy as np
import tensorflow as tf
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

    ### load model ###
    model = SSD300v2(input_shape, num_classes=NUM_CLASSES)
    model.load_weights('./checkpoints/weights.23-0.85.hdf5', by_name=True)

    inputs = []
    images = []
    add_num = 250
    gt_result = []
    #for i in range(num_val):
    for i in range(20):
        img_path = val_keys[i+add_num]
        #img_path = train_keys[i+add_num]
        gt_result.append(gt[val_keys[i+add_num]])
        img = image.load_img(img_path, target_size=(300, 300))
        img = image.img_to_array(img)
        images.append(img)
        inputs.append(img.copy())

    inputs = preprocess_input(np.array(inputs))
    preds = model.predict(inputs, batch_size=1, verbose=1)
    results = bbox_util.detection_out(preds)

    for i, img in enumerate(images):
        # Parse the outputs.
        det_label = results[i][:, 0]
        det_conf = results[i][:, 1]
        det_xmin = results[i][:, 2]
        det_ymin = results[i][:, 3]
        det_xmax = results[i][:, 4]
        det_ymax = results[i][:, 5]

        # Get detections with confidence higher than 0.6.
        #top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.2]
        top_indices = [0]
        top_conf = det_conf[top_indices]
        top_label_indices = det_label[top_indices].tolist()
        top_xmin = det_xmin[top_indices]
        top_ymin = det_ymin[top_indices]
        top_xmax = det_xmax[top_indices]
        top_ymax = det_ymax[top_indices]

        colors = plt.cm.hsv(np.linspace(0, 1, 4)).tolist()
        currentAxis = plt.gca()
        currentAxis.cla()
        plt.imshow(img / 255.)

        gt_img = gt_result[i]
        if len(gt_img):
            gt_top_xmin = gt_img[0][0]
            gt_top_ymin = gt_img[0][1]
            gt_top_xmax = gt_img[0][2]
            gt_top_ymax = gt_img[0][3]

        for j in range(top_conf.shape[0]):
            xmin = int(round(top_xmin[j] * img.shape[1]))
            ymin = int(round(top_ymin[j] * img.shape[0]))
            xmax = int(round(top_xmax[j] * img.shape[1]))
            ymax = int(round(top_ymax[j] * img.shape[0]))
            score = top_conf[j]
            label = int(top_label_indices[j])
            #label_name = voc_classes[label - 1]
            display_txt = '{:0.2f}'.format(score)
            coords = (xmin, ymin), xmax - xmin + 1, ymax - ymin + 1
            color = colors[label]
            currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
            currentAxis.text(xmin, ymin, display_txt, bbox={'facecolor': color, 'alpha': 0.5})

        # plt GT
        if len(gt_img):
            xmin = int(round(gt_top_xmin * img.shape[1]))
            ymin = int(round(gt_top_ymin * img.shape[0]))
            xmax = int(round(gt_top_xmax * img.shape[1]))
            ymax = int(round(gt_top_ymax * img.shape[0]))
            coords = (xmin, ymin), xmax - xmin + 1, ymax - ymin + 1
            color = colors[3]
            currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))

        plt.draw()
        plt.waitforbuttonpress(3)


if __name__ == "__main__":
    main()


