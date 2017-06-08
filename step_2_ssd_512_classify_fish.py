"""
This is a SSD fish detector
"""

from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image
import matplotlib.pyplot as plt
import pickle
import random
import os
import numpy as np
from SSD.ssd_512_v2 import SSD512v2
from SSD.ssd_utils import BBoxUtility


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
    fish_names = ['OTHER', 'ALB', 'BET', 'DOL', 'LAG', 'SHARK', 'YFT']

    input_shape = (512, 512, 3)
    priors = pickle.load(open('./SSD/prior_boxes_ssd512.pkl', 'rb'))
    bbox_util = BBoxUtility(NUM_CLASSES, priors)

    gt = pickle.load(open('./data/gt.pkl', 'rb'))
    gt = gt_classification_convert(gt)
    keys = sorted(gt.keys())
    random.shuffle(keys)

    num_train = int(round(0.9 * len(keys)))
    val_keys = keys[num_train:]

    ### load model ###
    model = SSD512v2(input_shape, num_classes=NUM_CLASSES)
    model.load_weights('./checkpoints_classification/weights_512.17-0.82.hdf5', by_name=True)

    inputs = []
    images = []
    add_num = 40+19
    gt_result = []
    #for i in range(num_val):
    for i in range(20):
        img_path = val_keys[i + add_num]
        if os.path.isfile(img_path):
            gt_result.append(gt[val_keys[i+add_num]])
            img = image.load_img(img_path, target_size=(512, 512))
            img = image.img_to_array(img)
            images.append(img)
            inputs.append(img.copy())

    inputs = preprocess_input(np.array(inputs))
    preds = model.predict(inputs, batch_size=1, verbose=1)
    results = bbox_util.detection_out(preds)

    for i, img in enumerate(images):
        currentAxis = plt.gca()
        currentAxis.cla()
        plt.imshow(img / 255.)
        # Parse the outputs.
        if len(results[i]):
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
            for j in range(top_conf.shape[0]):
                xmin = int(round(top_xmin[j] * img.shape[1]))
                ymin = int(round(top_ymin[j] * img.shape[0]))
                xmax = int(round(top_xmax[j] * img.shape[1]))
                ymax = int(round(top_ymax[j] * img.shape[0]))
                score = top_conf[j]
                label = int(top_label_indices[j])
                label_name = fish_names[label - 1]
                display_txt = '{:0.2f}, {}'.format(score, label_name)
                coords = (xmin, ymin), xmax - xmin + 1, ymax - ymin + 1
                color = 'g'
                currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
                currentAxis.text(xmin, ymin, display_txt, bbox={'facecolor': color, 'alpha': 0.5})

        # plt GT
        gt_img = gt_result[i]
        if len(gt_img):
            gt_top_xmin = gt_img[0][0]
            gt_top_ymin = gt_img[0][1]
            gt_top_xmax = gt_img[0][2]
            gt_top_ymax = gt_img[0][3]
        if len(gt_img):
            xmin = int(round(gt_top_xmin * img.shape[1]))
            ymin = int(round(gt_top_ymin * img.shape[0]))
            xmax = int(round(gt_top_xmax * img.shape[1]))
            ymax = int(round(gt_top_ymax * img.shape[0]))
            coords = (xmin, ymin), xmax - xmin + 1, ymax - ymin + 1
            color = 'r'
            ## gt label
            label = int(np.argmax(gt_img[0][4:]))
            label_name = fish_names[label]
            currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
            currentAxis.text(xmin, ymax, label_name, bbox={'facecolor': color, 'alpha': 0.5})

        plt.draw()
        plt.waitforbuttonpress(3)


if __name__ == "__main__":
    main()


