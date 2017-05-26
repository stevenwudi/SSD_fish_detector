"""
This script is used to convert fish gt to the format of: [xmin, ymin, xmax, ymax, prob1, prob2, prob3, ...],
xmin, ymin, xmax, ymax are in relative coordinates.
Since fish is the only class with one-hot encoding
"""
from collections import defaultdict
import os
import json
import numpy as np
from keras.preprocessing import image
import cPickle as pickle

import matplotlib.pyplot as plt


def main(sloth_annotations_url=None):
    fish_folders = ['OTHER', 'ALB', 'BET', 'DOL', 'LAG', 'SHARK', 'YFT']
    NUM_CLASSES = len(fish_folders) + 1
    gt = defaultdict(list)

    # we also add NoF into this category:
    for nof_img in os.listdir('../train/NoF'):
        gt['../train/NoF/'+nof_img] = []

    def f_annotation(l, fld, fn):
        gt = {}
        count = 0
        for el in l:
            if 'filename' in el:
                key = el['filename']
                if key[:8] == '../data/':
                    img_path = key[:3] + key[8:]
                else:
                    img_path = '../train/'+fld.upper()+'/'+key

                if 'annotations' not in el or len(el['annotations']) < 1:
                    print("No annotation for " + fld + '/' + key)
                elif not os.path.isfile(img_path):
                    print("No image for: " + img_path)
                else:
                    gt[img_path] = []
                    img = image.load_img(img_path)
                    img = image.img_to_array(img)
                    for anno in el['annotations']:
                        count += 1
                        gt_annot = np.zeros(4+NUM_CLASSES-1)
                        anno['x'] = min(max(0, anno['x']), img.shape[1])
                        anno['y'] = min(max(0, anno['y']), img.shape[0])
                        # img_fish = img[int(anno['y']):int(anno['y'] + anno['height']),
                        #            int(anno['x']):int(anno['x'] + anno['width']), :]
                        ymin = anno['y'] / img.shape[0]
                        xmin = anno['x'] / img.shape[1]
                        ymax = (anno['y'] + anno['height']) / img.shape[0]
                        xmax = (anno['x'] + anno['width']) / img.shape[1]
                        gt_annot[:4] = [xmin, ymin, xmax, ymax]
                        gt_annot[fn+4] = 1
                        gt[img_path].append(gt_annot)
        print('Finish converting %s, total annotated fish number is %d in total image of %d.'%(fld, count, len(gt)))
        return gt

    if sloth_annotations_url is not None:
        for fn, fld in enumerate(fish_folders):
            json_path = os.path.join(sloth_annotations_url, fld.lower() + "_labels.json")
            sloth_annotations_list = json.load(open(json_path, 'r'))
            gt.update(f_annotation(sloth_annotations_list, fld, fn))

    with open('./data/gt.pkl', 'wb') as fp:
        pickle.dump(gt, fp)
    print("Finish loading images, total number of images is: " + str(len(gt)))


if __name__ == "__main__":
    sloth_annotations_url = '../Kaggle_Nature_conservancy/boundingbox_annotation'
    main(sloth_annotations_url)