import os
import numpy as np
from ultralytics.utils.ops import xywhr2xyxyxyxy

DATA_PATH = "./labels_path"
OUTPUT_DIR = "./labels_obb_path"
os.makedirs(OUTPUT_DIR, exist_ok=True)
r = 0
for label_txt in os.listdir(DATA_PATH):
    label_file = open(DATA_PATH + '/' + label_txt, 'r')
    output_file = open(OUTPUT_DIR + '/' + label_txt, 'w')
    for bbox in label_file.readlines():
        if (len(bbox.split()) == 5):
            c, x, y, w, h = [float(i) for i in bbox.split()]
            output_file.write(f"{int(c)}")
            xyxy = xywhr2xyxyxyxy(np.array([x, y, w, h, r]))
            for x, y in xyxy:
                output_file.write(f" {x} {y}")
            output_file.write('\n')
    label_file.close()
    output_file.close()