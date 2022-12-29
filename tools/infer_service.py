#!/usr/bin/env python3
# -*- coding:utf-8 -*-

# import time
# from infer import run
import os

ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from yolov6.core.inferer import Inferer

from flask import Flask, jsonify
# from cStringIO import StringIO
# import cv2
# import numpy as np

app = Flask(__name__)

@app.route('/inference', methods = ['POST'])
def upload_file():
    data = request.get_data()
    with open("image.jpg", "wb") as f:
        f.write(data)
    # img_array = np.asarray(data, dtype=np.uint8)
    # imgdecoded = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    inferer = Inferer('/home/YOLOv6/image.jpg', '/home/YOLOv6/yolov6n_base.pt', '0', 'data/coco.yaml', [640,480], False)
    inferer.infer(0.4, 0.45, [0], False, 1000, '/home/YOLOv6/runs/inference/exp', True, True, False, False, False)

    with open("/home/YOLOv6/runs/inference/exp/image.txt", "r") as f:
        ret = f.readlines()

    return jsonify(ret)

if __name__ == '__main__':
    app.run(debug = True)

# inferer = Inferer(source, weights, device, yaml, img_size, half)
# inferer.infer(conf_thres, iou_thres, classes, agnostic_nms, max_det, save_dir, save_txt, not not_save_img, hide_labels, hide_conf, view_img)

