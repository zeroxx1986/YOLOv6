#!/usr/bin/env python3
# -*- coding:utf-8 -*-

# import time
# from infer import run
import os
import sys
import uuid

ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from yolov6.core.inferer import Inferer

from flask import Flask, jsonify, request
# from cStringIO import StringIO
# import cv2
# import numpy as np

app = Flask(__name__)

@app.route('/inference', methods = ['POST'])
def upload_file():
    data = request.get_data()
    name = uuid.uuid4()
    with open(f"{name}.jpg", "wb") as f:
        f.write(data)
    # img_array = np.asarray(data, dtype=np.uint8)
    # imgdecoded = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    inferer = Inferer(f'/home/YOLOv6/{name}.jpg', '/home/YOLOv6/yolov6n_base.pt', None, 'data/coco.yaml', [640,480], False)
    inferer.infer(0.4, 0.45, [0], False, 1000, '/home/YOLOv6/runs/inference/exp', True, True, False, False, False)

    if os.path.exists(f'/home/YOLOv6/runs/inference/exp/{name}.txt'):
        with open(f'/home/YOLOv6/runs/inference/exp/{name}.txt', 'r') as f:
            ret = f.readlines()
        os.remove(f'/home/YOLOv6/runs/inference/exp/{name}.txt')
    else:
        ret = []
    os.remove(f'/home/YOLOv6/{name}.jpg')
    
    return jsonify(ret)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug = True)

# inferer = Inferer(source, weights, device, yaml, img_size, half)
# inferer.infer(conf_thres, iou_thres, classes, agnostic_nms, max_det, save_dir, save_txt, not not_save_img, hide_labels, hide_conf, view_img)

