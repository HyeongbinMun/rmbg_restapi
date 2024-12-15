# -*- coding: utf-8 -*-
import cv2
from AnalysisEngine.config import DEBUG
from AnalysisEngine.celerys import app
from celery.signals import worker_init, worker_process_init
from billiard import current_process
from utils import Logging

@worker_init.connect
def model_load_info(**__):
    print(Logging.i("===================="))
    print(Logging.s("Worker Analyzer Initialize"))
    print(Logging.s("===================="))

@worker_process_init.connect
def module_load_init(**__):
    global analyzer

    if not DEBUG:
        worker_index = current_process().index
        print(Logging.i("====================\n"))
        print(Logging.s("Worker Id: {0}".format(worker_index)))
        print(Logging.s("===================="))

    # TODO:
    #   - Add your model
    #   - You can use worker_index if you need to get and set gpu_id
    #       - ex) gpu_id = worker_index % TOTAL_GPU_NUMBER
    # from model.yolov7.main import YOLOv7
    from model.RMBG_2_0.main import RMBG2
    analyzer = RMBG2()


@app.task(acks_late=True, queue='WebAnalyzer', routing_key='webanalyzer_tasks')
def analyzer_by_image(file_path, mask_blur=0, mask_offset=0, invert_output=False):
    image = cv2.imread(file_path)
    parameters = {
        'mask_blur': mask_blur,
        'mask_offset': mask_offset,
        'invert_output': invert_output
    }
    out_images = analyzer.inference(image, parameters)
    return out_images