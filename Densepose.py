# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
##import argparse
from glob import glob
##import logging
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import sys
import torch
import numpy as np
import cv2
##import tqdm

sys.path.insert(0,'./detectron2/projects/DensePose')

from detectron2.config import get_cfg
from detectron2.engine.defaults import DefaultPredictor

##from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

from densepose import add_densepose_config
# from densepose.data.structures import DensePoseResult
from densepose.structures import quantize_densepose_chart_result
#from densepose.vis.densepose import DensePoseResultsVisualizer
from densepose.vis.densepose_results import DensePoseResultsVisualizer
from densepose.vis.extractor import CompoundExtractor, create_extractor

def setup_config(config_fpath: str, model_fpath: str, confidence_threshold):
    cfg = get_cfg()
    add_densepose_config(cfg)
    cfg.merge_from_file(config_fpath)
    cfg.MODEL.WEIGHTS = model_fpath

    ##args = get_parser().parse_args()
    ##confidence_threshold=0.75
    ##cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = confidence_threshold
    ##cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
    ##cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = confidence_threshold

    cfg.freeze()
    return cfg

cfg_file = './models/densepose_rcnn_R_50_FPN_DL_WC1_s1x.yaml'
model_file = './models/model_final_b1e525.pkl'

def eraseShape(img, shape): ##Elimina silueta
    return cv2.add(img, whitenShape(shape))


def whitenShape(imgShape): ##Blanquea la silueta
    return np.uint8(imgShape > 0) * 255
    ##return cv2.cvtColor(np.uint8(imgShape > 0) * 255, cv2.COLOR_BGR2GRAY)

def apply_mask_to_img(mask, img): ##Aplica la mascara con un factor de transparencia del 0.5 en las dos imagenes
    return cv2.addWeighted(img, 0.5, mask, 0.5, 0.0)

##def color_shape_and_rectangle(imgshapes, shape, nShape, coord):
def color_shape_and_rectangle(imgshapes, shape, coord):
    color = list(np.random.random(size=3) * 256) ##Genera un color aleatorio
    
    ##printNumber(imgshapes, coord[0], coord[1], coord[2], coord[3], nShape, color) ##Imprime un numero en el centro de cada figura
    cv2.rectangle(imgshapes, (coord[0],coord[1]), (coord[2],coord[3]), color, 2) ##Dibuja un rectángulo alrededor de cada figura

    shape[np.where((shape==[255, 255, 255]).all(axis=2))] = color ## Colorea la silueta con el color generado
    imgshapes = cv2.addWeighted(imgshapes, 1, shape, 1, 0.0) ##Anyade la silueta coloreada a la imagen resultante
    
    return imgshapes, shape

""" def printNumber(img, x, y, w, h, num, color):
    
    mediox = (x+w)/2
    medioy = (y+h)/2

    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (int(mediox),int(medioy))
    fontScale = 1
    fontColor = color
    lineType = 2

    cv2.putText(img, str(int(num)), bottomLeftCornerOfText, font, fontScale, fontColor, lineType) """



def calculate_mask_and_shapes(img, confidence_threshold):


    cfg = setup_config(cfg_file, model_file, confidence_threshold)
    predictor = DefaultPredictor(cfg)

    with torch.no_grad():
        
        ##img = dataset

        visualizer = DensePoseResultsVisualizer()
        extractor = create_extractor(visualizer)

        outputs = predictor(img)

        ##Colorea Automatico
        """ output = Visualizer(img[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
        out = output.draw_instance_predictions(outputs['instances'].to("cpu"))
        ##cv2.imshow('imgVisualizer', out.get_image()[:, :, ::-1]) """


        instances = outputs['instances']
        scores = instances._fields['scores']
        data = extractor(instances)

        uv_results, bboxes = data
        results = [(scores[s], uv_results[s], bboxes[s]) for s in range(len(scores))]
        results = sorted(results, key=lambda r: r[0])

        shapes = []
        colorMask = np.zeros_like(img)
        for result in results: ##Posibles resultados
            score, res, bbox = result
            if score > confidence_threshold: ##Filtra posibles resultados

                iuv_img_single = np.zeros_like(img) ##Inicializa imagen resultado una figura
                qres = quantize_densepose_chart_result(res)
                iuv_arr = qres.labels_uv_uint8.cpu()
                x, y, w, h = map(int, bbox)
                for c in range(3):
                    iuv_img_single[y:y+h,x:x+w,c] = iuv_arr[c, :, :]
                
                
                ##colorMask, colorShape = color_shape_and_rectangle(colorMask, whitenShape(iuv_img_single), nShape, [x,y,x+w,y+h])
                colorMask, colorShape = color_shape_and_rectangle(colorMask, whitenShape(iuv_img_single), [x,y,x+w,y+h])
                ##colorShapes.append(colorShape)
                shapes.append(whitenShape(iuv_img_single))

        
        return apply_mask_to_img(colorMask, img), shapes##, coordinates

def white_pixels(shape):
    ##return cv2.countNonZero(cv2.cvtColor((np.uint8(shape > 0) * 255),cv2.COLOR_BGR2GRAY))
    ##(np.uint8(shape > 0) * 255)
    return cv2.countNonZero(cv2.cvtColor(whitenShape(shape),cv2.COLOR_BGR2GRAY))

def order_denseposes(denseshapes):
    
    denseshapes.sort(key=white_pixels, reverse=True)
    ##ordered_denseposes = sorted(denseshapes, key=white_pixels)
    
    return denseshapes

def densePose(dataset_dir, confidence_threshold):
    ##dataset_dir = cv2.imread("images/mayor.jpg")
    
    ## Obtenemos la mascara, las siluetas y sus coordenadas
    ##mask, shapes, coordinates = calculate_mask_shapes_and_coordinates(dataset_dir, confidence_threshold)
    mask, shapes = calculate_mask_and_shapes(dataset_dir, confidence_threshold)

    ## Aplicamos la mascara a la imagen 
    ##imgMask = apply_mask_to_img(mask, dataset_dir)

    return mask, order_denseposes(shapes)