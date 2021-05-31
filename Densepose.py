# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from glob import glob
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import sys
import torch
import numpy as np
import cv2

sys.path.insert(0,'./detectron2/projects/DensePose')

from detectron2.config import get_cfg
from detectron2.engine.defaults import DefaultPredictor

from detectron2.data import MetadataCatalog, DatasetCatalog

from densepose import add_densepose_config
from densepose.structures import quantize_densepose_chart_result
from densepose.vis.densepose_results import DensePoseResultsVisualizer
from densepose.vis.extractor import CompoundExtractor, create_extractor

def setup_config(config_fpath: str, model_fpath: str, confidence_threshold):
    cfg = get_cfg()
    add_densepose_config(cfg)
    cfg.merge_from_file(config_fpath)
    cfg.MODEL.WEIGHTS = model_fpath
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = confidence_threshold

    cfg.freeze()
    return cfg

cfg_file = './models/densepose_rcnn_R_50_FPN_DL_WC1_s1x.yaml'
model_file = './models/model_final_b1e525.pkl'


def whitenShape(imgShape): ##Blanquea la silueta
    return np.uint8(imgShape > 0) * 255

def apply_mask_to_img(mask, img): ##Aplica la mascara con un factor de transparencia del 0.5 en las dos imagenes
    return cv2.addWeighted(img, 0.5, mask, 0.5, 0.0)

##def color_shape_and_rectangle(imgshapes, shape, nShape, coord):
def color_shapes_and_rectangle(imgshapes, shape, coord):
    color = list(np.random.random(size=3) * 256) ##Genera un color aleatorio
    
    cv2.rectangle(imgshapes, (coord[0],coord[1]), (coord[2],coord[3]), color, 2) ##Dibuja un rectángulo alrededor de cada figura

    shape[np.where((shape==[255, 255, 255]).all(axis=2))] = color ## Colorea la silueta con el color generado
    imgshapes = cv2.addWeighted(imgshapes, 1, shape, 1, 0.0) ##Anyade la silueta coloreada a la imagen resultante
    
    return imgshapes

def white_pixels(shape):
    return cv2.countNonZero(cv2.cvtColor(shape,cv2.COLOR_BGR2GRAY))

def order_denseposes(denseshapes):
    
    denseshapes.sort(key=white_pixels, reverse=True)
    
    return denseshapes


def densePose(img, confidence_threshold):


    ##Generamos la configuración del detector
    cfg = setup_config(cfg_file, model_file, confidence_threshold)

    ##Creamos el detector
    predictor = DefaultPredictor(cfg)

    with torch.no_grad():
        

        ##Declaramos el visualizador DensePose
        visualizer = DensePoseResultsVisualizer()

        ##Generamos el extractor mediante el visualizador
        extractor = create_extractor(visualizer)

        ##Almacenamos los resultados del detector
        outputs = predictor(img)

        ##Extraemos las instancias
        instances = outputs['instances']

        ##Almacenamos las puntuaciones
        scores = instances._fields['scores']

        ##Aplicamos el extractor de DensePose a las instancias
        data = extractor(instances)

        ##Generamos unos resultados con la puntuación, los resultados DensePose y las bounding boxes
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
                
                
                whiteshape = whitenShape(iuv_img_single) ##Blanqueamos la máscara
                colorMask = color_shapes_and_rectangle(colorMask, whiteshape, [x,y,x+w,y+h])##Anyadimos la máscara con un color aleatorio a la máscara general

                shapes.append(whiteshape)##Almacenamos la máscara en la lista de máscaras individuales

        
        return apply_mask_to_img(colorMask, img), order_denseposes(shapes)