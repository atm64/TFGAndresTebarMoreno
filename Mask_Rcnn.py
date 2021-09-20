import glob
import multiprocessing as mp
import os
import cv2
import numpy as np
from detectron2.config import get_cfg
from detectron2.engine.defaults import DefaultPredictor

from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog


def setup_cfg(confidence_threshold):##(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    
    ##Establecemos la congiuracion del modelo a utilizar
    cfg.merge_from_file("detectron2/configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml")
    cfg.merge_from_list([])

    # Asignamos el umbral de confianza
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = confidence_threshold
    cfg.freeze()
    return cfg

def resize_Rcnn_Img_Show(output, human_instances, rcnnShapes):
    out = output.draw_instance_predictions(human_instances.to("cpu"))
    width = int(rcnnShapes[0].shape[1])
    height = int(rcnnShapes[0].shape[0])
    dim = (width, height)

    return cv2.resize(out.get_image()[:, :, ::-1], dim, interpolation = cv2.INTER_AREA)

def calculate_Rcnn_mask(img, confidence_threshold):
    
    ##Datos de configuración
    cfg = setup_cfg(confidence_threshold)

    ##Construimos el predictor
    predictor = DefaultPredictor(cfg)

    ##Aplicamos el predictor a la imagen
    outputs = predictor(img)

    ##Construimos el visualizador
    output = Visualizer(img[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)

    ##Filtramos los humanos
    human_instances = outputs['instances'][outputs['instances'].pred_classes == 0]
    
    ##Extreamos las imágenes
    masks = human_instances.pred_masks.cpu().numpy()

    ##Convertimos las máscaras a imágenes y las almacenamos en un array
    images_mask = [] ##Declaramos el array
    for mask in masks:
        imgMask = np.zeros_like(img) ##Declaramos la imagen, en negro
        imgMask[np.where(mask==1)] = [255, 255, 255] ##Blanqueamos los píxeles de la máscara
        images_mask.append(imgMask) ##Anyadimos al array a devolver

    ##Generamos y redimensionamos la imagen a mostrar
    img_to_show = resize_Rcnn_Img_Show(output, human_instances, images_mask)

    return img_to_show, images_mask

