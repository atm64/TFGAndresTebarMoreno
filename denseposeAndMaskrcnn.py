# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
from glob import glob
import logging
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

##import pickle
import sys
##from typing import Any, ClassVar, Dict, List
import torch
import numpy as np
import cv2
##import multiprocessing as mp
import tqdm
##import check_done_videos as chk
##from utils.filesystem import create_dir
##import json

from Mask_Rcnn import calculate_Rcnn_mask

from Densepose import densePose ##, order_denseposes, calculate_mask_shapes_and_coordinates

from ImageInpainting import image_inpainting_simple, image_inpainting_no_people, image_inpainting_complete, inpaint

# sys.path.insert(0,'/home/pau/git/detectron2_repo/projects/DensePose')
# sys.path.insert(0,'/home/i2rc/code/detectron2/projects/DensePose')
##sys.path.insert(0,'/home/andres/TFG/densepose/detectron2/projects/DensePose')
sys.path.insert(0,'./detectron2/projects/DensePose')

from detectron2.config import get_cfg
from detectron2.engine.defaults import DefaultPredictor

from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

from densepose import add_densepose_config
# from densepose.data.structures import DensePoseResult
from densepose.structures import quantize_densepose_chart_result
#from densepose.vis.densepose import DensePoseResultsVisualizer
from densepose.vis.densepose_results import DensePoseResultsVisualizer
from densepose.vis.extractor import CompoundExtractor, create_extractor



from detectron2.data.detection_utils import read_image
##from detectron2.utils.logger import setup_logger

from detectron2.engine import DefaultPredictor


def get_parser():
    parser = argparse.ArgumentParser(description="TFG Andres Tebar")
    """ parser.add_argument(
        "--config-file",
        ##default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        default="detectron2/configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    ) """
    parser.add_argument(
        "--input",
        type=str,
        default="images/michaeljordan.jpg",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.75,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--inpaint",
        type=int,
        default=3,
        help="Tipo de inpainting a realizar",
    )
    parser.add_argument(
        "--iterate",
        type=int,
        default=0,
        help="Cuando se realiza el borrado se trabaja sobre la imagen original o sobre el resultado",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


def mousePoints(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:##Al clicar
        ##print(x,y)##Imprime coordenadas
        img = params[0]
        shapes = params[1]
        all_masks = params[2]
        inpaint = params[3]
        shape = False

        for shapeI in shapes:
            shapeImg = cv2.cvtColor(shapeI, cv2.COLOR_BGR2RGB) ##Imagen de referencia
            ##imgMask = cv2.cvtColor(shapeI, cv2.COLOR_BGR2RGB) #Imagen con la que llamar al inpainting
            
            b, g, r = shapeImg[y,x]
            if b == 255 and g == 255 and r == 255:
                shape = True

                if inpaint==1:
                    shape_inpainted = image_inpainting_simple(img, shapeImg)
                elif inpaint ==2:    
                    shape_inpainted = image_inpainting_no_people(img, shapeImg, all_masks)
                else:
                    shape_inpainted = image_inpainting_complete(img, shapeImg, all_masks)
                
                cv2.imshow('Image Inpainting', shape_inpainted)
        
        ##if not shape:
            ##print(str(-1))

""" def eraseShape(img, shape): ##Elimina silueta
    return cv2.add(img, whitenShape(shape)) """


def whitenShape(imgShape): ##Blanquea la silueta
    return np.uint8(imgShape > 0) * 255
    ##return cv2.cvtColor(np.uint8(imgShape > 0) * 255, cv2.COLOR_BGR2GRAY)


def union_masks_instances(denseshapes, rcnnShapes):##Une siluetas teniendo en cuenta el minimo numero de pixeles en blanco

##HAY QUE ARREGLAR EL PROBLEMA DE evitar que las sliuetas pequeñas se junten con las grandes, PARA ELLO ELIMINAR DEL ARRAY
    union_masks = []
    for i in range(0, len(denseshapes)):
        im1 = cv2.cvtColor(whitenShape(denseshapes[i]), cv2.COLOR_BGR2GRAY)
        ##im1 = cv2.cvtColor(whitenShape(denseshapes[i]), cv2.COLOR_BGR2RGB)
        ##im2 = rcnnShapes[i]
        ##im2 = cv2.cvtColor(rcnnShapes[i], cv2.COLOR_BGR2RGB)
        im2 = cv2.cvtColor(rcnnShapes[i], cv2.COLOR_BGR2GRAY)
        best_match = i
        min_union = cv2.countNonZero(np.maximum(im1, im2))
        for j in range(0, len(rcnnShapes)):
            if(cv2.countNonZero(np.maximum(im1, cv2.cvtColor(rcnnShapes[j], cv2.COLOR_BGR2GRAY))) < min_union):
                min_union = cv2.countNonZero(np.maximum(im1, cv2.cvtColor(rcnnShapes[j], cv2.COLOR_BGR2GRAY)))
                im2 = cv2.cvtColor(rcnnShapes[j], cv2.COLOR_BGR2GRAY)
                best_match = j

        ##union_masks.append(cv2.add(im1, im2))
        union_masks.append(np.maximum(im1, im2))
        ##rcnnShapes.pop(best_match)##Eliminamos para evitar que las sliuetas pequeñas se junten con las grandes
        ##denseshapes.pop(i)
        ##denseshapes.remove(im1)
        
    return union_masks

def mask_all_instances(union_masks, img):
    all_masks = np.zeros_like(img) ##Imagen de referencia

    for mask in union_masks:
        shapeImg = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        all_masks[np.where((shapeImg==[255, 255, 255]).all(axis=2))] = 255
    
    return all_masks


""" def resize_Rcnn_Img_Show(output, rcnnShapes):
    out = output.draw_instance_predictions(human_instances.to("cpu"))
    width = int(rcnnShapes[0].shape[1])
    height = int(rcnnShapes[0].shape[0])
    dim = (width, height)

    return cv2.resize(out.get_image()[:, :, ::-1], dim, interpolation = cv2.INTER_AREA) """

if __name__ == '__main__':
    
    print('Start')

    args = get_parser().parse_args()
    
    
    img = cv2.imread(args.input)
    confidence_threshold = args.confidence_threshold
    inpaint_option = args.inpaint
    ##confidence_threshold=0.75
    ##human_instances, img_to_show, rcnnShapes = calculate_Rcnn_mask(img, confidence_threshold)
    img_to_show, rcnnShapes = calculate_Rcnn_mask(img, confidence_threshold)
    denseshapes = densePose(img, confidence_threshold)
    ##dense_ordered = order_denseposes(denseshapes)

    
    print(len(rcnnShapes))
    print(len(denseshapes))
    
    ##union_masks = union_masks_instances(dense_ordered, rcnnShapes)
    union_masks = union_masks_instances(denseshapes, rcnnShapes)
    if(len(union_masks) == 1):
        inpaint_option = 1

    all_masks = mask_all_instances(union_masks, img)
    
    ##cv2.imshow('Mask-R_cnn', resize_Rcnn_Img_Show(output, rcnnShapes))
    ##cv2.imshow('Mask-R_cnn', img_to_show)
    cv2.imshow('Shapes', img_to_show)

    ## Encapsulamos parametros: images, siluetas
    ##masks = human_instances.pred_masks.cpu().numpy()
    params = (img, union_masks, all_masks, inpaint_option) ##, masks)##, coordinates)

    ## Llamamos al evento al dar click
    ##cv2.setMouseCallback('Mask-R_cnn', mousePoints, params)
    cv2.setMouseCallback('Shapes', mousePoints, params)

    k = cv2.waitKey(0)
    if k == ord('q'):
        cv2.destroyAllWindows()

    print('Done.')