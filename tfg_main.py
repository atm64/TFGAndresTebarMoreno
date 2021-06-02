# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
from glob import glob
import logging
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import cv2

from Mask_Rcnn import calculate_Rcnn_mask

from Densepose import densePose

from ImageInpainting import image_inpainting_simple, image_inpainting_no_people, image_inpainting_complete, image_inpainitng_restore_others, delete_all_people



def get_parser():
    parser = argparse.ArgumentParser(description="TFG Andres Tebar")
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
        "--detection-mask",
        type=int,
        default=0,
        help="Mascara para seleccionar instancia (0: Mask R-cnn, 1: Densepose)",
    )
    parser.add_argument(
        "--inpaint",
        type=int,
        default=0,
        help="Tipo de inpainting a realizar (0: Completa, 1: Simple, 2: Sin personas)",
    )
    parser.add_argument(
        "--recursive",
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
        recursive = params[4]

        for shapeImg in shapes:
            b, g, r = shapeImg[y,x] ##Extrae los colores del la mascara en el punto clicado
            if b == 255 and g == 255 and r == 255: ##Comprueba que es blanco

                print('Realizando inpaintinng...')
                if inpaint==1:
                    shape_inpainted = image_inpainting_simple(img, shapeImg)
                elif inpaint==2:    
                    shape_inpainted = image_inpainting_no_people(img, shapeImg, all_masks)
                elif inpaint==3:
                    shape_inpainted = image_inpainting_complete(img, shapeImg, all_masks)
                elif inpaint==4:
                    shape_inpainted = delete_all_people(img, all_masks)
                else:
                    shape_inpainted = image_inpainitng_restore_others(img, shapeImg, all_masks, shapes)

                print('Inpaintinng realizado')

                if recursive:
                    img[:] = shape_inpainted
                
                cv2.imshow('Image Inpainting', shape_inpainted)
                ##cv2.imwrite('./results/Jordan_Inpaint.jpg', shape_inpainted)

def union_masks_instances(denseshapes, rcnnShapes):##Une siluetas teniendo en cuenta el minimo numero de pixeles en blanco

##HAY QUE ARREGLAR EL PROBLEMA DE evitar que las sliuetas pequeñas se junten con las grandes, PARA ELLO ELIMINAR DEL ARRAY
    union_masks = []
    for i in range(0, len(denseshapes)):
        im1 = cv2.cvtColor(denseshapes[i], cv2.COLOR_BGR2GRAY)
        im2 = cv2.cvtColor(rcnnShapes[i], cv2.COLOR_BGR2GRAY)
        best_match = i
        min_union = cv2.countNonZero(np.maximum(im1, im2))
        for j in range(0, len(rcnnShapes)):
            if(cv2.countNonZero(np.maximum(im1, cv2.cvtColor(rcnnShapes[j], cv2.COLOR_BGR2GRAY))) < min_union):
                min_union = cv2.countNonZero(np.maximum(im1, cv2.cvtColor(rcnnShapes[j], cv2.COLOR_BGR2GRAY)))
                im2 = cv2.cvtColor(rcnnShapes[j], cv2.COLOR_BGR2GRAY)
                best_match = j

        ##union_masks.append(cv2.add(im1, im2))
        union_masks.append(cv2.cvtColor(np.maximum(im1, im2), cv2.COLOR_BGR2RGB))
        ##rcnnShapes.pop(best_match)##Eliminamos para evitar que las sliuetas pequeñas se junten con las grandes
        ##denseshapes.pop(i)
        ##denseshapes.remove(im1)
        
    return union_masks

def mask_all_instances(union_masks, img):
    all_masks = np.zeros_like(img) ##Imagen con todas las máscaras unidas

    for mask in union_masks:##Iterador para anyadir las máscaras
        ##shapeImg = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        ##all_masks[np.where((shapeImg==[255, 255, 255]).all(axis=2))] = 255
        all_masks[np.where((mask==[255, 255, 255]).all(axis=2))] = 255
    
    return all_masks


if __name__ == '__main__':
    
    print('Start')

    ##Porcesamos los argumentos indicados en la ejecucion del programa
    args = get_parser().parse_args()
    
    ##Declaramos la imagen para la ejecucion
    img = cv2.imread(args.input)

    ##Establecemos el umbral de confianza
    confidence_threshold = args.confidence_threshold

    ##Indicamos el tipo de inpainting
    inpaint_option = args.inpaint

    ##Comprobamos si se realiza un borrado recursivo
    recursive = (args.recursive==1)
    
    ##Aplicamos el modelo de deteccion de objetos Mask R-CNN
    img_mask_rcnn, rcnnShapes = calculate_Rcnn_mask(img, confidence_threshold)

    ##Aplicamos el modelo de deteccion de objetos DensePose
    img_mask_densepose, denseshapes = densePose(img, confidence_threshold)
    
    ##print(len(rcnnShapes))
    ##print(len(denseshapes))
    
    ##Unimos las mascaras de los dos modelos de deteccion
    union_masks = union_masks_instances(denseshapes, rcnnShapes)
    
    ##Generamos las mascara con todas las instancias
    if(len(union_masks) == 1):
        inpaint_option = 1
        all_masks = union_masks[0]
    else:
        all_masks = mask_all_instances(union_masks, img)
    
    if(args.detection_mask == 1):
        cv2.imshow('Shapes', img_mask_densepose)
    else:
        cv2.imshow('Shapes', img_mask_rcnn)
    
    """ cv2.imshow('R-CNN 1', rcnnShapes[0])
    cv2.imwrite('./results/Jordan_R-CNN1.jpg', rcnnShapes[0])
    cv2.imshow('Mask R-CNN', img_mask_rcnn)
    cv2.imwrite('./results/Jordan_MaskR-CNN.jpg', img_mask_rcnn)
    cv2.imshow('DensePose 1', denseshapes[0])
    cv2.imwrite('./results/Jordan_DensePose1.jpg', denseshapes[0])
    cv2.imshow('Mask DensePose', img_mask_densepose)
    cv2.imwrite('./results/Jordan_MaskDensePose.jpg', img_mask_densepose)
    cv2.imshow('Union 1', union_masks[0])
    cv2.imwrite('./results/Jordan_Union1.jpg', union_masks[0]) """


    ## Encapsulamos parametros: images, siluetas individuales, siluetas global, opción inpaint 
    params = (img, union_masks, all_masks, inpaint_option, recursive)

    ##Disparamos el evento al dar click sobre la imagen
    cv2.setMouseCallback('Shapes', mousePoints, params)

    k = cv2.waitKey(0)
    if k == ord('q'):
        cv2.destroyAllWindows()

    print('Done.')