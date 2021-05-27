# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import glob
import multiprocessing as mp
import os
import time
import cv2
import tqdm

from detectron2.config import get_cfg


if __name__ == '__main__':
    
    print('Start')

    img = cv2.imread("images/tarko.jpg")
    ##img = cv2.imread("images/mayor.jpg")
    ##img = cv2.imread("images/Nostalghia.png")
    
    human_instances, output, rcnnShapes = maskRcnn(img)
    denseshapes = densePose(img)
    dense_ordered = order_denseposes(denseshapes)

    
    print(len(rcnnShapes))
    print(len(denseshapes))
    
    union_masks = union_masks_instances(dense_ordered, rcnnShapes)
    
    for i in range(0, len(union_masks)):
        erasedImg = cv2.add(img, cv2.cvtColor(union_masks[i], cv2.COLOR_BGR2RGB))
        ##erasedImg = np.maximum(img, cv2.cvtColor(union_masks[i], cv2.COLOR_BGR2RGB))
        
        ##cv2.imshow('union'+str(i), erasedImg)
        ##cv2.imshow('union'+str(i), cv2.cvtColor(union_masks[i], cv2.COLOR_BGR2RGB))

    
    ##cv2.imshow('Mask-R_cnn', out.get_image()[:, :, ::-1])
    cv2.imshow('Mask-R_cnn', resize_Rcnn_Img_Show(output, rcnnShapes))

    ## Encapsulamos parametros: images, siluetas y coordenadas
    masks = human_instances.pred_masks.cpu().numpy()
    params = (img, union_masks, masks)##, coordinates)

    ## Llamamos al evento al dar click
    cv2.setMouseCallback('Mask-R_cnn', mousePoints, params)

    k = cv2.waitKey(0)
    if k == ord('s'):
        cv2.destroyAllWindows()

    print('Done.')