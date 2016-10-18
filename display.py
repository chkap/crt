import os
import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import cv2

from simgeo import Rect


plt.ion()


def show_float_image(image,figure_id = 62346):
    image = np.asarray((image+0.5)*255.0,dtype=np.uint8)
    g,b,r = cv2.split(image)
    image = cv2.merge((r,g,b))
    plt.figure(figure_id)
    plt.clf()
    plt.imshow(image)
    plt.show()


def show_map(map, figure_id=0):
    plt.figure(figure_id)
    plt.clf()
    plt.imshow(map, norm=matplotlib.colors.Normalize(vmin=0.0, vmax=1.0), aspect='auto',interpolation='nearest')
    plt.colorbar()
    plt.show()
    plt.pause(0.01)


def show_track_res(frame_id, image, rect, gt_rect, fid):
    image = image.copy()
    cv2.rectangle(image,rect.get_tl(),rect.get_dr(),(0,0,255),2)
    cv2.rectangle(image,gt_rect.get_tl(),gt_rect.get_dr(),(255,0,0),2)
    g,b,r = cv2.split(image)
    image = cv2.merge((r,g,b))
    plt.figure(fid)
    plt.cla()
    plt.imshow(image)
    plt.title('fid:{:4d}'.format(frame_id))
    plt.show()
    plt.pause(0.01)


def show_res(sec_image,obj_image,conf,out_conf,preprocess):
    conf = conf.copy()
    out_conf = out_conf.copy()
    si = sec_image.copy()

    si = np.asarray((si+0.5)*255.0,dtype=np.uint8)
    si_w = si.shape[1]
    si_h = si.shape[0]

    ti = obj_image.copy()
    ti = np.asarray((ti+0.5)*255.0,dtype=np.uint8)

    conf_y,conf_x = np.unravel_index(np.argmax(out_conf),out_conf.shape)
    predict_w = si_w/2.5
    predict_h = si_h/2.5

    srect = Rect(0,0,sec_image.shape[1],sec_image.shape[0])
    pcx,pcy = preprocess.predict_location(srect,conf_x,conf_y)
    tlx = int(pcx - (predict_w -1)/2.0 +0.5)
    tly = int(pcy - (predict_h -1)/2.0 +0.5)
    obj_rect = Rect(tlx,tly,predict_w,predict_h).get_int_rect()
    cv2.rectangle(si,obj_rect.get_tl(),obj_rect.get_dr(),(255,0,0),2)

    plt.figure(23245)
    plt.clf()
    plt.imshow(si)


    plt.figure(12897)
    plt.clf()
    plt.subplot(131)
    plt.imshow(ti)
    plt.subplot(132)
    plt.imshow(conf)
    plt.colorbar()
    plt.subplot(133)
    plt.imshow(out_conf)
    plt.colorbar()

    plt.show()
    plt.pause(0.1)


def show_seq_data(track_seq):
    for i in range(len(track_seq.images)):
        g,b,r = cv2.split(track_seq.images[i])
        image = cv2.merge([r,g,b])
        rect = track_seq.rects[i].get_int_rect()
        cv2.rectangle(image,rect.get_tl(),rect.get_dr(),(255,0,0,),thickness=2)

        plt.figure(13334)
        plt.cla()
        plt.title('%s frame %d'%(track_seq.name,i))
        plt.imshow(image)
        plt.show()
        plt.pause(0.001)



if __name__ == '__main__':

    # display()
    pass

