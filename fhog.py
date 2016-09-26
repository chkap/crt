__author__ = 'chkap'


import math
import numpy as np
import matplotlib.pyplot as plt
import cv2

_orientation_vector = None

def _prepare_orientation_vector(orientation_bin):
    global _orientation_vector

    if _orientation_vector is not None and _orientation_vector.shape[0] == orientation_bin*2:
        return _orientation_vector
    else:
        tmp = []
        step = np.pi/orientation_bin
        for i in range(0,2*orientation_bin):
            alpha = i* step
            vy = math.sin(alpha)
            vx = math.cos(alpha)
            tmp.append([vx,vy])
        _orientation_vector = np.array(tmp,dtype=np.float)
        return _orientation_vector

def _get_gradient(im):
    w = im.shape[1]
    h = im.shape[0]
    border_im = cv2.copyMakeBorder(im,1,1,1,1,cv2.BORDER_REPLICATE)
    if border_im.ndim == 2:
        border_im = border_im[:,:,np.newaxis]
    _up = border_im[0:0+h,1:1+w,:]
    _down = border_im[2:2+h,1:1+w,:]
    dy = _down - _up
    _left = border_im[1:1+h,0:0+w,:]
    _right = border_im[1:1+h,2:2+w,:]
    dx = _right - _left

    dy2 = dy*dy
    dx2 = dx*dx
    dmag = dx2+dy2
    col_ind = np.arange(0,h)
    row_ind = np.arange(0,w)
    max_mag_ind = np.argmax(dmag,axis=2)
    mag = dmag[col_ind[:,np.newaxis],row_ind,max_mag_ind]
    mag = np.sqrt(mag)
    max_dy = dy[col_ind[:,np.newaxis],row_ind,max_mag_ind]
    max_dx = dx[col_ind[:,np.newaxis],row_ind,max_mag_ind]
    dxy = np.concatenate((max_dx[:,:,np.newaxis],max_dy[:,:,np.newaxis]),axis=2)

    # plt.figure()
    # plt.clf()
    # plt.subplot(131)
    # plt.imshow(mag)
    # plt.subplot(132)
    # plt.imshow(dxy[:,:,0])
    # plt.subplot(133)
    # plt.imshow(dxy[:,:,1])
    return mag,dxy

def _get_histogram(mag,dxy,bin):
    ov = _prepare_orientation_vector(bin)
    dot = dxy[:,:,np.newaxis,:]*ov[np.newaxis,np.newaxis,:,:]
    sum_dot = np.sum(dot,axis=3)
    max_orientation = np.argmax(sum_dot,axis=2)
    w = dxy.shape[1]
    h = dxy.shape[0]
    hist = np.zeros((h,w,bin*2),dtype=np.float)
    col_ind = np.arange(0,h)
    row_ind = np.arange(0,w)
    hist[col_ind[:,np.newaxis],row_ind,max_orientation] = mag
    return hist

def _get_cell_hist(pixel_hist, cw=4,ch=4):
    w = pixel_hist.shape[1]
    h = pixel_hist.shape[0]
    cell_num_x = int(w / float(cw) + 0.5)
    cell_num_y = int(h / float(ch) + 0.5)
    hist_bin = pixel_hist.shape[2]
    cell_hist = np.zeros((cell_num_y,cell_num_x,hist_bin),dtype=np.float)
    for i in range(0, cell_num_x):
        for j in range(0,cell_num_y):
            cx = cw/2.0-0.5+i*cw
            cy = ch/2.0-0.5+j*ch
            left = math.ceil(cx-cw)
            left = int(max(left,0))
            right = math.ceil(cx+cw)
            right = int(min(right,w))
            top = math.ceil(cy-ch)
            top = int(max(top,0))
            bottom = math.ceil(cy+ch)
            bottom = int(min(bottom,h))
            assert right > left and bottom > top
            xind = np.arange(left,right)
            xind = np.tile(xind[np.newaxis,:],(bottom-top,1))
            yind = np.arange(top,bottom)
            yind = np.tile(yind[:,np.newaxis],(1,right-left))
            pos = np.concatenate((xind[:,:,np.newaxis],yind[:,:,np.newaxis]),axis=2)
            dp = pos - np.array([cx,cy])[np.newaxis,np.newaxis,:]
            rel_dp = np.abs(dp) / (np.array((cw,ch),dtype=np.float)[np.newaxis,np.newaxis,:])
            weights = 1.0 - rel_dp
            weights = weights[:,:,0]*weights[:,:,1]
            assert (weights >= 0).all()
            neighbor_pixel_hist = pixel_hist[top:bottom,left:right:,:]
            weight_neighbors = neighbor_pixel_hist * weights[:,:,np.newaxis]
            weight_neighbors = np.reshape(weight_neighbors,(weight_neighbors.shape[0]*weight_neighbors.shape[1],hist_bin))
            cell_hist[j,i,:] = np.sum(weight_neighbors,axis=0)


    norm_tmp = cell_hist[:,:,0:hist_bin/2]+cell_hist[:,:,hist_bin/2:hist_bin]
    cell_norm = (norm_tmp*norm_tmp).sum(axis=2)
    return cell_hist,cell_norm


def _normalize_hist(cell_hist, cell_norm):
    w = cell_norm.shape[1]
    h = cell_norm.shape[0]
    border_norm = cv2.copyMakeBorder(cell_norm,1,1,1,1,cv2.BORDER_REPLICATE)
    sx = 0; sy = 0
    topleft = border_norm[sy:sy+h,sx:sx+w]
    sx = 1; sy = 0
    top = border_norm[sy:sy+h,sx:sx+w]
    sx = 2; sy = 0
    topright = border_norm[sy:sy+h,sx:sx+w]
    sx = 0; sy = 1
    left = border_norm[sy:sy+h,sx:sx+w]
    sx = 1; sy = 1
    center = border_norm[sy:sy+h,sx:sx+w]
    sx = 2; sy = 1
    right = border_norm[sy:sy+h,sx:sx+w]
    sx = 0; sy = 2
    bottomleft = border_norm[sy:sy+h,sx:sx+w]
    sx = 1; sy = 2
    bottom = border_norm[sy:sy+h,sx:sx+w]
    sx = 2; sy = 2
    bottomright = border_norm[sy:sy+h,sx:sx+w]

    eps = 0.002
    norm1 = 1.0/np.sqrt(topleft + top + center + left + eps)
    norm2 = 1.0/np.sqrt(top + topright + center + right + eps)
    norm3 = 1.0/np.sqrt(left + center + bottomleft + bottom + eps)
    norm4 = 1.0/np.sqrt(center + right + bottom + bottomright + eps)

    hist_bin = cell_hist.shape[2]
    insensitive_hist = cell_hist[:,:,0:hist_bin//2] + cell_hist[:,:,hist_bin//2:hist_bin]
    hist = np.concatenate((cell_hist,insensitive_hist),axis=2)
    h1 = hist*norm1[:,:,np.newaxis]
    h1 = np.fmin(h1,0.2)
    t1 = np.sum(h1[:,:,0:hist_bin],axis=2) * 0.2357
    h2 = hist*norm2[:,:,np.newaxis]
    h2 = np.fmin(h2,0.2)
    t2 = np.sum(h2[:,:,0:hist_bin],axis=2) * 0.2357
    h3 = hist*norm3[:,:,np.newaxis]
    h3 = np.fmin(h3,0.2)
    t3 = np.sum(h3[:,:,0:hist_bin],axis=2) * 0.2357
    h4 = hist*norm4[:,:,np.newaxis]
    h4 = np.fmin(h4,0.2)
    t4 = np.sum(h4[:,:,0:hist_bin],axis=2) * 0.2357

    hist_feature = (h1+h2+h3+h4)*0.5
    final_fhog = np.concatenate((hist_feature,t1[:,:,np.newaxis],t2[:,:,np.newaxis],
                                 t3[:,:,np.newaxis],t4[:,:,np.newaxis]),axis=2)
    return final_fhog


def get_fhog(im, cell_size, bin):
    if im.ndim == 2:
        im = im[:,:,np.newaxis]
    assert im.ndim == 3
    im = np.asarray(im,dtype=np.float)
    mag, dxy = _get_gradient(im)
    pixel_hist = _get_histogram(mag,dxy,bin)
    cell_hist,cell_norm = _get_cell_hist(pixel_hist,cell_size,cell_size)
    fhog = _normalize_hist(cell_hist,cell_norm)
    return fhog


def test():
    im_path = 'E:\\test.jpg'
    im = cv2.imread(im_path)
    plt.figure()
    plt.imshow(im)

    im = np.asarray(im,np.float)
    fhog = get_fhog(im,4,9)
    for i in range(fhog.shape[2]):
        plt.figure()
        plt.imshow(fhog[:,:,i])
        plt.colorbar()
        plt.title('channel:%d'%i)
    plt.show()

if __name__ == '__main__':
    test()


