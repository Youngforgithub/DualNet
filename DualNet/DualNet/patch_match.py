import numpy as np
import cv2
from PIL import Image
import tensorflow as tf



def normalize(F_L):
    return F_L/np.sqrt(np.sum(np.square(F_L)))

def cal_dist(A, B, ay, ax, by, bx, patch_size):
    """
    Calculate distance between a patch in A to a patch in B.
    :return: Distance calculated between the two patches
    """
    dx0 = dy0 = patch_size // 2
    dx1 = dy1 = patch_size // 2 + 1
	#这里取最小值得意义是防止矩阵溢出，实际都是在nnf的基础上搜索path大小的邻域
    dx0 = min(ax, bx, dx0)
    dx1 = min(A.shape[1] - ax, B.shape[1] - bx, dx1)
    dy0 = min(ay, by, dy0)
    dy1 = min(A.shape[0] - ay, B.shape[0] - by, dy1)
    return np.sum ((A[ay - dy0:ay + dy1, ax - dx0:ax + dx1] - B[by - dy0:by + dy1, bx - dx0:bx + dx1]) ** 2)/ ((dx1 + dx0) * (dy1 + dy0))


def initialise_nnf(S, D, patch_size):
    """
    Set up a random NNF
    Then calculate the distances to fill up the NND
    :return:
    """
    nnd = np.zeros(shape=(S.shape[0], S.shape[1]))  # the distance map for the nnf
    nnf = np.zeros(shape=(2, S.shape[0], S.shape[1])).astype(np.int)  # the nearest neighbour field
    nnf[0] = np.random.randint(D.shape[1], size=(S.shape[0], S.shape[1]))
    nnf[1] = np.random.randint(D.shape[0], size=(S.shape[0], S.shape[1]))
    nnf = nnf.transpose((1, 2, 0))
    for i in range(D.shape[0]):
        for j in range(D.shape[1]):
            pos = nnf[i, j]
            nnd[i, j] = cal_dist(S, D, i, j, pos[1], pos[0], patch_size)
    return nnf, nnd

def propagate(A, B, nnf, nnd, iters=2, rand_search_radius=6, patch_size=3, queue=None):
    """
    Optimize the NNF using PatchMatch Algorithm
    :param iters: number of iterations
    :param rand_search_radius: max radius to use in random search
    :return:
    """

    a_cols = A.shape[1]
    a_rows = A.shape[0]

    b_cols = B.shape[1]
    b_rows = B.shape[0]

    for it in range(iters):
        ystart = 0
        yend = a_rows
        ychange = 1
        xstart = 0
        xend = a_cols
        xchange = 1

        if it % 2 == 1:
            xstart = xend - 1
            xend = -1
            xchange = -1
            ystart = yend - 1
            yend = -1
            ychange = -1

        ay = ystart
        while ay != yend:
            ax = xstart
            while ax != xend:
                xbest, ybest = nnf[ay, ax]
                dbest = nnd[ay, ax]
				##搜索周边单位为一范围内的邻域的距离？？？
                if ax - xchange < a_cols and ax - xchange >= 0:
                    vp = nnf[ay, ax - xchange]
                    xp = vp[0] + xchange
                    yp = vp[1]
                    if xp < b_cols and xp >= 0:
                        val = cal_dist(A, B, ay, ax, yp, xp, patch_size)
                        if val < dbest:
                            xbest, ybest, dbest = xp, yp, val

                if abs(ay - ychange) < a_rows and ay - ychange >= 0:
                    vp = nnf[ay - ychange, ax]
                    xp = vp[0]
                    yp = vp[1] + ychange
                    if yp < b_rows and yp >= 0:
                        val = cal_dist(A, B, ay, ax, yp, xp, patch_size)
                        if val < dbest:
                            xbest, ybest, dbest = xp, yp, val
                if rand_search_radius is None:
                    rand_d = max(B.shape[0], B.shape[1])
                else:
                    rand_d = rand_search_radius

                while rand_d >= 1:
                    try:
						##设置搜索范围，最小为0，最大为矩阵的max，在xbest,ybest的基础上加减rand_d
                        xmin = max(xbest - rand_d, 0)
                        xmax = min(xbest + rand_d, b_cols)

                        ymin = max(ybest - rand_d, 0)
                        ymax = min(ybest + rand_d, b_rows)

                        if xmin > xmax:
                            rx = -np.random.randint(xmax, xmin)
                        if ymin > ymax:
                            ry = -np.random.randint(ymax, ymin)

                        if xmin <= xmax:
                            rx = np.random.randint(xmin, xmax)
                        if ymin <= ymax:
                            ry = np.random.randint(ymin, ymax)

                        val = cal_dist(A, B,ay, ax, ry, rx, patch_size)
                        if val < dbest:
                            xbest, ybest, dbest = rx, ry, val

                    except Exception as e:
                        print(e)
                        print(rand_d)
                        print(xmin, xmax)
                        print(ymin, ymax)
                        print(xbest, ybest)
                        print(B.shape)

                    rand_d = rand_d // 2

                nnf[ay, ax] = [xbest, ybest]
                nnd[ay, ax] = dbest

                ax += xchange
            ay += ychange

    if queue:
        queue.put(nnf)
    return nnf, nnd

def reconstruct_image(img_a, nnf,img_b):
    """
    Reconstruct image using the NNF and img_a.
    :param img_a: the patches to reconstruct from
    :return: reconstructed image
    """
    final_img = np.zeros([img_b.shape[0],img_b.shape[1],img_b.shape[2]])
    size = nnf.shape[0]
    for i in range(nnf.shape[0]):
        for j in range(nnf.shape[1]):
            x, y = nnf[i, j]
            if final_img[ y: (y + 1),  x: (x + 1)].shape == img_a[ i: (i + 1),j: (j + 1)].shape:
                final_img[ y: (y + 1),  x: (x + 1),:] = img_a[ i: (i + 1),j: (j + 1),:]
    final_img = final_img[np.newaxis,:,:,:].astype(np.float32)

    return final_img

def patch_match(s_out,d_out):
	#patch match

	##normalize
	#crop_S=crop(s_out,d_out)
	#Image.fromarray(np.uint8(crop_S[0,:,:,:]*255)).show()
	#Image.fromarray(np.uint8(s_out[0,:,:,0]*255)).show()
	#phi is the map from S to D,and nnd is the distance form S to D
	phi,nnd = initialise_nnf(s_out[0, :, :, :], d_out[0, :, :, :], 3)
	phi,nnd = propagate(s_out[0, :, :, :], d_out[0, :, :, :], phi, nnd, iters=5, rand_search_radius=6, patch_size=3)
	d_res=reconstruct_image(d_out[0,:,:,:],phi,s_out[0,:,:,:])
	#Image.fromarray(np.uint8(d_res[0,:,:,0]*255)).show()
	return d_res
	

