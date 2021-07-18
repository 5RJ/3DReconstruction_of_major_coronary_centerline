'''
author:5R+ , 2020.3.19
'''

import numpy as np
import math
from scipy.spatial.distance import squareform
from math import pi
from geomdl import BSpline
from geomdl import utilities
from .Basic import pix2cor

def ed_1dim(m, n): # for one pair points
    return np.sqrt(np.sum((m - n) ** 2))

def norm_angle(angle):
    # angles betewwn[0,2*pi]
    if angle>=0:
        return angle
    elif angle<0:
        return 2*pi+angle
    else:
        print('Error:angle!!')
        return 0

def cost(X,Y):
    eps = np.finfo(float).eps
    X = np.mat(X).squeeze()
    Y = np.mat(Y).squeeze()
    m = X.shape[0]
    n = Y.shape[0]
    D = np.zeros((m,n))
    D= np.mat(D)
    for i in range(n):
        yi = Y[i,:] # sc of one point on Y
        yiRep = np.repeat(yi,m,axis=0) # (1,60) ---> (m,60)
        s = yiRep + X
        d = yiRep - X
        d = np.array(d)
        s = np.array(s)
        D[:,i] = np.sum(np.square(d)/(s + eps), axis=1).reshape(m,1)

    return D


class ShapeContext(object):
    def __init__(self,points,param_list,curve,r1=0.125, r2=2.2, nbins_theta=12, nbins_r=5):
        print ('Step into Shape context...')
        self.r1 = r1
        self.r2 = r2
        self.nbins_theta = nbins_theta
        self.nbins_r = nbins_r
        self.desc_size = nbins_r * nbins_theta
        self.points = points
        self.param_list = param_list
        self.curve = curve

    def compute_dist(self):
        print ('    compute_dist')
        points = self.points
        dist_mat = np.zeros((len(points),len(points)))
        for i in range(len(points)):
            p1 = points[i,:]
            for j in range(i+1,len(points)):
                p2 = points[j,:]
                dist_mat[i,j] = ed_1dim(p1,p2)
                dist_mat[j,i] = dist_mat[i,j]
        mean_dist = np.mean(dist_mat)
        dist_mat = dist_mat/mean_dist # normalization

        return dist_mat,mean_dist

    def compute_angles(self):
        print ('    compute_angles')
        """ compute angles between a set of points """
        points = self.points # [sample_num,2]
        param_list = self.param_list
        angles = np.zeros((len(points), len(points)))
        for i in range(len(points)):
            p1 = points[i, :]
            param = param_list[i]
            # from ipdb import set_trace; set_trace()
            tangent_vector = self.curve.tangent(param)[1]
            tan_angle = math.atan2(tangent_vector[1],tangent_vector[0])
            for j in range(len(points)):
                if i==j:
                    continue
                p2 = points[j, :]
                # compute the angle between the line,which is composed of p1 and p2, and the +x axis(i.e.slope)
                angles[i, j] = norm_angle(math.atan2((p2[1] - p1[1]), (p2[0] - p1[0]))-tan_angle)
        return angles # radian

    def compute_angles_abs(self):
        print ('    compute_angles')
        """ compute angles between a set of points """
        points = self.points # [sample_num,2]
        angles = np.zeros((len(points), len(points)))
        for i in range(len(points)):
            p1 = points[i, :]
            for j in range(len(points)):
                if i==j:
                    continue
                p2 = points[j, :]
                # compute the angle between the line,which is composed of p1 and p2, and the +x axis(i.e.slope)
                angles[i, j] = norm_angle(math.atan2((p2[1] - p1[1]), (p2[0] - p1[0])))
        return angles # radian

    def compute_histogram(self,radius):
        print ('    compute_histogram')
        # compute distance matrix
        dist_mat,mean_dist = self.compute_dist()
        dist_mat = squareform(dist_mat)
        # compute relative angles
        angles_mat = self.compute_angles()
        # compute absolute angles
        # angles_mat = self.compute_angles_abs()

        #quantize angles to a bin
        angle_bins = np.floor(angles_mat / (2 * pi / self.nbins_theta)) # the row idx of every 'other' point

        # quantize dist to bins
        # ----origin-------
        # radius_regions = np.logspace(self.r1,self.r2, num=self.nbins_r)

        # ---0417: change the region into equaldistant section-----
        # radius_regions = list(range(self.r1,self.r2+1))

        # -----0507:mean_dist----------
        radius_regions = np.logspace(0,(np.log10(radius)), self.nbins_r+1)-1

        radius_bins = np.ones(dist_mat.shape) * (-1)
        for i in range(len(dist_mat)):
            dis = dist_mat[i]
            for idx in range(len(radius_regions)):
                if dis<=radius_regions[idx]:
                    radius_bins[i] = idx
                    break
        radius_bins = squareform(radius_bins) # the cloumn idx of every 'other' point
        assert angle_bins.shape == radius_bins.shape,"Error:angle_bins.shape != radius_bins.shape"

        # get scd-60dims for every point,first make matrix row=angles,column=radius,then flatten,and assign to scd[i]
        scd_mat = np.zeros((angle_bins.shape[0],self.desc_size))
        for i in range(angle_bins.shape[0]):
            scd = np.zeros((self.nbins_theta,self.nbins_r)) # 12*5
            angles = angle_bins[i,:]
            radiuses = radius_bins[i,:]
            cor_list = list(zip(angles,radiuses))
            for cor in cor_list:
                x = int(cor[0]) # angle
                y = int(cor[1]) # radius
                if 0<=cor[1]<self.nbins_r:
                    if x>11:
                        x=11
                    scd[x,y] += 1
            scd_mat[i,:] = scd.flatten()
        scd_mat = scd_mat/np.sum(scd_mat)
        return scd_mat

def scd(view_dic_list):
    print ('define_crv')
    point_list1 = view_dic_list[0]['point_list']
    point_list2 = view_dic_list[1]['point_list']
    crv1 = BSpline.Curve()
    crv1.degree = 4
    crv1.ctrlpts = point_list1
    crv1.knotvector = utilities.generate_knot_vector(crv1.degree, len(crv1.ctrlpts)) # len(knotvector) = degree + len(ctrlpts) + 1
    u_list = crv1.knotvector[2:-3]

    crv2 = BSpline.Curve()
    crv2.degree = 4
    crv2.ctrlpts = point_list2
    crv2.knotvector = utilities.generate_knot_vector(crv2.degree, len(crv2.ctrlpts))
    v_list = crv2.knotvector[2:-3]

    k1 = view_dic_list[0]['k']
    cor_pts1 = []
    for coordinate in point_list1:
        temp = pix2cor(coordinate, k1)
        cor_pts1.append(temp)

    k2 = view_dic_list[1]['k']
    cor_pts2 = []
    for coordinate in point_list2:
        temp = pix2cor(coordinate, k2)
        cor_pts2.append(temp)

    cor_pts1 = np.array(cor_pts1)
    cor_pts2 = np.array(cor_pts2)
    print ('Create Shape context...')
    sc_cls1 = ShapeContext(cor_pts1, u_list, crv1)
    _,mean_dist1 = sc_cls1.compute_dist()
    sc_cls2 = ShapeContext(cor_pts2, v_list, crv2)
    _,mean_dist2 = sc_cls2.compute_dist()
    radius = np.mean(np.array([mean_dist1,mean_dist2]))
    scd1 = sc_cls1.compute_histogram(radius)
    scd2 = sc_cls2.compute_histogram(radius)
    return scd1,scd2

