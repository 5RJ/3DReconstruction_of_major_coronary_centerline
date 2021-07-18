print('__file__={0:<35} | __name__={1:<20} | __package__={2:<20}'.format(__file__,__name__,str(__package__)))

import numpy as np
from .Basic import *
import json


def get_r_matrix(view_dic,view1_name,view2_name,genome):
    '''
    :param view_dic: correspond to a person
    :param view1_name: eg. 'view1_dic'---->0
    :param view2_name: eg. 'view2_dic'---->1
    :return: r_matrix
    '''
    print('get_r_matrix...')
    D_sid1 = view_dic[int(view1_name)]['D_sid']
    D_sod1 = view_dic[int(view1_name)]['D_sod']
    k1 = view_dic[int(view1_name)]['k']
    alpha1 = view_dic[int(view1_name)]['alpha']
    beta1 = view_dic[int(view1_name)]['beta']
    point_list1 = view_dic[int(view1_name)]['point_list']
    M1 = getMmatrix(alpha1, beta1)
    F1, O1 = getF_O(D_sid1, D_sod1, M1)
    F1 = tuple(F1.squeeze())


    D_sid2 = view_dic[int(view2_name)]['D_sid']
    D_sod2 = view_dic[int(view2_name)]['D_sod']
    k2 = view_dic[int(view2_name)]['k']
    alpha2 = view_dic[int(view2_name)]['alpha']
    beta2 = view_dic[int(view2_name)]['beta']
    point_list2 = view_dic[int(view2_name)]['point_list']
    M2 = getMmatrix(alpha2, beta2)
    F2, O2 = getF_O(D_sid2, D_sod2, M2)
    F2 = tuple(F2.squeeze())

    delta_theta1 = [genome[0], genome[1], genome[2]]
    delta_O1 = np.array([genome[3], genome[4], genome[5], 1], dtype=float).reshape(4, 1)
    delta_I = np.array([genome[6], genome[7], genome[8], 1], dtype=float).reshape(4, 1)
    delta_I_1 = np.array([genome[9], genome[10], genome[11], 1], dtype=float).reshape(4, 1)
    delta_theta1_x = delta_theta1[0] / 180 * np.pi
    delta_theta1_y = delta_theta1[1] / 180 * np.pi
    delta_theta1_z = delta_theta1[2] / 180 * np.pi

    R_theta1 = getRmatrix(delta_theta1_x, delta_theta1_y, delta_theta1_z, 1)
    R_theta_inverse1 = np.squeeze(np.mat(R_theta1)).I
    delta_theta2 = [genome[12], genome[13], genome[14]]
    delta_O2 = np.array([genome[15], genome[16], genome[17], 1], dtype=float).reshape(4, 1)

    delta_theta2_x = delta_theta2[0] / 180 * np.pi
    delta_theta2_y = delta_theta2[1] / 180 * np.pi
    delta_theta2_z = delta_theta2[2] / 180 * np.pi

    R_theta2 = getRmatrix(delta_theta2_x, delta_theta2_y, delta_theta2_z, 1)
    R_theta_inverse2 = np.squeeze(np.mat(R_theta2)).I

    if len(point_list1) > len(point_list2):
        m = len(point_list2)
        n = len(point_list1)
        short_list = point_list2
        long_list = point_list1
        flag = 1 # the index of short one
        print('m(short):point_list2\nn(long):point_list1')
    else:
        m = len(point_list1)
        n = len(point_list2)
        short_list = point_list1
        long_list = point_list2
        flag = 0
        print ('m(short):point_list1\nn(long):point_list2')

    #--------generate r_matrix---------------
    r_matrix = np.zeros((m,n))

    for i in range(m):
        for j in range(n):
            if flag == 0:
                p1 = point_list1[i]
                p2 = point_list2[j]
            elif flag ==1:
                p1 = point_list1[j]
                p2 = point_list2[i]
            else:
                raise ValueError('flag is not legal! flag={}'.format(str(flag)))

            P1 = get3Dposition2(O1, M1, k1, p1, delta_O1,R_theta1,delta_I_1)
            P2 = get3Dposition2(O2, M2, k2, p2, delta_O2,R_theta2,delta_I)


            P1 = tuple(P1.squeeze())
            P2 = tuple(P2.squeeze())
            P, isParallel, C1, C2 = get_finalP(F1[:3], P1[:3], F2[:3], P2[:3])
            P = [P]

            TwoD_list1, loss1, _ = ray_tracing(F1, O1, P, M1, R_theta_inverse1, delta_O1, [p1], k1,1, delta_I_1)
            TwoD_list2, loss2, _ = ray_tracing(F2, O2, P, M2, R_theta_inverse2, delta_O2, [p2], k2,2, delta_I)

            loss = (loss1+loss2)/2
            r_matrix[i,j] = loss

    # print('r_matrix:\n{}'.format(str(r_matrix)))
    return r_matrix,short_list,long_list,flag

#-------------MVM------------------
def DTW(r,m,n):
    print ('start DTW...')
    pathcost = np.zeros((m,n))
    path = np.zeros((m,n,2))
    for i in range(m):
        for j in range(n):
            if i == 0 and j == 0:
                pathcost[i,j] = r[i,j]
                path[i,j] = [0,0]

            elif i == 0  and j>0:
                pathcost[i,j] = r[i,j] + pathcost[i,j-1]
                path[i,j] = [i,j-1]

            elif i>0 and j==0:
                pathcost[i,j] = r[i,j] + pathcost[i-1,j]
                path[i,j] = [i-1,j]

            else:
                candidate = [pathcost[i-1,j],pathcost[i,j-1],pathcost[i-1,j-1]]

                index = np.argmin(candidate)
                minium = min(candidate)
                pathcost[i,j] = r[i,j] + minium
                if index == 0:
                    path[i,j] = [i-1,j]

                elif index == 1:
                    path[i,j] = [i,j-1]

                elif index == 2:
                    path[i,j] = [i-1,j-1]
                else:
                    print ('Index_Eroor! index:{}'.format(index))

    # print ('pathcost:\n{}'.format(str(pathcost)))
    # print ('path:\n{}'.format(str(path)))
    return pathcost,path


def generate_final_path(pathcost,path,short_list,long_list):
    print ('generate final path...')
    m = len(short_list)
    n = len(long_list)
    final_path = []
    final_path.append([m-1,n-1])
    index = (path[-1,-1]).astype(np.int32).tolist()

    while index[0] >0 or index[1]>0:
        final_path.append(index)
        index = (path[index[0],index[1]]).astype(np.int32).tolist()
    final_path.append(index) # append [0,0]
    # print ('final_path:\n{}'.format(str(final_path)))

    #add sort in ele[1]
    final_path.sort(key=lambda elem:elem[1])

    # reconstruct 2 list
    new_long_list = []
    new_short_list = []
    for item in final_path:
        new_short_list.append(short_list[item[0]])
        new_long_list.append(long_list[item[1]])
    assert len(new_short_list) == len(new_long_list),\
        "The length of new_short_list and of new_long_list don't match!"+str(len(new_short_list))+'!='+str(len(new_long_list))

    return final_path,new_short_list,new_long_list



