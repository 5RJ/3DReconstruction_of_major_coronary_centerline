print('__file__={0:<35} | __name__={1:<20} | __package__={2:<20}'.format(__file__,__name__,str(__package__)))

import numpy as np
import matplotlib.pyplot as plt
import os


def Normalize0908(array):
    array = np.array(array)
    max_row = np.max(array,axis=0,keepdims=True)
    min_row = np.min(array,axis=0,keepdims=True)
    array = (array - min_row)/(max_row - min_row)
    max_col = np.max(array,axis=1,keepdims=True)
    min_col = np.min(array,axis=1,keepdims=True)
    array = (array - min_col)/(max_col - min_col)
    return array

def getMmatrix(alpha, beta):
    M = np.eye(4)
    pi = np.pi
    cos_beta = np.cos(beta * pi / 180.0)
    sin_beta = np.sin(beta * pi / 180.0)
    cos_alpha = np.cos(alpha * pi / 180.0)
    sin_alpha = np.sin(alpha * pi / 180.0)
    R = np.array([[cos_beta,  sin_alpha*sin_beta, cos_alpha*sin_beta],
                  [0,         cos_alpha,           -sin_alpha ],
                  [-sin_beta, sin_alpha*cos_beta,cos_alpha*cos_beta]])
    M[:3, :3] = R

    return M     # M.shape=[4,4]

def getRmatrix(delta_theta_x,delta_theta_y,delta_theta_z,state):
    R_theta = np.eye(4)
    rotate_x = np.array([[1, 0, 0],
                         [0, np.cos(delta_theta_x), np.sin(delta_theta_x)],
                         [0, -1 * np.sin(delta_theta_x), np.cos(delta_theta_x)]])

    rotate_y = np.array([[np.cos(delta_theta_y), 0, -1 * np.sin(delta_theta_y)],
                         [0, 1, 0],
                         [np.sin(delta_theta_y), 0, np.cos(delta_theta_y)]])

    rotate_z = np.array([[np.cos(delta_theta_z), np.sin(delta_theta_z), 0],
                         [-1 * np.sin(delta_theta_z), np.cos(delta_theta_z), 0],
                         [0, 0, 1]])
    if state == 1:
        R_theta[:3,:3] = np.dot(rotate_x, np.dot(rotate_y, rotate_z))
    elif state == -1:
        R_theta[:3, :3] = np.dot(rotate_z.T, np.dot(rotate_y.T, rotate_x.T))

    return R_theta  # R_theta.shape=[4,4]


def getF_O(D_sid, D_sod, M):
    F = np.zeros((4, 1))
    F[2, 0] = -D_sod
    F[3, 0] = 1
    F = np.dot(M, F)

    O = np.zeros((4, 1))
    O[2, 0] = D_sid - D_sod
    O[3, 0] = 1
    O = np.dot(M, O)

    return F,O

def pix2cor(pixel,k):
    pix_x = pixel[0]
    pix_y = pixel[1]

    pu = k*(pix_y-256)
    pv = k*(pix_x-256)

    return [pu,pv]

def cor2pix(coor,k):
    pu = coor[0]
    pv = coor[1]

    pix_y = pu/k+256
    pix_x = pv/k+256

    return [pix_x,pix_y]


def get3Dposition2(O, M, k, coordinate,delta_O,R_theta,delta_I,Match=False):
    [pu,pv] = pix2cor(coordinate,k)
    TwoD_coordinate = np.zeros((4, 1))
    TwoD_coordinate[0, 0] = pu
    TwoD_coordinate[1, 0] = pv
    TwoD_coordinate[3, 0] = 1

    if Match:
        P_3Dcoordinate = O + np.dot(M, TwoD_coordinate)
    else:
        P_3Dcoordinate = O + np.dot(R_theta, np.dot(M, (delta_O + TwoD_coordinate))) + delta_I
    assert P_3Dcoordinate.shape == (4,1),'P_3Dcoordinate.shape != (4,1)'
    return P_3Dcoordinate
    #shape(4,1)


def get_finalP(F1, P1, F2, P2):
    '''
        l1 = F1 + t1 * v1
        l2 = F2 + t2 * v2

    '''
    v1 = np.array([P1[i] - F1[i] for i in range(len(F1))])
    v2 = np.array([P2[i] - F2[i] for i in range(len(F2))])
    a = sum(v1*v2) # inner product
    b = sum(v1*v1)
    c = sum(v2*v2)
    d = sum(np.array([F2[i] - F1[i] for i in range(len(F1))])*v1) # F2-F1 projected onto l1
    e = sum(np.array([F2[i] - F1[i] for i in range(len(F1))])*v2) # F2-F1 projected onto l2
    isParallel = False

    if a==0:        # 对应两直线垂直
        t1 = d/b
        t2 = -e/c
    elif abs(a*a - b*c) > 0.001:     # 普通情况，这里因为浮点数的原因不要用等于0
        t1 = (a * e - c * d) / (a * a - b * c)
        t2 = b * t1 / a - d / a
    else:   # 两直线平行，垂足有无数对，通过在任一一条直线上随便指定一个点，另一条直线上的垂足也就随之确定
        isParallel = True
        t1 = 0
        t2 = - d / a

    C1 = [F1[i] + t1 * v1[i] for i in range(len(F1)) ]
    C2 = [F2[i] + t2 * v2[i] for i in range(len(F2)) ]
    point = [(C1[i]+C2[i])/2 for i in range(len(C1))]

    return point, isParallel,C1, C2


def Projection(P,O,F):
    '''
    ray-tracing with F and P, they are 3D points, get the intersection point on detector,
    return P_plane np.shape=(3,1)
    '''
    O0 = np.zeros((3,1))
    O0[:,0] =O[:3,0]
    O = O0

    F0 = np.zeros((3,1))
    F0[:,0] = F[:3]
    F = F0

    P0 = np.zeros((3,1))
    P0[:,0] = P
    P = P0

    vec_FO = O - F
    vec_FP = P - F
    len_FO = np.sqrt(np.sum(vec_FO * vec_FO, 0).item())
    len_FP1 = np.sum(vec_FP *vec_FO,0).item()/len_FO
    n = len_FO / len_FP1
    P_plane = F + n * vec_FP
    return P_plane

def ComputeLoss(ori_coor,new_coor,k):
    '''
    transform the ori pixel coordinate into uv axes ,then compute loss with the new (pu,pv)
    '''
    ori_coor = pix2cor(ori_coor,k)
    ori_coor = np.squeeze(np.array(ori_coor))
    new_coor = np.squeeze(np.array(new_coor))
    return np.sqrt(np.sum(np.square(new_coor-ori_coor),0).item())



def rotate(P_list,alpha,beta):
    alpha1 = -alpha / 180 * np.pi  
    beta1 = -beta / 180 * np.pi  

    alpha = alpha1
    beta = beta1

    rotate_x = np.array([[1, 0, 0, 0],
                         [0, np.cos(alpha), np.sin(alpha), 0],
                         [0, -1 * np.sin(alpha), np.cos(alpha), 0],
                         [0, 0, 0, 1]])
    rotate_y = np.array([[np.cos(beta), 0, -1 * np.sin(beta), 0],
                         [0, 1, 0, 0],
                         [np.sin(beta), 0, np.cos(beta), 0],
                         [0, 0, 0, 1]])

    final_matrix = np.dot(rotate_x, rotate_y) 



    new_P_list = []
    for i in range(len(P_list)):
        P = np.zeros((4, 1))
        P[:3, 0] = P_list[i]
        P[3, 0] = 1
        new_P = tuple(np.dot(final_matrix, P).squeeze())[:3]
        new_P_list.append(new_P)
    return new_P_list

def ray_tracing(F,O,P_list,M,R_theta_inverse,delta_O,coordinate_list,k,view,delta_I,Match=False):
    TwoD_list = []
    loss = 0
    loss_list = []
    for i in range(len(P_list)):
        P_plane = Projection(P_list[i], O, F)
        temp = np.zeros((4,1))
        temp[:3] = P_plane
        temp[3,0] = 1
        P_plane = temp
        if Match:
            P_plane = np.dot(M.T, (P_plane - O ))
        else:
            if view <=2:
                P_plane = np.dot(np.dot(M.T, R_theta_inverse), (P_plane - O - delta_I)) - delta_O
            else:
                P_plane = np.dot(np.dot(M.T,R_theta_inverse),(P_plane-O))-delta_O


        assert P_plane.shape == (4,1),'ray_tracing:P_plane.shape != (4,1)'
        TwoD_point = tuple(P_plane[:2, 0].squeeze())
        TwoD_point = np.squeeze(np.array(TwoD_point)).tolist()
        TwoD_list.append(TwoD_point)
        tmp = ComputeLoss(coordinate_list[i], TwoD_point, k)
        loss += tmp
        loss_list.append(tmp)
    return TwoD_list,loss,loss_list



def draw2D(new_view_list,ori_view_list,view,img_out_path,epsilon,view_dic):
    plt.switch_backend('agg')
    alpha = view_dic['alpha']
    beta = view_dic['beta']
    new_img_path = os.path.join(img_out_path,str(view)+'_'+'{:.4f}'.format(epsilon)+'_'+'({},{})'.format(alpha,beta)+'.png')
    view_list = ori_view_list
    plt.figure(num=1, figsize=(8, 5))

    ax = plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.spines['bottom'].set_position(('data', 0))
    ax.yaxis.set_ticks_position('left')
    ax.spines['left'].set_position(('data', 0))
    ax.set_xlabel('v')
    ax.set_ylabel('u')
    plt.cla()
    ax = plt.gca()
    ax.xaxis.set_ticks_position('top')
    ax.invert_yaxis()

    for i in range(len(new_view_list)):
        ax.scatter(new_view_list[i][1], new_view_list[i][0], label=('P_' + str(i)), marker='x',s=20, c='k',alpha = 1/2)

    for i in range(len(view_list)):
        ax.scatter(view_list[i][1], view_list[i][0], label=(('P2_'if view ==2 else 'P1_') + str(i)), marker='o',alpha=1/3,s=10, c='g' if view == 2 else 'r')


    plt.savefig(new_img_path)

def create_output(point, filename):
    ply_header = '''ply
                format ascii 1.0
                element vertex %(vert_num)d
                property float x
                property float y
                property float z
                property uchar red
                property uchar green
                property uchar blue
                end_header
                \n
                '''

    file = open(filename, 'w', encoding='utf-8')
    file.write(ply_header % dict(vert_num=len(point)))
    file = open(filename, 'a', encoding='utf-8')
    np.savetxt(file, point,fmt='%f %f %f %d %d %d')  # 必须先写入，然后利用write()在头部插入ply header   #  写入点
    file.flush()
    file.close()


def numpy2ply( point_file_path, output_ply_file):
    a = np.load(point_file_path)
    b = np.float32(a)

    # Generate point cloud
    print("\n Creating the output file... \n")
    create_output(b, output_ply_file)

def write_3d_pts(img_out_dir, P_list):
    ply_file_path = os.path.join(img_out_dir, '3d_model.ply')
    xx = []
    yy = []
    zz = []
    for item in P_list:
        xx.append(item[0])
        yy.append(item[1])
        zz.append(item[2])
        
    u = np.array(xx)
    v = np.array(yy)
    x = u
    y = v
    z = np.array(zz)

    point = np.vstack([x, y, z]).T

    one = np.ones((len(x), 3)) * [255,0,0]
    point = np.hstack([point, one])

    point_file_path = os.path.join(img_out_dir, 'point.npy')
    np.save(point_file_path, point)
    numpy2ply(point_file_path, ply_file_path)
    os.remove(point_file_path)