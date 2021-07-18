import os
import numpy as np
import math
from lib.Basic import *
from lib.DTW_Match import *
from lib.GA import *
from lib.shape_context import *
import time
import json
import yaml
# import sys
# cnt = sys.argv[1]

def Reconstruction_for_one_pair(view_dic_list, id_grp, OUT_ROOT, cfg, name):
    localtime = time.asctime(time.localtime(time.time()))
    view1_name = id_grp[0] # view1_dic
    view2_name = id_grp[1] # view2_dic
    img_out_dir = os.path.join(OUT_ROOT, name+'-{},{}'.format(str(view1_name),str(view2_name)))
    RESULT_PATH = os.path.join(OUT_ROOT,'result.txt')
    LOG_PATH = os.path.join(OUT_ROOT,'log.txt')
    MATCH_PATH = os.path.join(img_out_dir, name+'_match.txt')
    PLIST_TXT_PATH = os.path.join(img_out_dir,'P_list.txt')
    if not os.path.exists(img_out_dir):
        os.makedirs(img_out_dir)
    
    print(name)
    with open(LOG_PATH,'a') as f:
        f.write(name+'\n')
    with open(RESULT_PATH,'a') as f:
        f.write(name+'\n')

    ori_point_list1 = view_dic_list[0]['point_list']
    ori_point_list2 = view_dic_list[1]['point_list']

    # coarse_calibration
    best_genome = Calibration(cfg['Optimizer'],view_dic_list,LOG_PATH,img_out_dir,state='coarse',iter_num=100)
    print (f'{name} match {view1_name} and {view2_name}:{len(view_dic_list[0]["point_list"])} vs {len(view_dic_list[1]["point_list"])}')
    with open(MATCH_PATH,'a') as f:
        f.write('localtime:{}\n' .format(str(localtime)))
        f.write('match {} and {}:{} vs {}\n'.format(str(view1_name), str(view2_name), \
                                                    str(len(view_dic_list[0]['point_list'])),
                                                    str(len(view_dic_list[1]['point_list']))))
    
    new_view_dic_list = Match(cfg['Match'], view_dic_list, best_genome)
    # fine calibration
    fine_best_genome = Calibration(cfg['Optimizer'],new_view_dic_list,LOG_PATH,img_out_dir,state='refine',iter_num=100)

    Reconstruction('reconstruction',fine_best_genome,img_out_dir,PLIST_TXT_PATH,ori_point_list1,ori_point_list2, new_view_dic_list)


def Reconstruction(state,the_genome,img_out_path,PLIST_TXT_PATH, ori_point_list1, ori_point_list2, view_dic_list):
    individual = Individual(the_genome)
    epsilon,all_TwoD_list,P_list,loss_list1,loss_list2,cors1,cors2 = individual.get_epsilon(view_dic_list,state)

    with open(PLIST_TXT_PATH,'w') as f:
        localtime = time.asctime(time.localtime(time.time()))
        f.write('localtime:{}\n'.format(str(localtime)))
        f.write('P_list:\n'+str(P_list))

    for view_index in range(len(view_dic_list)):
        view1_dic = view_dic_list[view_index]
        coordinate_list1 = eval('ori_point_list'+str(view_index+1))
        k1 = view1_dic['k']
        ori_coor1 = []
        for coordinate in coordinate_list1:
            temp = pix2cor(coordinate,k1)
            ori_coor1.append(temp)

        draw2D(all_TwoD_list[view_index], ori_coor1, view_index+1, img_out_path, epsilon, view1_dic)
    write_3d_pts(img_out_path, P_list)





def Match(cfg, view_dic_list, best_genome):
    if cfg['type'] == 'DTW':
        r_matrix, short_list, long_list,flag = get_r_matrix(view_dic_list,0,1,best_genome)
        if 'scd' in cfg['dist']:
            scd1,scd2 = scd(view_dic_list)
            cost_mat = cost(scd1, scd2)
            shape = cost_mat.shape
            if shape[0] > shape[1]:
                cost_mat = cost_mat.transpose(1,0)
        if 'euc' in cfg['dist']:
            edu_matrix = np.zeros_like(r_matrix)
            tmp_k = view_dic_list[0]['k']
            for s_id in range(len(short_list)):
                for l_id in range(len(long_list)):
                    s_pt = short_list[s_id]
                    l_pt = long_list[l_id]
                    edu_matrix[s_id,l_id] = math.sqrt((s_pt[0]-l_pt[0])**2 + (s_pt[1]-l_pt[1])**2)
            edu_matrix = edu_matrix * tmp_k
        #####--------only SCD matrix: cost_mat shoud be converted into (short,long) shape
        if cfg['dist'] == 'scd':
            total_matrix = cost_mat
        #####--------only Repro matrix-------
        elif cfg['dist'] == 'repro':
            total_matrix = r_matrix
        #####--------only Euclidean matrix-------
        elif cfg['dist'] == 'euc':
            total_matrix = edu_matrix
        ####---------Repro + SCD matrix
        elif cfg['dist'] == 'repro+scd':
            total_matrix = r_matrix + cost_mat
        #####--------SCD + Euclidean---------
        elif cfg['dist'] == 'edu+scd':
            total_matrix = edu_matrix + cost_mat
        ####--------Repro + Euclidean Distance-----
        elif cfg['dist'] == 'repro+euc':
            total_matrix = r_matrix + edu_matrix
        #####----------Repro + SCD + Euclidean-------
        elif cfg['dist'] == 'repro+euc+scd':
            total_matrix = r_matrix + edu_matrix + cost_mat

        pathcost, path = DTW(total_matrix,len(short_list),len(long_list))
        final_path, new_short_list, new_long_list = generate_final_path(pathcost,path,short_list,long_list)
        refine_index_list = np.random.choice(len(new_short_list),100,replace=False)
        new_pts1, new_pts2 = None, None

        if flag == 0:
            new_pts1 = new_short_list
            new_pts2 = new_long_list
        elif flag == 1:
            new_pts1 = new_long_list
            new_pts2 = new_short_list
        else:
            raise ValueError('flag is not legal! flag={}'.format(str(flag)))
        '''update'''
        # ----------build refine list, update view_dic_list------------
        refine_index_list = np.random.choice(len(new_pts1),100,replace=False)
        view_dic_list[0]['point_list'] = new_pts1
        view_dic_list[1]['point_list'] = new_pts2
        view_dic_list[0]['refine_point_list'] = [new_pts1[idx] for idx in refine_index_list]
        view_dic_list[1]['refine_point_list'] = [new_pts2[idx] for idx in refine_index_list]
    else:
        raise NotImplementedError
    return view_dic_list


def Calibration(cfg,view_dic_list,LOG_PATH,img_out_path,state,iter_num=100):
    assert cfg["type"] in ['GA'], f'{cfg["type"]} not implemented!'
    params = cfg['params']
    start = time.clock()
    pc = params['pc']
    pm = params['pm']
    POPULATION_SIZE = params['POPULATION_SIZE']  # population_size
    Min_delta_theta = params['Min_delta_theta']
    Max_delta_theta = params['Max_delta_theta']
    Min_delta_O = params['Min_delta_O']
    Max_delta_O = params['Max_delta_O']
    Min_delta_I = params['Min_delta_I']
    Max_delta_I = params['Max_delta_I']
    out_file_name = 'GA_coarse.txt' if state.lower() == 'coarse' else 'GA_refine.txt'
    title = 'Coarse EPOCH' if state.lower() == 'coarse' else 'Refine EPOCH'
    best_individual_genome_list=[]
    best_epsilon_list = []
    min_epsilon = float('inf')
    GA_dict = {}
    state_of_fitness = None if state.lower()=='coarse' else 'refine'

    population_list = species_origin(POPULATION_SIZE,Min_delta_I,Max_delta_I,Min_delta_O,Max_delta_O,Min_delta_theta,Max_delta_theta)
    counter = 0
    while (counter<iter_num):
        print(f'{title}{counter}')
        fitness_list,best_fitness,best_individual = get_fitness(population_list,view_dic_list[0],view_dic_list[1],state_of_fitness)  # compute fitness
        best_individual_genome_list.append(best_individual.genome)
        best_epsilon_list.append(1/best_fitness)
        population_list = select(population_list,fitness_list,POPULATION_SIZE) # only half of the POPULATION_SIZE, they are going to be parents.
        population_list = crossover(POPULATION_SIZE,population_list,pc) # random choose one to cross with the best one until len(pop) = POPULATION_SIZE
        population_list = mutate(population_list,pm,Min_delta_I,Max_delta_I,Min_delta_O,Max_delta_O,Min_delta_theta,Max_delta_theta)
        print(1/best_fitness,best_individual.genome)
        min_epsilon = np.min(best_epsilon_list)
        GA_dict[counter] = min_epsilon
        counter += 1
        with open(LOG_PATH,'a') as f:
            strings = [f'{title}{counter}\n',str(1/best_fitness)+' ',str(best_individual.genome)+'\n']
            f.writelines(strings)
    out_file = os.path.join(img_out_path,out_file_name)
    with open(out_file,'w') as f:
        f.write(str(GA_dict))

    min_epsilon = np.min(best_epsilon_list)
    min_index = np.argmin(best_epsilon_list)
    the_genome = best_individual_genome_list[min_index]
    print('Min_epsilon:{}'.format(min_epsilon))
    print('The genome:{}'.format(the_genome))
    end = time.clock()
    cost_time = end - start
    print('Running time: %s Seconds' % (cost_time))
    return the_genome


def deal_one_name(name, view_id_list, INFOMATION_PATH, OUT_ROOT, cfg):
    print(name)
    for id_grp in view_id_list:
        with open(INFOMATION_PATH, encoding='utf-8') as file:
            dataset = json.loads(file.read())
        view1_name = id_grp[0] # view1_dic
        view2_name = id_grp[1] # view2_dic
        view_dic_list = []
        view_dic_list.append(dataset[name]['view'+str(view1_name)+'_dic'])
        view_dic_list.append(dataset[name]['view'+str(view2_name)+'_dic'])
        Reconstruction_for_one_pair(view_dic_list, id_grp, OUT_ROOT, cfg, name)

def main(cfg_path):
    with open(cfg_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    INFOMATION_PATH = config['data']['list_path']
    OUT_ROOT = config['OUT_ROOT']
    if not os.path.exists(OUT_ROOT):
        os.makedirs(OUT_ROOT)
    view_and_name_dict = config['data']["data_list"]
    for name in view_and_name_dict.keys():
        view_id_list = view_and_name_dict[name]
        deal_one_name(name, view_id_list, INFOMATION_PATH, OUT_ROOT, config)

if __name__ == "__main__":
    main('./config.yaml')


