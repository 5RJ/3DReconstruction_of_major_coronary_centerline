print('__file__={0:<35} | __name__={1:<20} | __package__={2:<20}'.format(__file__,__name__,str(__package__)))

import numpy as np
import random
from .Basic import *


class Individual():
    def __init__(self,genome):
        # delta_theta,delta_O,delta_I
        self.genome = genome
        self.epsilon = 0

    def set_genome(self,new_genome):
        self.genome = new_genome

    def get_epsilon(self,view_dic_list,state=None):
        # constant params
        view1_dic = view_dic_list[0]
        coordinate_list1 = view1_dic['coordinate_list']
        alpha1 = view1_dic['alpha']
        beta1 = view1_dic['beta']
        D_sid1 = view1_dic['D_sid']
        D_sod1 = view1_dic['D_sod']
        k1 = view1_dic['k']
        view1 = 1  # view1_dic['view1']
        view2_dic = view_dic_list[1]
        coordinate_list2 = view2_dic['coordinate_list']
        alpha2 = view2_dic['alpha']
        beta2 = view2_dic['beta']
        D_sid2 = view2_dic['D_sid']
        D_sod2 = view2_dic['D_sod']
        k2 = view2_dic['k']
        view2 = 2  #view2_dic['view2']
        if state == 'reconstruction':
            coordinate_list1 = view1_dic['point_list']
            coordinate_list2 = view2_dic['point_list']
        elif state == 'refine':
            coordinate_list1 = view1_dic['refine_point_list']
            coordinate_list2 = view2_dic['refine_point_list']


        # -----------(∆θ1,∆θ2,∆O_P_1,∆O_P_2,∆O_1,∆O_2 )--------------
        delta_theta1 = [self.genome[0],self.genome[1],self.genome[2]]
        delta_O1 = np.array([self.genome[3],self.genome[4],self.genome[5],1],dtype=float).reshape(4,1)
        delta_I = np.array([self.genome[6],self.genome[7],self.genome[8],1],dtype=float).reshape(4,1)
        delta_I_1 = np.array([self.genome[9],self.genome[10],self.genome[11],1],dtype=float).reshape(4,1)
        delta_theta1_x = delta_theta1[0] /180 * np.pi
        delta_theta1_y = delta_theta1[1] /180 * np.pi
        delta_theta1_z = delta_theta1[2] /180 * np.pi

        R_theta1 = getRmatrix(delta_theta1_x,delta_theta1_y,delta_theta1_z,1)
        R_theta_inverse1 = np.squeeze(np.mat(R_theta1)).I
        delta_theta2 = [self.genome[12],self.genome[13],self.genome[14]]
        delta_O2 = np.array([self.genome[15],self.genome[16],self.genome[17],1],dtype=float).reshape(4,1)

        delta_theta2_x = delta_theta2[0] /180 * np.pi
        delta_theta2_y = delta_theta2[1] /180 * np.pi
        delta_theta2_z = delta_theta2[2] /180 * np.pi

        R_theta2 = getRmatrix(delta_theta2_x,delta_theta2_y,delta_theta2_z,1)
        R_theta_inverse2 = np.squeeze(np.mat(R_theta2)).I


        # get P1_list
        F1 = None
        P1_list = []
        M1 = getMmatrix(alpha1, beta1)
        F1, O1 = getF_O(D_sid1, D_sod1, M1)
        F1 = tuple(F1.squeeze())
        for i in range(len(coordinate_list1)):
            P1 = get3Dposition2(O1, M1, k1, coordinate_list1[i], delta_O1,R_theta1,delta_I_1)
            P1 = tuple(P1.squeeze())
            P1_list.append(P1)

        # get P2_list
        F2 = None
        P2_list = []
        M2 = getMmatrix(alpha2, beta2)
        F2, O2 = getF_O(D_sid2, D_sod2, M2)
        F2 = tuple(F2.squeeze())
        for i in range(len(coordinate_list2)):
            P2 = get3Dposition2(O2, M2, k2, coordinate_list2[i],delta_O2,R_theta2,delta_I)
            P2 = tuple(P2.squeeze())
            P2_list.append(P2)

        # get final P with 2 views
        P_list = []
        assert len(P1_list) == len(P2_list), 'P1_list.length:' + str(len(P1_list)) + ' != P2_list.length:' + str(len(P2_list))
        for i in range(len(P1_list)):
            point, isParallel, C1, C2 = get_finalP(F1[:3], P1_list[i][:3], F2[:3], P2_list[i][:3])
            point = tuple(point)
            P_list.append(point)

        # ray_tracing
        TwoD_list1,loss1,loss_list1 = ray_tracing(F1,O1,P_list,M1,R_theta_inverse1,delta_O1,coordinate_list1,k1,view1,delta_I_1)
        TwoD_list2,loss2,loss_list2 = ray_tracing(F2,O2,P_list,M2,R_theta_inverse2,delta_O2,coordinate_list2,k2,view2,delta_I)

        all_TwoD_list = []
        all_TwoD_list.append(TwoD_list1)
        all_TwoD_list.append(TwoD_list2)

        # caculate epsilon
        loss = loss1 + loss2
        self.epsilon = loss/(2*len(P_list))

        if state == 'reconstruction':
            return self.epsilon,all_TwoD_list,P_list,loss_list1,loss_list2,coordinate_list1,coordinate_list2
        else:
            return self.epsilon


def get_fitness(population_list,view1_dic,view2_dic,state=None):
    all_fitness = 0
    fitness_list = []
    best_fitness = 0
    best_individual = None
    for i in range(len(population_list)):
        individual = population_list[i]
        epsilon = individual.get_epsilon([view1_dic,view2_dic],state)
        fitness = 1/ epsilon
        if fitness < 0:
            fitness = 0
        fitness_list.append(fitness)
        all_fitness += fitness
        # TODO:What if there are multi best individuals?
        if fitness > best_fitness:
            best_fitness = fitness
            best_individual = individual
    fitness_list = fitness_list / all_fitness # normalize
    return fitness_list,best_fitness,best_individual

def cumsum(fitness_list):
    acum_fitness_list = []
    tmp = 0
    for i in range(len(fitness_list)):
        tmp += fitness_list[i]
        acum_fitness_list.append(tmp)
    return acum_fitness_list

def species_origin(POPULATION_SIZE,Min_delta_I,Max_delta_I,Min_delta_O,Max_delta_O,Min_delta_theta,Max_delta_theta):
    population_list = []
    for i in range(POPULATION_SIZE):
        delta_theta1 = [random.uniform(Min_delta_theta,Max_delta_theta) for l in range(3)]
        delta_O1 = [random.uniform(Min_delta_O,Max_delta_O) for l in range(3)]
        delta_I = [random.uniform(Min_delta_I,Max_delta_I) for l in range(3)]
        delta_I_1 = [random.uniform(Min_delta_I,Max_delta_I) for l in range(3)]
        delta_theta2 = [random.uniform(Min_delta_theta,Max_delta_theta) for l in range(3)]
        delta_O2 = [random.uniform(Min_delta_O,Max_delta_O) for l in range(3)]


        genome = []
        genome.extend(delta_theta1)
        genome.extend(delta_O1)
        genome.extend(delta_I)
        genome.extend(delta_I_1)
        genome.extend(delta_theta2)
        genome.extend(delta_O2)
        population_list.append(Individual(genome))
    return population_list


def select(population_list,fitness_list,POPULATION_SIZE): # select half of the POPULATION_SIZE to be parents
    '''
    fitness_list: normalized, i.e. sum(fitness_list) == 1
    '''
    new_population_list = []
    new_population_list.append(population_list[np.argmax(fitness_list)])
    acum_fitness_list = cumsum(fitness_list)
    ms = []
    for i in range(int(POPULATION_SIZE/2-1)):
        ms.append(random.random())
    ms.sort()
    fitin = 0
    newin = 0
    while newin < len(ms):
        if (ms[newin]<acum_fitness_list[fitin]):
            new_population_list.append(population_list[fitin])
            newin += 1
        else:
            fitin += 1

    return new_population_list

def crossover(POPULATION_SIZE,population_list,pc):
    # use best-genome to crossover
    best_genome = population_list[0].genome
    while len(population_list)< POPULATION_SIZE:
        rate_1 = random.random()
        rate_2 = random.random()
        rate_3 = random.random()
        index = random.randint(1,POPULATION_SIZE/2-1)
        parent= population_list[index]
        new_genome = [] # for all

        # update delta_O2
        temp_1 = best_genome[15] * rate_1 + parent.genome[15]*(1-rate_1)
        temp_2 = best_genome[16] * rate_2 + parent.genome[16]*(1-rate_2)
        temp_3 = best_genome[17] * rate_3 + parent.genome[17]*(1-rate_3)
        new_genome.extend([temp_1,temp_2,temp_3])

        # update delta_theta2
        temp_1 = best_genome[12] * rate_1 + parent.genome[12]*(1-rate_1)
        temp_2 = best_genome[13] * rate_2 + parent.genome[13]*(1-rate_2)
        temp_3 = best_genome[14] * rate_3 + parent.genome[14]*(1-rate_3)
        temp_genome = [temp_1, temp_2, temp_3]
        temp_genome.extend(new_genome)
        new_genome = temp_genome

        # update delta_I_1
        temp_1 = best_genome[9] * rate_1 + parent.genome[9]*(1-rate_1)
        temp_2 = best_genome[10] * rate_2 + parent.genome[10]*(1-rate_2)
        temp_3 = best_genome[11] * rate_3 + parent.genome[11]*(1-rate_3)
        temp_genome = [temp_1, temp_2, temp_3]
        temp_genome.extend(new_genome)
        new_genome = temp_genome

        # update delta_I
        temp_1 = best_genome[6] * rate_1 + parent.genome[6]*(1-rate_1)
        temp_2 = best_genome[7] * rate_2 + parent.genome[7]*(1-rate_2)
        temp_3 = best_genome[8] * rate_3 + parent.genome[8]*(1-rate_3)
        temp_genome = [temp_1, temp_2, temp_3]
        temp_genome.extend(new_genome)
        new_genome = temp_genome

        # update delta_O
        temp_1 = best_genome[3] * rate_1 + parent.genome[3]*(1-rate_1)
        temp_2 = best_genome[4] * rate_2 + parent.genome[4]*(1-rate_2)
        temp_3 = best_genome[5] * rate_3 + parent.genome[5]*(1-rate_3)
        temp_genome = [temp_1,temp_2,temp_3]
        temp_genome.extend(new_genome)
        new_genome = temp_genome

        # update delta_theta
        temp_1 = best_genome[0] * rate_1 + parent.genome[0]*(1-rate_1)
        temp_2 = best_genome[1] * rate_2 + parent.genome[1]*(1-rate_2)
        temp_3 = best_genome[2] * rate_3 + parent.genome[2]*(1-rate_3)
        temp_genome = [temp_1,temp_2,temp_3]
        temp_genome.extend(new_genome)
        new_genome = temp_genome

        assert len(new_genome) == 18, 'crossover:len(new_genome)!=18 '
        new_individual = Individual(new_genome)
        population_list.append(new_individual)
    return population_list


def mutate(population_list,pm,Min_delta_I,Max_delta_I,Min_delta_O,Max_delta_O,Min_delta_theta,Max_delta_theta):
    # prob of mutation, randomly generate a value to decide wheter to conduct mutation. If conduct mutation, 2 plans but both ensure the
    # genome after mutation is still in the range.
    k = 0.2 # constant for mutation
    M_list = [Max_delta_theta,Min_delta_theta,Max_delta_O,Min_delta_O,Max_delta_I,Min_delta_I,\
              Max_delta_I,Min_delta_I,Max_delta_theta,Min_delta_theta,Max_delta_O,Min_delta_O,]
    N = 6 # the number of items contained in genome
    for i in range(len(population_list)):
        new_genome = []
        for j in range(N): # j = 0,1,2,3
            Maxium = M_list[2*j]
            Minium = M_list[2*j+1]
            temp_item = population_list[i].genome[(3*j):(3*j)+3]
            # METHOD1:mutate in the unit of delta
            if random.random() < pm:
                rate = random.random()
                if rate > 0.5:
                    for l in range(len(temp_item)):
                        temp_item[l] = temp_item[l] + k * (Maxium - temp_item[l]) * random.random()
                else:
                    for l in range(len(temp_item)):
                        temp_item[l] = temp_item[l] - k * (temp_item[l] - Minium) * random.random()

            # METHOD2:mutate in the unit of single genome
            # if random.random() < pm:
            #     for l in range(len(temp_item)):
            #             rate = random.random()
            #             if rate >0.5:
            #                 temp_item[l] = temp_item[l] + k*(Maxium-temp_item[l])*random.random()
            #             else:
            #                 temp_item[l] = temp_item[l] - k * (temp_item[l] - Minium) * random.random()

            new_genome.extend(temp_item)
        population_list[i].set_genome(new_genome)



    return population_list

