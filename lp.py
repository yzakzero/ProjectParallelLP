#coding=utf-8
import numpy as np
import time
from multiprocessing import Pool
import os, multiprocessing

global affinity_matrix
global label_function


def knn(dataSet, query, k):
    numSamples = dataSet.shape[0]
 
    ## step 1: calculate Euclidean distance
    diff = np.tile(query, (numSamples, 1)) - dataSet
    squaredDiff = diff ** 2
    squaredDist = np.sum(squaredDiff, axis = 1) # sum is performed by row
 
    ## step 2: sort the distance
    sortedDistIndices = np.argsort(squaredDist)
    if k > len(sortedDistIndices):
        k = len(sortedDistIndices)
 
    return sortedDistIndices[0:k]
 
 
# build a big graph (normalized weight matrix)
def buildGraph(MatX, kernel_type, rbf_sigma = None, knn_num_neighbors = None):
    num_samples = MatX.shape[0]

    affinity_matrix = np.zeros((num_samples, num_samples), np.float32)
    if kernel_type == 'rbf':
        if rbf_sigma == None:
            raise ValueError('You should input a sigma of rbf kernel!')
        for i in range(num_samples):
            row_sum = 0.0
            for j in range(num_samples):
                diff = MatX[i, :] - MatX[j, :]
                affinity_matrix[i][j] = np.exp(sum(diff**2) / (-2.0 * rbf_sigma**2))
                row_sum += affinity_matrix[i][j]
            affinity_matrix[i][:] /= row_sum
    elif kernel_type == 'knn':
        if knn_num_neighbors == None:
            raise ValueError('You should input a k of knn kernel!')
        for i in range(num_samples):
            k_neighbors = knn(MatX, MatX[i, :], knn_num_neighbors)
            affinity_matrix[i][k_neighbors] = 1.0 / knn_num_neighbors
    else:
        raise NameError('Not support kernel type! You can use knn or rbf!')
    
    return affinity_matrix
 
 
# label propagation
def labelPropagation(Mat_Label, Mat_Unlabel, labels, kernel_type = 'rbf', rbf_sigma = 1.5, \
                    knn_num_neighbors = 10, max_iter = 500, tol = 1e-3):
    # initialize
    num_label_samples = Mat_Label.shape[0]
    num_unlabel_samples = Mat_Unlabel.shape[0]
    num_samples = num_label_samples + num_unlabel_samples
    labels_list = np.unique(labels)
    num_classes = len(labels_list)
    
    MatX = np.vstack((Mat_Label, Mat_Unlabel))
    clamp_data_label = np.zeros((num_label_samples, num_classes), np.float32)
    for i in range(num_label_samples):
        clamp_data_label[i][labels[i]] = 1.0
    
    label_function = np.zeros((num_samples, num_classes), np.float32)
    label_function[0 : num_label_samples] = clamp_data_label
    label_function[num_label_samples : num_samples] = -1
    
    # graph construction
    affinity_matrix = buildGraph(MatX, kernel_type, rbf_sigma, knn_num_neighbors)
   # print(affinity_matrix)
    # start to propagation
    iter = 0; pre_label_function = np.zeros((num_samples, num_classes), np.float32)
    changed = np.abs(pre_label_function - label_function).sum()

    start = time.time()

    while iter < max_iter and changed > tol:
        if iter % 1 == 0:
            #print ("---> Iteration %d/%d, changed: %f" % (iter+1, max_iter, changed))
            print()
        pre_label_function = label_function
        iter += 1
        
        # propagation
        #label_function = np.dot(affinity_matrix, label_function)







        label_function=MP(affinity_matrix,label_function)
        #label_function = GEMM(affinity_matrix, label_function)
        #label_function = f1(affinity_matrix,label_function)
        #print(label_function)





       # print(type(label_function),affinity_matrix.shape[0])

        # clamp
        label_function[0 : num_label_samples] = clamp_data_label
        
        # check converge
        changed = np.abs(pre_label_function - label_function).sum()

    end = time.time() - start
    print(end)


    # get terminate label of unlabeled data
    unlabel_data_labels = np.zeros(num_unlabel_samples)
    print(num_unlabel_samples)
    for i in range(num_unlabel_samples):

        unlabel_data_labels[i] = np.argmax(label_function[i+num_label_samples])
    
    return unlabel_data_labels

#GEMM
def GEMM(affinity_matrix,label_function):
    c = np.zeros((affinity_matrix.shape[0], label_function.shape[1]))
    for i in range(0, affinity_matrix.shape[0]):
        for j in range(0, label_function.shape[1], 2):
            temp_m0n0 = 0
            temp_m0n1 = 0
            # c[i][j + 0] = 0
            # c[i][j + 1] = 0
            # c[i][j + 2] = 0
            # c[i][j + 3] = 0

            for k in range(0, affinity_matrix.shape[1]):
                temp_m0 = affinity_matrix[i + 0][k]
                #temp_m1 = affinity_matrix[i + 1][k]

                temp_n0 = label_function[k][j + 0]
                temp_n1 = label_function[k][j + 1]

                temp_m0n0 += temp_m0 * temp_n0
                temp_m0n1 += temp_m0 * temp_n1

            c[i + 0][j + 0] = temp_m0n0
            c[i + 0][j + 1] = temp_m0n1
                #c[i][j + 0] += affinity_matrix[i][k] * label_function[k][j + 0]
                #c[i][j + 1] += affinity_matrix[i][k] * label_function[k][j + 1]
                #c[i][j + 2] += affinity_matrix[i][k] * label_function[k][j + 2]
                #c[i][j + 3] += affinity_matrix[i][k] * label_function[k][j + 3]

    return c





# 普通并行
def f (affinity_matrix,label_function,corei,core):
    M = affinity_matrix.shape[0]
    N = label_function.shape[1]
    K = affinity_matrix.shape[1]

    C = np.zeros((M,N))

    prange = int(M / core)
    print(M)
    for m in range (corei*prange,(corei+1)*prange):
        for n in range (0,N):
            #temp_m0n0=0

           # C[m][n + 0] = 0
           # C[m][n + 1] = 0

            for k in range (K):
                #temp_m0 = affinity_matrix[m + 0][k]
                #temp_n0 = label_function[k][n + 0]
                #temp_m0n0 += temp_m0*temp_n0

                C[m][n + 0] += affinity_matrix[m+0][k] * label_function[k][n + 0]
               # C[m][n + 1] += affinity_matrix[m][k] * label_function[k][n + 1]
        #C[m][n + 0] = temp_m0n0

    print(os.getpid())
    return C


# 原始方法
def f1 (affinity_matrix,label_function):
    c = np.zeros((affinity_matrix.shape[0], label_function.shape[1]))
    a0=int(affinity_matrix.shape[0])
    l1=int(label_function.shape[1])
    a1=int(affinity_matrix.shape[1])
    for i in range (0,a0):
        for j in range (0,l1):
            for k in range (0,a1):
                c[i][j] = c[i][j] + affinity_matrix[i][k] * label_function[k][j]
 #   print(os.getpid())
    return c



#并行
def MP(affinity_matrix,label_function):
    #affinity_matrix=affinity_matrix
  # global c
   #c = np.zeros((affinity_matrix.shape[0], label_function.shape[1]))
   # print('多进程执行')  # 并行执行
    #pool = Pool(multiprocessing.cpu_count())  # 创建拥有4个进程数量的进程池
    #print(os.getpid())
    core=2
    pool=multiprocessing.Pool(core)

#    params = [A(i, j) for i in range(10) for j in range(5)]
    # #    pool.map(f, params)

    # r1 = pool.apply_async(f2, (affinity_matrix, label_function))
    # #res = pool.apply_async(os.getpid, ())
    # m_result=[pool.apply_async(os.getpid,()) for i in range(core)]

    # for i in range(core):
    #     r = pool.apply_async(f, (affinity_matrix, label_function,i,core))
       # r1 = pool.apply_async(f1, (affinity_matrix, label_function))
       # r2 = pool.apply_async(f2, (affinity_matrix, label_function))
        #print(i)

    sub_results = [pool.apply_async(f, (affinity_matrix, label_function,i,core)) for i in range(core)]
    #print()
    complete_result = 0
    print(sub_results)
    for sub in sub_results:
        #print(sub.get())
        complete_result += sub.get()


    #result=pool.apply(f,args=(affinity_matrix,label_function))
    #result=pool.map(f, range (0,affinity_matrix.shape[0]))
    # pool.close()  # 关闭进程池，不再接受新的任务
    # pool.join()  # 主进程阻塞等待子进程的退出
    #print()
    #return result
    #print(r.get())
    #print(r1.get()+r2.get())
    #return r1.get()+r2.get()
    #return r.get()
    #print(complete_result)
    return complete_result

    #return sub_results
   # t3 = time.time()

