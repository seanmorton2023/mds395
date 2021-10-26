import numpy as np
import matplotlib.pyplot as plt
import math
import time

from sklearn.cluster import KMeans


def read_data():

    #open up our training data
    f = open('normalized_csv.csv', 'r')
    counter = -1

    #create two arrays: one for names, one for data for each metal
    name_array = []
    data_array = []

    for line in f.readlines():
    
        #skip the first line in the spreadsheet
        counter += 1
        if counter == 0:
            continue

        #split line by csv
        line = line.replace('\n','')
        line_list = line.split(',')
        #print(line_list)

        #once we get to an empty line, stop reading lines
        if line_list[0] == '':
            break

        #let each index of each list correspond to the data of one metal
        name = line_list[0:2]
        data = line_list[2:]
        data = [float(x) for x in data]
    
        name_array.append(name)
        data_array.append(data)

        #time.sleep(1)

    return name_array, data_array

def fsdam(array):
    '''calculates the sum of deviations from array mean

    Args:
     - array: an (m*n) array of m samples (rows), and n attributes
        (columns)

    Calculated within:
     - array_mean: a (1*n) array of n attributes, where each index
        in the array gives the mean of each attribute

    Returns:
     - value of sdam
    '''

    #value of deviations from array mean for one row of array
    #for normalized data, means are 0 for all data
    array_means = [0]*len(array[0])

    #this should give the number of rows
    num_samples = len(array)
    distances_array_means = [np.linalg.norm(array[i]-array_means) for i in range(num_samples)]
    sum_distances = sum(distances_array_means)

    #print(distances_array_means)
    return sum_distances

def fsdcm(array, class_means, class_ids):
    '''calculates the sum of deviations from class mean
        
    Args:
     - array: an (m*n) array of m samples (rows), and n attributes
        (columns)
     - class_means: a (k*n) array of k clusters (rows) and n attributes
        of the data (columns). each row corresponds to the sample mean
        of a different cluster
     - classifications: a (1*m) array that shows which  samples correspond
        to which labels

    Returns:
     - value of sdcm
     '''
    num_samples = len(array)
    dcms = []
   
    #iterate through all the samples; use indexing to find which class
    #they belong to and find deviation from class mean
    for i in range(num_samples):
        class_id = class_ids[i]
        sample_array = array[i]
        cluster_mean_array = class_means[class_id]

        #print('\nSample array and cluster mean array:')
        #print(sample_array)
        #print(cluster_mean_array)
        #print('class_id = ' + str(class_id))
        distance_cluster_mean = np.linalg.norm(sample_array - cluster_mean_array)
        dcms.append(distance_cluster_mean)

    #print(dcms)
    return sum(dcms)

def fgvf(array, class_means, class_ids):
    sdcm = fsdcm(array, class_means, class_ids)
    sdam = fsdam(array)
    gvf = (sdam - sdcm) / sdam
    return gvf

def find_k_vs_gvf(array, k_max = 10):
    #try out different values of k to determine what the optimal
    #value of k is ("the elbow of the GVF plot")
    #keep in mind we only have 34 data points. don't make too many clusters
    k_array = list(range(2,k_max))
    gvf_array = []

    for k in k_array:

        print('\nValue of k: ' + str(k))
        #use SKLearn KMeans class to analyze the data
        kmeans = KMeans(n_clusters = k, random_state = 0).fit(array)
        class_means = kmeans.cluster_centers_
        class_ids = kmeans.labels_
        #print('\nCluster centers:')
        #print(class_means)

        #print('\nGVF for k = ' + str(k) + ':')
        gvf = fgvf(array, class_means, class_ids)
        gvf_array.append(gvf)
        #print(gvf)
                                                       
    #plot the relation between k and GVf
    return k_array, gvf_array

def plot_k_vs_gvf(mech_array, therm_array, elec_array):
    #find optimal values of k and plot them
    print('\nMechanical properties array:')
    k_array_mech, gvf_array_mech = find_k_vs_gvf(mech_array, k_max = 25)
    print('\nThermal properties array:')
    k_array_therm, gvf_array_therm = find_k_vs_gvf(therm_array,  k_max = 12)
    k_array_elec, gvf_array_elec = find_k_vs_gvf(elec_array,k_max = 11)

    plt.figure(1)
    plt.plot(k_array_mech, gvf_array_mech, c='blue')
    plt.title('Mechanical properties: k vs. GVF')
    plt.xlabel('Value of k')
    plt.ylabel('Goodness of Fit (GVF)')

    plt.figure(2)
    plt.plot(k_array_therm,gvf_array_therm, c='red')
    plt.title('Thermal properties: k vs. GVF')
    plt.xlabel('Value of k')
    plt.ylabel('Goodness of Fit (GVF)')
    
    plt.figure(3)
    plt.plot(k_array_elec, gvf_array_elec, c='green')
    plt.title('Electrical properties: k vs. GVF')
    plt.xlabel('Value of k')
    plt.ylabel('Goodness of Fit (GVF)')
    plt.show()


if __name__ == '__main__':
    name_list, data_list = read_data()

    #extract mechanical, thermal, electrical data from array of lists
    mech_data = [inner_list[0:7] for inner_list in data_list]
    therm_data = [inner_list[7:10] for inner_list in data_list]
    elec_data = [inner_list[10:11] for inner_list in data_list]

    #convert to numpy format
    mech_array = np.asarray(mech_data)
    therm_array = np.asarray(therm_data)
    elec_array = np.asarray(elec_data)

    #make plots to determine optimal k for each property
    plot_k_vs_gvf(mech_array, therm_array, elec_array)