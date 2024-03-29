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

        #once we get to an empty line, stop reading lines
        if line_list[0] == '':
            continue

        #let each index of each list correspond to the data of one metal
        name = line_list[0:2]
        data = line_list[2:]
        data = [float(x) for x in data]
    
        name_array.append(name)
        data_array.append(data)

        #time.sleep(1)

    means_and_stdevs = data_array[-2:]
    data_points_array = data_array[:-2]

    return name_array, data_points_array, means_and_stdevs

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
        distance_cluster_mean = np.linalg.norm(sample_array - cluster_mean_array)
        dcms.append(distance_cluster_mean)

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
        gvf = fgvf(array, class_means, class_ids)
        gvf_array.append(gvf)
                                                       
    #plot the relation between k and GVf
    return k_array, gvf_array

def plot_k_vs_gvf(mech_array, therm_array, elec_array):
    #make plots to determine optimal k for each property
    k_array_mech, gvf_array_mech = find_k_vs_gvf(mech_array, k_max = 25)
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

def plot_elec_clusters(elec_array, elec_k):
    #graph electrical properties to test this out

    elec_kmeans = KMeans(n_clusters = elec_k, random_state = 0).fit(elec_array)
    elec_class_ids = elec_kmeans.labels_
    elec_class_means = elec_kmeans.cluster_centers_
    
    #separate the data points into different lists based on which
    #classID they are. iterate through each of the class IDs to figure out
    classid = 0
    lst_by_class = []

    for classid in range(elec_k):
        data = [elec_array[i][0] for i in range(len(elec_array)) if 
                elec_class_ids[i] == classid]
        lst_by_class.append(data)

    #iterate through and plot the data of different classes by color
    plt.figure(1)
    for i in range(elec_k):
        classx = lst_by_class[i]
        y_axis = [0]*len(classx)
        plt.scatter(classx, y_axis, s=32)

    plt.show()

def plot_therm_clusters(therm_array, k_therm, means_and_stdevs):
    #plot thermal data in 3D to visualize clusters
    kmeans_therm = KMeans(n_clusters = k_therm, random_state = 0).fit(therm_array)
    therm_class_ids = kmeans_therm.labels_
    therm_class_means = kmeans_therm.cluster_centers_

    #de-normalize data using d = d_normal * sigma + mu
    denorm_array = []
    for data in therm_array:
        denorm_array.append([data[i] * means_and_stdevs[1][i] 
                + means_and_stdevs[0][i] for i in range(len(data))] )

    therm_array = denorm_array[:]

    #separate the data points into different lists based on which
    #classID they are. iterate through each of the class IDs to figure out
    classid = 0
    lst_by_class = []

    for classid in range(k_therm):
        data = [therm_array[i] for i in range(len(therm_array)) if 
                therm_class_ids[i] == classid]
        lst_by_class.append(data)

    #iterate through and plot the data of different classes by color
    fig = plt.figure(1)
    ax = fig.add_subplot(projection = '3d')

    for i in range(k_therm):

        class_points = lst_by_class[i]
        class_x = [point[0] for point in class_points]
        class_y = [point[1] for point in class_points]
        class_z = [point[2] for point in class_points]
       
        ax.scatter(class_x, class_y, class_z)

    ax.set_xlabel('Thermal conductivity')
    ax.set_ylabel('Specific heat capacity')
    ax.set_zlabel('Coefficient of thermal expansion')
    ax.set_title('Thermal properties of steels with k=6 clusters')
    plt.show()

def classify_and_export(mech_array, therm_array, elec_array):
    #selected values of k
    k_mech = 11
    k_therm = 6
    k_elec = 5
    
    #find clusters of data and show them with the normalized data
    mech_kmeans = KMeans(n_clusters = k_mech, random_state = 0).fit(mech_array)
    therm_kmeans = KMeans(n_clusters = k_therm, random_state = 0).fit(therm_array)
    elec_kmeans = KMeans(n_clusters = k_elec, random_state = 0).fit(elec_array)

    mech_class_ids = mech_kmeans.labels_
    therm_class_ids = therm_kmeans.labels_
    elec_class_ids = elec_kmeans.labels_

    #group together the normalized data and their classIDs to export to csv
    mech_array_and_classid = [list(np.concatenate(( [mech_class_ids[i]], mech_array[i] )) )
                                              for i in range(len(mech_array)) ]
    mech_array_and_classid = np.array(mech_array_and_classid)


    therm_array_and_classid = [list(np.concatenate(( [therm_class_ids[i]], therm_array[i] )) )
                                              for i in range(len(therm_array)) ]
    therm_array_and_classid = np.array(therm_array_and_classid)


    elec_array_and_classid = [list(np.concatenate(( [elec_class_ids[i]], elec_array[i] )) )
                                              for i in range(len(elec_array)) ]
    elec_array_and_classid = np.array(elec_array_and_classid)

    #export all these class data arrays to CSV
    f = open('mech_array_labeled.csv', 'w')
    g = open('therm_array_labeled.csv', 'w')
    h = open('elec_array_labeled.csv', 'w')

    num_points = len(mech_array_and_classid)
    for i in range(num_points):

        str_mech = ','.join([str(x) for x in mech_array_and_classid[i]])
        str_therm = ','.join([str(x) for x in therm_array_and_classid[i]])
        str_elec = ','.join([str(x) for x in elec_array_and_classid[i]])

        f.write(str_mech + '\n')
        g.write(str_therm + '\n')
        h.write(str_elec + '\n')

    f.close()
    g.close()
    h.close()

if __name__ == '__main__':
    name_list, data_list, means_and_stdevs = read_data()

    #extract mechanical, thermal, electrical data from array of lists
    mech_data = [inner_list[0:7] for inner_list in data_list]
    therm_data = [inner_list[7:10] for inner_list in data_list]
    elec_data = [inner_list[10:11] for inner_list in data_list]

    #means and stdevs for data, for use in de-normalizing data
    mech_mus_sigmas = [inner_list[0:7] for inner_list in means_and_stdevs]
    therm_mus_sigmas = [inner_list[7:10] for inner_list in means_and_stdevs]
    elec_mus_sigmas = [inner_list[10:11] for inner_list in means_and_stdevs]

    #convert to numpy format
    mech_array = np.asarray(mech_data)
    therm_array = np.asarray(therm_data)
    elec_array = np.asarray(elec_data)

    #for finding ideal values of k for each set of properties
    #plot_k_vs_gvf(mech_array, therm_array, elec_array)

    #selected values of k
    k_mech = 11
    k_therm = 6
    k_elec = 5

    #plot_elec_clusters(elec_array, k_elec)
    plot_therm_clusters(therm_array, k_therm, therm_mus_sigmas)
    #classify_and_export(mech_array, therm_array, elec_array)
