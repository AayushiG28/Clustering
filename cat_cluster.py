#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import gower
from pyclustering.cluster.kmedoids import kmedoids
from pyclustering.cluster import cluster_visualizer
from pyclustering.utils import read_sample
from pyclustering.samples.definitions import FCPS_SAMPLES
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# In[3]:




def preprocess(path,filename):
    
    filepath=path
    fileName=filename
    train_df = pd.read_csv(fileName)
    #print(train_Df)
    #http://localhost:8888/notebooks/Untitled15.ipynb#
        
    train_df.drop('RESOURCE', axis=1, inplace=True)
    train_df.drop('Unnamed: 0', axis=1, inplace=True)
 

    col_names = list(train_df)

    for col in col_names:
        if(col!="RESOURCE"):
            train_df[col] = train_df[col].astype('category',copy=False)
    
    print(train_df.shape)
    return train_df
    


# In[4]:


def dismlrty_matrix(train_df):
    
    X = np.asarray(train_df)

    print("Getting the dissimilarity matrix from the gower distance") 
    matrix=gower.gower_matrix(X)
    print("Dissimilarity matrix Built 😊") 
    return matrix


# In[5]:


def cluster_medoids(matrix,initial_medoids):
    kmedoids_instance = kmedoids(matrix, initial_medoids, data_type='distance_matrix')
    kmedoids_instance.process()
    clusters = kmedoids_instance.get_clusters()
    medoids = kmedoids_instance.get_medoids()
    
    return clusters


# In[6]:


def cluster_label(cluster,num_clust,matrix):
    lst4=[]
    x=num_clust
    if x==3:
        x1=cluster[0]
        x2=cluster[1]
        x3=cluster[2]
    
        for i,x in enumerate(matrix):
            if i in x1:
                lst4.append(0)
            elif i in x2:
                lst4.append(1)
            else:
                lst4.append(2) 
        
    elif x==4:
        x1=cluster[0]
        x2=cluster[1]
        x3=cluster[2]
        x4=cluster[3]
    
        for i,x in enumerate(matrix):
            if i in x1:
                lst4.append(0)
            elif i in x2:
                lst4.append(1)
            elif i in x3:
                lst4.append(2)
            else:
                lst4.append(3) 
            
    elif x==5:
        x1=cluster[0]
        x2=cluster[1]
        x3=cluster[2]
        x4=cluster[3]
        x5=cluster[4]
    
        for i,x in enumerate(matrix):
            if i in x1:
                lst4.append(0)
            elif i in x2:
                lst4.append(1)
            elif i in x3:
                lst4.append(2)            
            elif i in x4:
                lst4.append(3)
            else:
                lst4.append(4)
    return lst4


# In[8]:


def best_cluster(train_df,matrix,num_clust):
    
    
    silhouet_new=[]
    # finding initial medioids frpm multiple iterations
    for iteration in range(5):
        silhouet=[]
        n_clusters=num_clust
        row1 = train_df.sample(n = num_clust) 
        initial_medoids=[]
        for i in row1.index:
            initial_medoids.append(i)
        
        print("no of iteration id",iteration)
        #print(initial_medoids)
        silhouet.append(initial_medoids)
        # calling the clustering function to perform clustering on the dissimilarity matrix
        
        clusters=cluster_medoids(matrix,initial_medoids)
        
        # labelling of clusters
        
        labels= cluster_label(clusters,num_clust,matrix)
        
        
        #calculating silhouette_score        
        silhouette_avg=silhouette_score(matrix,labels,metric='precomputed')
        
        
        silhouet.append(silhouette_avg)
        silhouet.append(labels)
        
        #print(silhouet)
        
        if len(silhouet_new)==0:
            silhouet_new.append(silhouet)
            print("true")
        else:
            if silhouet[1]>silhouet_new[0][1]:
                silhouet_new.clear()
                silhouet_new.append(silhouet)
                #print("false")
            
            else:
                print("True1")
                
        #print(silhouet_new)
        #silhouet.clear()  
        
        
    return silhouet_new
            
            
            
        
    
  

  
    
    


# In[9]:


def silhouette_graph(silh_score,matrix,train_Df,n_clusters):
    
    #print(silh_score)
    label=silh_score[0][2]
    sample_silhouette_values = silhouette_samples(matrix, label)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)
    ax1.set_xlim([-0.1, 1])
    
    ax1.set_ylim([0, len(train_Df) + (n_clusters + 1) * 10])
    
    cluster_new = np.asarray(label)
    
    silhouette_avg=silh_score[0][1]
    
    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        #print(i)
        ith_cluster_silhouette_values =             sample_silhouette_values[cluster_new == i]
        
        #print(ith_cluster_silhouette_values)
        
        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    
    

plt.show()
plt.savefig('sil_graph.png')
    
    
    


# In[10]:


def main():
    train=preprocess(r"C:\Users\1185833\Downloads","valid.csv")
    #print(train)
    
    matrix=dismlrty_matrix(train)
    
    print("..........")
    
    print("clustering the data......")
    silh_score=best_cluster(train,matrix,3)
    
    silhouette_graph(silh_score,matrix,train,3)
    
    return silh_score,matrix,train


# In[11]:


x=main()


# In[2]:


cd Downloads


# In[12]:


print(x)


# In[ ]:




