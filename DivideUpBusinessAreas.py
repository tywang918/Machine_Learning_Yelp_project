import os, re
import pandas as pd
import numpy as np
from scipy.cluster.vq import kmeans2
from sklearn.cluster import AgglomerativeClustering as AC
from scipy.spatial.distance import pdist
from sklearn.cluster import DBSCAN
%pylab

dat_path="C:/Users/SpiffyApple/Documents/GitHub/Machine_Learning_Yelp_project" 
os.chdir(dat_path)          
"""
##########################################################################################################
    INPUT:
        1. cmpltd_sanDiego_biz.csv
        
    OUTPUT:
        1. divided city into k-areas
##########################################################################################################
DESCRIPTION:
    Using k-means, divide the city into k-smaller blocks and return the dividing boundaries as well as
    respective classes. 
##########################################################################################################
"""

##########################################################################################################
############################################ Data Prep ###################################################
city=pd.read_csv("cmpltd_sanDiego_biz.csv", encoding='latin-1')  

city.drop_duplicates(inplace=True)
city.dropna(subset=[['X', 'Y']], inplace=True)

##########################################################################################################
############################################ k-means #####################################################    
city['k_group']=kmeans2(np.array(city[['X', 'Y']]), k=40, iter=10000)[1] 
city.to_csv('cmpltd_sanDiego_biz_kgroups.csv', index=False)

##########################################################################################################
############################################ Match to Index ##############################################
city=pd.read_csv('cmpltd_sanDiego_biz_kgroups.csv', encoding='latin-1')
gIdx=pd.read_csv('index_results.csv')
city=city.merge(gIdx, right_on='group', left_on='k_group', how='left')

city.to_csv('cmpltd_sanDiego_biz_gindex.csv', index=False)
##########################################################################################################
############################################ Plot ########################################################
plt.scatter(x=city.X, y=city.Y, c=city.k_group, cmap=plt.cm.nipy_spectral, s=10)
plt.subplots_adjust(left=0.00, right=1.00, top=1.00, bottom=0.00)
plt.title="Subdivided City"
plt.legend(city.k_group.unique())

#########################################################################################################
############################################# DBSCAN ####################################################
#dist=pdist(city[['X', 'Y']], 'euclidean')
db=DBSCAN(eps=0.001, min_samples=15)
db.fit(city[['X', 'Y']])
db.labels_

city['db_group']=db.labels_
plt.scatter(x=city.X, y=city.Y, c=city.db_group, cmap=plt.cm.nipy_spectral, s=15)
plt.subplots_adjust(left=0.00, right=1.00, top=1.00, bottom=0.00)
plt.title="Subdivided City"
