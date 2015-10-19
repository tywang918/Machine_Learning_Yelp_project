import os
import pandas as pd
import numpy as np
import re
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn import svm

work_path="C:/Users/SpiffyApple/Documents/Dropbox/Gene/Yelp_scrapping/OldData"
os.chdir(work_path)

"""
##################################################################################################################################################
    INPUT: 
        1. cmpltd_sanDiego_biz.csv
        2. restaurants_SanDiego_biz_giz.csv
    OUTPUT:
        1. --nearest neighbor classification results
        2. --SVM classification results
        3. --gentrifying business indicator
##################################################################################################################################################
DESCRIPTION:
    Using the complete or subsetted business dataset, find classify businesses as either gentrifying or not. We do this the following way:
        1. Find businesses in the $ price range and >$$ price range. 
        2. Categorize clusters of businesses as either expensive business cluster or cheap business cluster.
        3. If the cluster is categorized as cheap but there are a few >$$ businesses within that clustered, they will be mislabeled. These
            mislabeled business we classify as gentrifiers. 
##################################################################################################################################################
"""

##################################################################################################################################################
################################################################### FUNCTIONS ####################################################################


##################################################################################################################################################
################################################################### Sandbox ######################################################################
biz=pd.read_csv("cmpltd_sanDiego_biz.csv", encoding='latin-1')



