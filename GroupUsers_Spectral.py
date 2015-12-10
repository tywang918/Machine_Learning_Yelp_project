import os, re
import pandas as pd
import numpy as np
from sklearn.cluster import spectral_clustering
import pickle as pkl

"""
##########################################################################################################
INPUT:
    1. UserLocations_cleaned.csv -- contains all users and the businesses they reviewed.
    
OUTPUT:
    1. UserLocations_grouped.csv
##########################################################################################################
DESCRIPTION:
    1. Classify users into n groups using an affinity matrix based on categories and price range.
    2. Plot groups
##########################################################################################################
"""
w_path="C:/Users/SpiffyApple/Documents/GitHub/Machine_Learning_Yelp_project"
os.chdir(w_path)
user_loc=pd.read_csv("UserLocations_cleaned.csv")

##########################################################################################################
############################################ Affinity Matrix #############################################
#---------------
#String Cleaning -- Removal Duplicates/plurals -- concat price range
user_loc.categories.dropna(axis=0, inplace=True)
user_loc.categories.replace("_", ' ', regex=True, inplace=True)
user_loc.categories.replace("&", ',', regex=True, inplace=True)
user_loc.categories.replace(' ', '', regex=True, inplace=True)

def remove_duplicates(row):
    cat_set=set([re.sub('s$', '', x) for x in row.split(",")])
    return(",".join(cat_set))
[j for j in enumerate(user_loc.columns.tolist())]    

def insert_price_to_cat(row):
    p=row[6]
    return(re.sub(',',str(p)+',',row[2]+","))
    
    
user_loc['new_cats']=user_loc.categories.apply(remove_duplicates)
#user_loc['new_cats']=user_loc.new_cats+","+user_loc.priceCat.astype(str)
user_loc['new_cats']=user_loc.apply(insert_price_to_cat, 1)
user_loc.to_csv('cmplt_SD_new_cats.csv', index=False)
#----------------
#Cross-tabulation 
cross_dict={}
cross_dict['userID']=[]
cross_dict['ctgs']=[]
for ID, data in user_loc.groupby('userID'):
    ctgs=[]
    for j in data['new_cats']:
        ctgs.extend(j.split(","))
    cross_dict['userID'].extend([ID]*len(ctgs))
    cross_dict['ctgs'].extend(ctgs)

temp_df=pd.DataFrame(cross_dict)  
cross_tab=pd.crosstab(temp_df.userID, temp_df.ctgs, colnames=['userID'], rownames=['ctgs']) 

##Spectral clustering
user_corr=np.corrcoef(cross_tab)
user_corr[np.where(user_corr<0)]=0
spctr= spectral_clustering(user_corr,n_clusters=5)
spctr_groups=spctr.tolist()
temp_dict={'userID':user_loc['userID'].drop_duplicates().tolist(), 'spctr_group':spctr_groups}
user_loc=user_loc.merge(pd.DataFrame(temp_dict), how='left')


user_loc.groupby('spctr_group').max()['priceCat']

%pylab
plt.scatter(x=user_loc.X, y=user_loc.Y,c=user_loc['spctr_group'], cmap=plt.cm.jet, s=13)
plt.subplots_adjust(left=0.00, right=1.00, top=1.00, bottom=0.00)
"""
dat_dict={}
for ID, data in user_loc.groupby('userID'):
    ctgs=[]
    for j in data['new_cats']:
        ctgs.extend(j.split(","))
    dat_dict[ID]=ctgs

document_dict={}
for key in dat_dict.keys():
    document_dict[key]=",".join(dat_dict[key])

with open("user_document_dict.pkl", 'wb') as f:
    pkl.dump(document_dict, f)

with open("user_document_dict.pkl", 'rb') as f:
    documents=pkl.load( f) 
"""   
##########################################################################################################
################################################ Sandbox  ################################################
##Make cat list
cat_list=list(set(user_loc.new_cats.tolist())) ##stupidly fast
uniq_cat=[]
for j in cat_list:
    uniq_cat.extend(j.split(","))
    
uniq_cat=list(set(uniq_cat)); len(uniq_cat)
