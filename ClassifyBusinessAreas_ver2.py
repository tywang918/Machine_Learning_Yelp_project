import os
import pandas as pd
import numpy as np
import re
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn import svm
from scipy.spatial.distance import pdist
%pylab

work_path="C:/Users/SpiffyApple/Documents/GitHub/Machine_Learning_Yelp_project"
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
def assign_label(price):
    """Return 0 if cheap and 1 if expensive"""
    if price is not np.nan:
        if price<2:
            return(0)
        if price>2:
            return(1)

##################################################################################################################################################
################################################################### Setup ######################################################################
#biz=pd.read_csv("cmpltd_sanDiego_biz.csv", encoding='latin-1')
#biz.drop('cheap_expnsv',axis=1 ,inplace=True)
biz=pd.read_csv('cmpltd_sanDiego_biz_kgroups.csv', encoding='latin-1')

biz['expensive']=biz.priceCat.apply(lambda x: assign_label(x))

##Drop rows with unknown label:
biz.dropna(subset=['expensive', 'X', 'Y'],  inplace=True)
print("Prop of expensive businesses: [%.2f]" %(biz.expensive.sum()/len(biz.expensive)))

##Scale the data:
X=biz[['X','Y']]
Y=biz.expensive

##################################################################################################################################################
################################################################### Sandbox ######################################################################
gammas=biz.groupby('k_group')[['X', 'Y']].apply(pdist).apply(lambda x: (1/percentile(x,20))*50).tolist()

def svm_plot(X, C=200):
    gamma=X.apply(pdist).apply(lambda x: (1/percentile(x,20))*50).tolist()
    clf = svm.SVC(kernel='rbf',C=C,gamma=gamma)
    clf.fit(np.array(X), Y)  

    ##Plotting the result:
    xx, yy = np.meshgrid(np.linspace(X.iloc[:, 0].min(), X.iloc[:, 0].max(), 500),
                         np.linspace(X.iloc[:, 1].min(), X.iloc[:, 1].max(), 500))
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.imshow(Z, interpolation='nearest',
               extent=(xx.min(), xx.max(), yy.min(), yy.max()), aspect='auto',
               origin='lower', cmap=plt.cm.PuOr_r)
    contours = plt.contour(xx, yy, Z, levels=[0], linewidths=2,
                           linetypes='--')
    plt.scatter(X.iloc[:, 0], X.iloc[:, 1], s=30, c=Y, cmap=plt.cm.Paired)
    plt.subplots_adjust(left=0.00, right=1.00, top=.94, bottom=0.00)
    plt.title("C: %d, Gamma: %d" %(C,gamma))
    #plt.xticks(())
    #plt.yticks(())
    #plt.axis([-3, 3, -3, 3])
    plt.show()
    
biz.groupby('k_group')[['X', 'Y']].apply(lambda x: svm_plot(x))    
#####################################################################################################################################
################################################################ SVM ################################################################
##Fitting
biz.groupby('k_group')[['X', 'Y']].apply(pdist).apply(lambda x: percentile(x,25))
gamma=50000
C=200
clf = svm.SVC(kernel='rbf',C=C,gamma=gamma)
clf.fit(X, Y)  

##Plotting the result:
xx, yy = np.meshgrid(np.linspace(X.iloc[:, 0].min(), X.iloc[:, 0].max(), 500),
                     np.linspace(X.iloc[:, 1].min(), X.iloc[:, 1].max(), 500))
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.figure()
plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()), aspect='auto',
           origin='lower', cmap=plt.cm.PuOr_r)
contours = plt.contour(xx, yy, Z, levels=[0], linewidths=2,
                       linetypes='--')
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], s=30, c=Y, cmap=plt.cm.Paired)
plt.subplots_adjust(left=0.00, right=1.00, top=.94, bottom=0.00)
plt.title("C: %d, Gamma: %d" %(C,gamma))
#plt.xticks(())
#plt.yticks(())
#plt.axis([-3, 3, -3, 3])


biz['svm_pred']=(biz.expensive>clf.predict(X)).astype(int)
plt.scatter(x=biz[biz.svm_pred==1].X,y=biz[biz.svm_pred==1].Y, s=20, c='g')
print("Prop of expensive businesses seen as gentrifiers [%.2f]" %(biz['svm_pred'].sum()/biz.expensive.sum()))
print("Prop of expensive businesses seen as gentrifiers [%.2f]" %(biz['svm_pred'].sum()/len(biz.expensive)))
#biz['gentrifier']=(biz.expensive>biz.svm_pred).astype(int)
#####################################################################################################################################
################################################################ Nearest Neighbor ###################################################
r=.00025 #A block is .001 and two blocks are .003; therefore, .011 scans about 8 blocks in diameter.
neigh = RadiusNeighborsClassifier(radius=r) #from qGis nneighbor analysis
neigh.fit(X, Y)
predictions=neigh.predict(X)
plt.scatter(X.iloc[:,0], X.iloc[:,1], s=30, c=Y, cmap=plt.cm.Paired); plt.title('True labels')
plt.subplots_adjust(left=0, bottom=0, right=1, top=.95, wspace=0, hspace=0)


plt.figure(); 
plt.scatter(X.iloc[:,0], X.iloc[:,1], s=30, c=predictions, cmap=plt.cm.Paired); plt.title('Predicted labels, rad=%.3f' %r)
plt.subplots_adjust(left=0, bottom=0, right=1, top=.95, wspace=0, hspace=0)


biz['rnn_gentrifier']=(biz.expensive>predictions).astype(int)
plt.scatter(x=biz[biz.rnn_gentrifier==1].X,y=biz[biz.rnn_gentrifier==1].Y, s=20, c='g')
print("Prop of expensive businesses seen as gentrifiers [%.2f]" %((biz.rnn_gentrifier.sum()/biz.expensive.sum())))
print("Prop of expensive businesses seen as gentrifiers [%.2f]" %((biz.rnn_gentrifier.sum()/len(biz.expensive))))
#####################################################################################################################################
################################################################  SAVE RESULTS ######################################################
biz.to_csv("cmpltd_sanDiego_biz_gentrifiers.csv", index=False)
