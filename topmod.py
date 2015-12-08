import os
import numpy as np
import re
import lda
import pickle as pkl
import pandas as pd
from sklearn.linear_model import LogisticRegression


'''
import lda.datasets

X = lda.datasets.load_reuters()
vocab = lda.datasets.load_reuters_vocab()
titles = lda.datasets.load_reuters_titles()
X.shape
X.sum()

#print "vocab:\n", vocab
#print "titles:\n", titles
print len(vocab)
print len(titles)

model = lda.LDA(n_topics=20, n_iter=1500, random_state=1)
model.fit(X) 

topic_word = model.topic_word_ 
n_top_words = 8
for i, topic_dist in enumerate(topic_word):
	topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
	print('Topic {}: {}'.format(i, ' '.join(topic_words)))
'''

with open("user_document_dict2.pkl", "rb") as f:
	documents=pkl.load(f)

#print(type(documents))
#print([documents[key] for key in documents.keys()][:10])
#keys = np.array(list(documents.keys()))
#print(keys[:10])
#print(documents['9oYCjAP464zJ6HURxPQ8Uw'])

'''
CONVERTING THE DICTIONARY TO A SPARCE DATAFRAME I CAN CROSSTABULATE
'''

userID = []
tag = []
for i,j in documents.items():
	ctgs = []
	ctgs.extend(j.split(","))
	userID.extend([i]*len(ctgs))
	tag.extend(ctgs)
#print(userID)
#print(tag)
df = pd.DataFrame(userID,columns = ["userID"])
df["tag"] = tag
#print(df)
res = pd.crosstab(df.userID,df.tag)
#print(res)
X = np.array(res)
#print(X)

n_topic = 7

model = lda.LDA(n_topics=n_topic, n_iter=1000, random_state=1)
model.fit(X) 

topic_word = model.topic_word_ 
n_top_words = 15
for i, topic_dist in enumerate(topic_word):
	topic_words = np.unique(np.array(tag))[np.argsort(topic_dist)][:-(n_top_words+1):-1]
	print('Topic {}: {}'.format(i, ' '.join(topic_words)))



'''
Topic 0:  mexican1.0 sandwiches1.0 burgers1.0 hotdogs1.0 fastfood1.0 seafood1.0 delis1.0 mattresses2.0 furniture2.0 furniturestores2.0 homedecor2.0 chickenwings1.0 grocery1.0 pizza1.0
Topic 1:  breakfast2.0 breakfastbrunch2.0 brunch2.0 newamerican2.0 american(new)2.0 bars2.0 italian2.0 winebars2.0 sandwiches2.0 american(traditional)2.0 tradamerican2.0 cafes2.0 pizza2.0 french2.0
Topic 2:  divebars1.0 pubs1.0 grocery2.0 bars1.0 barbers2.0 barbers1.0 sportsbars1.0 spirits2.0 wine2.0 beerandwine2.0 beer2.0 drugstores2.0 pizza1.0 musicvenues1.0
Topic 3:  japanese2.0 sushi2.0 sushibars2.0 seafood2.0 asianfusion2.0 mexican2.0 chinese2.0 japanese3.0 bars2.0 lounges2.0 juicebars2.0 sushi3.0 sushibars3.0 thai2.0
Topic 4: coffee1.0  tea1.0 sandwiches1.0 cafes1.0 bakeries1.0 breakfast1.0 brunch1.0 breakfastbrunch1.0 juicebars1.0 bagels1.0 delis1.0 mexican1.0 pizza1.0 donuts1.0
Topic 5:  bars2.0 sportsbars2.0 burgers2.0 tradamerican2.0 american(traditional)2.0 mexican2.0 pubs2.0 seafood2.0 pizza2.0 american(new)2.0 newamerican2.0 mexican1.0 italian2.0 cocktailbars2.0
Topic 6:  pizza1.0 italian1.0 sandwiches1.0 seafood3.0 salad1.0 steak3.0 steakhouses3.0 delis1.0 tobaccoshops2.0 italian3.0 italian2.0 seafood2.0 vapeshops2.0 winebars3.0
Topic 7:  coffee2.0 icecream1.0 chinese1.0 bakeries2.0 vietnamese1.0 desserts2.0 korean2.0 tea2.0 frozenyogurt1.0 desserts1.0 japanese1.0 chinese2.0 asianfusion1.0 barbeque2.0
Topic 8:  italian2.0 vegetarian2.0 pizza2.0 vegetarian1.0 vegan2.0 vegan1.0 mexican1.0 salad2.0 thai1.0 mediterranean1.0 glutenfree2.0 gluten-free2.0 buffets2.0 juicebars1.0

'''


#Categorize the users

doc_topic = model.doc_topic_
for i in range(100):
	print("{} (top topic: {})".format(np.unique(np.array(userID))[i], doc_topic[i].argmax()))

user_belong = []
for i in range(len(np.unique(np.array(userID)))):
	user_belong.extend([doc_topic[i].argmax()])

user_cluster = pd.DataFrame(list(set(userID)),columns=["userID"])
user_cluster["group"] = user_belong

user_cluster.to_csv("user_cluster", index=False)

user_cluster = user_cluster.sort(["group","userID"])
user_counts_by_group = user_cluster.groupby("group").count()

'''
       userID
group        
0        3048
1        3375
2        1723
3        1983
4        2521
5        3153
6        1759
7        2163
8        1736

'''

#read and parse the data
xlsxfile = pd.ExcelFile("busi.xlsx")
busi = xlsxfile.parse('Sheet1')
busi = busi[["url","userID"]]

user_group_dict = user_cluster.set_index('userID')['group'].to_dict()
busi["group"] = busi["userID"].map(user_group_dict)


#xlsxfile2 = pd.ExcelFile("cmpltd.xlsx")   #rnn classification
xlsxfile2 = pd.ExcelFile("cmpltd_sanDiego_biz_gentrifiers.xlsx")    #svm classification


cmpltd = xlsxfile2.parse('cmpltd_sanDiego_biz_gentrifiers')  #tab name
#cmpltd = cmpltd[["url","rnn_gentrifier"]]   #rnn
cmpltd = cmpltd[["url","svm_pred"]]   #svm

#busi_group_dict = cmpltd.set_index('url')['rnn_gentrifier'].to_dict()  #rnn
busi_group_dict = cmpltd.set_index('url')['svm_pred'].to_dict()  #svm
busi["gentrifier"] = busi["url"].map(busi_group_dict)
busi = busi[np.isfinite(busi['gentrifier'])]  #remove the businesses I don't classify to 0/1

busi_group_count = busi.groupby(["url","group"]).count()

#busi_group_count.get_group([,1])
busi_group_count.reset_index(inplace=True) ##reformat the result table to normal
#busi_group_count.loc[busi_group_count.group == 1,:]
busi_group_count = busi_group_count.drop("gentrifier",1)

busi_group_attach = pd.DataFrame(columns=["url","group","userID"])

for i in list(set(busi_group_count["url"])):
	tmp = busi_group_count[busi_group_count["url"] == i]
	if tmp.shape[0] < n_topic:
		for j in range(n_topic):
			if j not in set(tmp["group"]):
				busi_group_attach.loc[busi_group_attach.shape[0]] = [i,j,0]

busi_group_count2 = pd.concat([busi_group_count,busi_group_attach])
busi_group_count2 = busi_group_count2.sort(["url","group"])

#Logistic Regression
y = pd.DataFrame(list(set(busi_group_count2["url"])),columns = ["url"])
y = y.sort("url")
y["gentrifier"] = y["url"].map(busi_group_dict)
y = list(y["gentrifier"])

X = pd.DataFrame()
X["X0"] = [1]*len(y)
varname = ["X1","X2","X3","X4","X5","X6","X7"]
for i,j in enumerate(varname):
	x = list(busi_group_count2.loc[busi_group_count2.group == i,"userID"])
	X[j] = x

# instantiate a logistic regression model, and fit with X and y
model = LogisticRegression()
model = model.fit(X, y)

# check the accuracy on the training set
model.score(X, y)

logit = pd.DataFrame(X.columns,columns=["var"])
logit["coef"] = model.coef_.tolist()[0]

print(logit)

'''
rnn logistic regression result:

  var      coef
0  X0 -2.188770
1  X1 -0.006921
2  X2 -0.157815
3  X3  0.390991
4  X4  0.213588
5  X5 -0.067527
6  X6  0.097989
7  X7 -0.063429
8  X8 -0.191999
9  X9 -0.307420



svm logistic regression result:

  var      coef
0  X0 -1.707240
1  X1  0.047583
2  X2 -0.042175
3  X3  0.022985
4  X4 -0.203906
5  X5  0.107342
6  X6  0.102000
7  X7 -0.125309
8  X8  0.058917
9  X9 -0.060631


'''

"""
7 groups:

Topic 0:  coffee2.0 bakeries2.0 desserts2.0 tea2.0 barbers2.0 mattresses2.0 grocery2.0 furniture2.0 furniturestores2.0 tobaccoshops2.0 homedecor2.0 drugstores2.0 hookahbars2.0 thriftstores1.0
Topic 1:  japanese2.0 sushi2.0 sushibars2.0 chinese1.0 vietnamese1.0 korean2.0 seafood2.0 chinese2.0 japanese1.0 asianfusion2.0 mexican1.0 asianfusion1.0 sushi1.0 sushibars1.0
Topic 2:  breakfastbrunch2.0 breakfast2.0 brunch2.0 bars2.0 newamerican2.0 american(new)2.0 seafood2.0 tradamerican2.0 american(traditional)2.0 mexican2.0 italian2.0 seafood3.0 sandwiches2.0 winebars2.0
Topic 3: coffee1.0  tea1.0 sandwiches1.0 icecream1.0 bakeries1.0 cafes1.0 juicebars1.0 breakfast1.0 brunch1.0 breakfastbrunch1.0 desserts1.0 bagels1.0 frozenyogurt1.0 mexican1.0
Topic 4:  italian2.0 pizza2.0 vegetarian2.0 vegetarian1.0 vegan2.0 vegan1.0 salad2.0 mexican1.0 gluten-free2.0 glutenfree2.0 winebars2.0 thai2.0 icecream1.0 chinese2.0
Topic 5:  mexican1.0 sandwiches1.0 pizza1.0 burgers1.0 delis1.0 hotdogs1.0 fastfood1.0 seafood1.0 italian1.0 salad1.0 grocery1.0 chickenwings1.0 barbers1.0 foodtrucks1.0
Topic 6:  sportsbars2.0 bars2.0 burgers2.0 mexican2.0 american(traditional)2.0 tradamerican2.0 pubs2.0 pizza2.0 seafood2.0 american(new)2.0 newamerican2.0 divebars1.0 cocktailbars2.0 mexican1.0


rnn:

  var      coef
0  X0 -2.184852
1  X1  0.047621
2  X2 -0.090805
3  X3  0.114205
4  X4 -0.166428
5  X5 -0.278772
6  X6  0.201090
7  X7 -0.06741


SVM:

  var      coef
0  X0 -1.688772
1  X1 -0.088777
2  X2  0.095261
3  X3 -0.027547
4  X4  0.000922
5  X5 -0.075818
6  X6  0.112027
7  X7 -0.043240

"""


