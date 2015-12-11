import os, re
import pandas as pd

"""
###################################################################################################
INPUT:
    1. SanDiego_biz_collection.csv -- the collection of all businesses collected
    2. 'cmpltd_sanDiego_biz.csv'
        ...through Yelp's API.
OUTPUT:
    2. [category]_SanDiego_biz.csv -- the collection of businesses that fit a category of interest. 
###################################################################################################
DESCRIPTION: Unfortunately, Yelp does not provide convenient 'restaurant' labels within 
its API, although it does in the API query but that would've been a more targeted look. Consequently,
we need to look at the categories of collected businesses and using key words tease out the 
category we desire. 

NOTE:
I have already looked through and collected the words for restaurants and shops.    
###################################################################################################
"""
############################# 0. Setup #############################################

data_path="C:/Users/SpiffyApple/Documents/GitHub/Machine_Learning_Yelp_project"
os.chdir(data_path)

biz=pd.read_csv('cmpltd_sanDiego_biz.csv', encoding='latin-1')

#############################  I. Assess what's available ###########################
biz.categories.dropna(axis=0, inplace=True)
biz.categories.replace("_", ' ', regex=True, inplace=True)
biz.categories.replace("&", ',', regex=True, inplace=True)
biz.categories.replace(' ', '', regex=True, inplace=True)

cats=biz.categories.str.split(',').to_dict()
unique_cats=[]
for key in cats:
    for x in cats[key]:
        if re.sub('s$', '',str.strip(x)) not in unique_cats:
            unique_cats.append(re.sub('s$', '',str.strip(x)))

all_cats=[]            
for key in cats:
    for x in cats[key]:
        all_cats.append(re.sub('s$', '',str.strip(x)))
        
temp_df=pd.DataFrame({'cat':all_cats})
X=temp_df.cat.value_counts()
X.sort(ascending=False)    
X[:100]       
############################# II. Get pertinent data ################################


food_list=['restaurant','grill','bar','cafe','bakery','pub','bbq','sushi','coffee',
'food','eat','food truck','juice','drinks', 'tea', 'deli', 'seafood', 'cajun', 'icecream',
'brunch', 'lunch', 'dinner', 'breakfast', 'bars', 'sandwiches', 'fast food', 'pizza', 'sandwiches',
'buffet', 'mediterrenean', 'bakeries', 'steak', 'divebar', 'mexican','italian', 'hawaiian', 'french', 
'dessert', 'yogurt', 'korean', 'bagels','asian fusion', 'chinese', 'soup', 'vegan', 'vegetarian', 'burgers']
food_str="|".join(food_list)

shop_list=['shop', 'shopping', 'candy store', 'candy', 'store','beer, wine & spirits', 'wine' ]
shop_str="|".join(shop_list)

food_data=biz[biz.categories.str.contains(food_str)]
shopping_data=biz[biz.categories.str.contains(shop_str)]

#food_data.to_csv("restaurants_SanDiego_biz.csv", index=False)
#shopping_data.to_csv('shopping_SanDiego_biz.csv', index=False) 
