#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import os 


# In[2]:


movies = pd.read_csv('tmdb_5000_movies.csv')


# In[3]:


credits = pd.read_csv('tmdb_5000_credits.csv')


# In[4]:


movies.head()


# In[5]:


credits.head()


# In[6]:


movies.shape


# In[7]:


credits.shape


# In[8]:


movies = movies.merge(credits,on = 'title')


# In[9]:


movies.head()


# In[10]:


movies.shape


# In[11]:


movies['original_language'].value_counts()


# In[12]:


movies.columns


# In[13]:


movies = movies[['movie_id','title','overview','genres','keywords','crew','cast']]


# In[14]:


movies.head()


# In[15]:


movies.isnull().sum()


# In[16]:


movies.dropna(inplace=True)


# In[17]:


movies.duplicated().sum()


# In[18]:


movies.isnull().sum()


# In[19]:


movies.iloc[0]['genres']


# In[20]:


import ast
def convert_genres(obj):
    lst =[]
    for i in ast.literal_eval(obj):
        lst.append(i['name'])
    return lst


# In[21]:


movies['genres']= movies['genres'].apply(convert_genres)


# In[22]:


movies.head(3)


# In[23]:


movies.iloc[0]['keywords']


# In[24]:


movies['keywords']=movies['keywords'].apply(convert_genres)


# In[25]:


movies.head(3)


# In[26]:


movies.iloc[0]['keywords']


# In[27]:


movies.iloc[0]['cast']


# In[28]:


def convert_cast(obj):
    lst =[]
    counter =0
    for i in ast.literal_eval(obj):
        if counter <3:
            lst.append(i['name'])
        counter +=1
    return lst


# In[29]:


movies['cast']=movies['cast'].apply(convert_cast)


# In[30]:


movies.head(3)


# In[31]:


movies.iloc[0]['crew']


# In[32]:


def convert_crew(obj):
    lst =[]
    for i in ast.literal_eval(obj):
        if i['job']=='Director':
            lst.append(i['name'])
            break
    return lst


# In[33]:


movies['crew']=movies['crew'].apply(convert_crew)


# In[34]:


movies.head(3)


# In[35]:


movies.iloc[0]['overview']


# In[36]:


movies['overview']=movies['overview'].apply(lambda x:x.split())
movies.head()


# In[37]:


movies.iloc[0]['overview']


# In[38]:


movies.head()


# In[39]:


def remove_space(word):
    lst =[]
    for i in word:
        lst.append(i.replace(" ",""))
    return lst


# In[40]:


movies['cast']=movies['cast'].apply(remove_space) 
movies['crew']=movies['crew'].apply(remove_space) 
movies['keywords']=movies['keywords'].apply(remove_space) 
movies['genres']=movies['genres'].apply(remove_space) 


# In[41]:


movies.head(3)


# In[42]:


movies['tags'] = movies['overview']+movies['genres']+movies['keywords']+movies['crew']+movies['cast']


# In[43]:


type(movies.iloc[0]['overview'])


# In[44]:


movies.head()


# In[45]:


new_df = movies[['movie_id','title','tags']]


# In[46]:


new_df.head()


# In[47]:


new_df.iloc[0]['tags']


# In[48]:


new_df['tags'] = new_df['tags'].apply(lambda x :" ".join(x))


# In[49]:


new_df.head()


# In[58]:


new_df.iloc[0]['tags']


# In[59]:


new_df['tags'] = new_df['tags'].apply(lambda x:x.lower())


# In[60]:


new_df.head()


# In[61]:


new_df.iloc[0]['tags']


# In[62]:


new_df.shape


# In[63]:


import nltk
from nltk.stem import PorterStemmer


# In[64]:


ps = PorterStemmer()


# In[65]:


def stems(text):
    lst =[]
    for i in text.split():
        lst.append(ps.stem(i))
    return " ".join(lst)


# In[66]:


new_df['tags'] = new_df['tags'].apply(stems)


# In[67]:


new_df.iloc[0]['tags']


# In[68]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 5000,stop_words = 'english')


# In[69]:


vector = cv.fit_transform(new_df['tags']).toarray()


# In[70]:


vector


# In[71]:


vector.shape


# In[72]:


from sklearn.metrics.pairwise import cosine_similarity


# In[73]:


similary = cosine_similarity(vector)


# In[74]:


similary


# In[75]:


similary.shape


# In[76]:


spiderman_indices = new_df[new_df['title'] == 'Spider_Man'].index


# In[77]:


def recommend(movie):
    index = new_df[new_df['title'] == movie].index[0]
    distances = sorted(list(enumerate(similary[index])),reverse = True ,key = lambda x:x[1])
    for i in distances[1:6]:
        print(new_df.iloc[i[0]].title)


# In[78]:


recommend('Spider-Man')


# In[79]:


recommend('Avatar')


# In[80]:


new_df.head(10)


# In[81]:


recommend('Avengers: Age of Ultron')


# In[85]:


import os
import pickle 

directory = 'artifacts'
if not os.path.exists(directory):
    os.makedirs(directory)

pickle.dump(new_df, open('artifacts/movie_list.pk1', 'wb'))

pickle.dump(similary, open('artifacts/similary.pk1', 'wb'))


# In[83]:


recommend('Batman Begins')


# In[87]:


import os 
import pickle


# In[89]:


pickle.dump(new_df,open('movies.pkl','wb'))


# In[90]:


new_df.to_dict()


# In[91]:


pickle.dump(new_df.to_dict(),open('movies_dict.pkl','wb'))


# In[93]:


pickle.dump(similary,open('similary.pkl','wb'))


# In[ ]:




