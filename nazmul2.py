#!/usr/bin/env python
# coding: utf-8

# In[33]:


import gensim
from gensim.models.word2vec import Word2Vec
import gensim.downloader as api
# Initialize WmdSimilarity.
from gensim.similarities import WmdSimilarity


# In[34]:


import json
# read file
with open('example.json', 'r') as myfile:
    data=myfile.read()


# In[35]:


data = json.loads(data)
#print(data)
#Data Link = https://sunnah.com/
#data =


# In[36]:


import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('punkt')# Download data for tokenizer.

nltk.download('wordnet')
stop_words = stopwords.words('english')


# In[37]:


def lematize_list(words):
    n = len(words)
    for i in range(n):
       w=words[i]
       words[i] = WordNetLemmatizer().lemmatize(w,'v')
       #print( words[i])
    
    return words


# In[38]:


def preprocess(doc):
    doc = doc.lower()  # Lower the text.
    doc = word_tokenize(doc)  # Split into words.
    doc = [w for w in doc if not w in stop_words]  # Remove stopwords.
    doc = [w for w in doc if w.isalpha()]  # Remove numbers and punctuation.
    return doc


# In[39]:


instance = WmdSimilarity.load("FinalProjectForWeb", mmap=None)


# In[40]:


sent = 'is Friday prayer mandatory for women and sick person ? and should everyone take a bath'

query = preprocess(sent)
query = lematize_list(query)
print (query)

sims = instance[query] 


# In[41]:


# Print the query and the retrieved documents, together with their similarities.
print()
print ('Query : ',sent)
print()
for i in range(3):
    print()
    print ( "Similarity = " , ( sims[i][1] ) )
    print (data[sims[i][0]])
    print()


# In[ ]:




