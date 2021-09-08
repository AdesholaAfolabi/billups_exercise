import re
import nltk
nltk.download('stopwords') # stop words removal
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english')) 
import csv
import multiprocessing

import warnings
warnings.filterwarnings('ignore')


PATH_TO_GLOVE_100D = '/Users/afolabiadeshola/Downloads/billups/data/glove.6B.100d.txt'
glove_embedding_dict_100d = {}

with open(PATH_TO_GLOVE_100D, encoding='utf-8') as f:
    reader = csv.reader(f, delimiter=' ',quoting=csv.QUOTE_NONE)       
    glove_embedding_dict_100d = {line[0]: np.array(list(map(float, line[1:]))) for line in reader}


def remove_stop_words(document):
    
    '''
    Remove uninformative words such as the, is, because, etc from the corpus
    
    Parameters:
    ------------------------
    document: The corpus to operate on
    
    Returns
    -------
        Corpus without stop words
    '''
  
    cleaned_document = " ".join([word.strip() for word in document.split() if word not in stop_words and len(word)>2])
    return cleaned_document

def preprocess_document_for_fine_tuning(document):
    
    '''
    Pre-process the corpus by converting each word to lower case, removing 
    special characters and apply the remove_stop_words method
    
    Parameters:
    ------------------------
    document: The corpus to operate on
    
    Returns
    -------
        Clean corpus
    '''
    document = document.lower()
    document = " ".join([re.sub('[^A-Za-z]+', ' ', word) for word in document.split()])
    document = remove_stop_words(document)
   
    return document


def get_embedding(word):
    
    '''
    Get word embeddings (using glove 100d in this case)
    
    Parameters:
    ------------------------
    word: The word to generate embeddings for
    
    Returns
    -------
        Embeddings of all words.
    '''
    glove_model = glove_embedding_dict_100d

    if word in glove_model:
        return glove_model[word]
    
    return glove_model['unk']

def get_document_mean_embedding(document):
    '''
    Aggregate all word embeddings by calculating the mean
    '''
    return np.mean([get_embedding(word) for word in document.split()],axis=0)

def flatten(list_of_lists):
    if len(list_of_lists) == 0:
        return list_of_lists
    if isinstance(list_of_lists[0], list):
        return flatten(list_of_lists[0]) + flatten(list_of_lists[1:])
    return list_of_lists[:1] + flatten(list_of_lists[1:])

def embedding_dataframe(data: pd.DataFrame) -> pd.DataFrame:
    
    '''
    Generate embeddings with n-dimension based on the trained word2vec or 
    glove100d model
    '''
    EMBEDDING_VECTOR_DIMENSION = 100
    emb_cols = [f'emb_{i}' for i in range(EMBEDDING_VECTOR_DIMENSION)]
    
    emb_list = []
    for index, row in data.iterrows():
        temp_list = []
        document = preprocess_document_for_fine_tuning(row['all_text'].strip())
        embeddings = get_document_mean_embedding(document).tolist()
        temp_list.append(embeddings) 
        emb_list.append(temp_list)
    
    emb_data = []
    for item in emb_list:
        emb_data.append(flatten(item))
        
    emb_frame = pd.DataFrame(emb_data, columns = emb_cols)
    return emb_frame

def merge_emb(data):
    '''
    Merge embeddings back to the original dataframe
    '''
    
    emb_frame = embedding_dataframe(data)
    emb_frame = emb_frame.fillna(0)
    mergedDf = data.merge(emb_frame, left_index=True, right_index=True)
    mergedDf.drop(['all_text'], axis=1, inplace=True)
    return mergedDf

if __name__ == '__main__':
    merge_emb(data)