
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dtime
import math
import seaborn as sns
from math import radians
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from scipy.cluster.vq import kmeans2, whiten
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import roc_auc_score
from sklearn.grid_search import GridSearchCV
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import nltk.data
import nltk
from sklearn import metrics
from sklearn import cross_validation
from sklearn.model_selection import cross_validate
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, roc_auc_score, precision_score
from sklearn.model_selection import cross_val_predict


# In[2]:


# Converting txt data into Dataframe


# In[3]:


text_file = open("train.txt", "r",encoding="utf8", errors='ignore')
lines = text_file.readlines()
text_file.close()


# In[4]:


type(lines)


# In[5]:


import pandas as pd
import numpy as np
print(len(lines))


# In[6]:


text_file = open("label.txt", "r",encoding="utf8", errors='ignore')
label = text_file.readlines()
text_file.close()
# print(len(label))
# print(lines[0])


# In[7]:


df_train = pd.DataFrame(lines)
df_labels = pd.DataFrame(label)


# In[8]:


print(df_train.shape)
print(df_labels.shape)


# In[9]:


print(df_train[0:2])


# In[10]:


#split label and text into different columns
df_train['ID'] = df_train.iloc[:,0].str[:11]


# In[11]:


df_train['text'] = df_train.iloc[:,0].str[12:]


# In[12]:


print(df_train.shape)
print(df_train[0:2])


# In[13]:


#drop raw data column
df_train = df_train.drop(df_train.columns[0], axis=1)


# In[14]:


print(df_train.shape)
print(df_train[0:2])


# In[15]:


#trim leading and trailing whitespaces
df_train['text'] = df_train['text'].str.strip()


# In[16]:


print(df_train.loc[0:2])


# In[17]:


print(df_labels.loc[0:2])


# In[18]:


#assiging column name
df_labels.columns = ['raw']


# In[19]:


#extracting labels from raw data and creating a new column
df_labels['Label'] = df_labels.iloc[:,0].str[12:]
print(df_labels.loc[0:2])


# In[20]:


#extracting id from the raw data column and removing redundant characters from label column
df_labels['ID'] = df_labels['raw'].astype(str).str[:-2].astype(np.int64)
df_labels['Label'] = df_labels['Label'].astype(str).str[:-1].astype(np.int64)
print(df_labels.loc[0:2])


# In[21]:


#drop raw data column
df_labels = df_labels.drop(df_labels.columns[0], axis=1)
print(df_labels.loc[0:2])


# In[22]:


print(type(df_train.ix[0,"ID"]))
print(type(df_labels.ix[0,"ID"]))


# In[23]:


#Ensure both the ID columns have the same data type to avoid any error while merging the data frames
df_labels["ID"] = df_labels["ID"].astype(str)


# In[24]:


#merging two data frames
df = pd.merge(df_train,
                 df_labels,
                 on='ID')

print(df.loc[0:2])


# In[ ]:


#Save the dataframe as csv file
df.to_csv('data.csv')


# # Data Processing

# In[25]:


#Lower Case
df['text'] = df['text'].apply(lambda x: " ".join(x.lower() for x in x.split()))
df['text'].head()


# In[26]:


#Remove Punctuation
df['text'] = df['text'].str.replace('[^\w\s]','')
df['text'].head()


# In[27]:


#Remove Stopwords
#import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
stop = stopwords.words('english')
df['text'] = df['text'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
df['text'].head()


# In[28]:


#Identify top 50 common words
freq = pd.Series(' '.join(df['text']).split()).value_counts()[:50]
freq


# In[29]:


#top 50 common word removal
freq = list(freq.index)
df['text'] = df['text'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))
df['text'].head()


# In[30]:


#Tokenization
from textblob import TextBlob
TextBlob(df['text'][1]).words


# In[31]:


#Stemming
from nltk.stem import PorterStemmer
st = PorterStemmer()
df['train'] = df['text'].apply(lambda x: " ".join([st.stem(word) for word in x.split()]))


# In[32]:


#Term Frequency
tf1 = (df['text'][1:2]).apply(lambda x: pd.value_counts(x.split(" "))).sum(axis = 0).reset_index()
tf1.columns = ['words','tf']
tf1


# In[33]:


#Inverse document frequency
for i,word in enumerate(tf1['words']):
    tf1.loc[i, 'idf'] = np.log(df.shape[0]/(len(df[df['text'].str.contains(word)])))
tf1


# In[34]:


#Tf-idf method
tf1['tfidf'] = tf1['tf'] * tf1['idf']
tf1.shape


# In[35]:


#Tokenization
df['train_text'] = df['train'].apply(nltk.word_tokenize)


# In[36]:


df.head()


# In[37]:


#counting different labels
df.groupby('Label').count()


# In[38]:


#setting X & y for spliting the dataset
X = df['train']
y = df['Label']


# In[39]:


#Spliting data into train & test set, P.S: we dont really use it
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, shuffle = False)


# In[40]:


#Computing dimension of the document term matrix, training set
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_features=1000, lowercase=True, analyzer='word',
stop_words= 'english',ngram_range=(1,1))
train_vect = tfidf.fit_transform(X_train)
train_vect


# In[41]:


#Computing dimension of the document term matrix, testing set
tfidf = TfidfVectorizer(max_features=1000, lowercase=True, analyzer='word',
stop_words= 'english',ngram_range=(1,1))
test_vect = tfidf.fit_transform(X_test)
test_vect


# In[42]:


#Computing dimension of the document term matrix, whole dataset
tfidf = TfidfVectorizer(max_features=1000, lowercase=True, analyzer='word',
stop_words= 'english',ngram_range=(1,1))
all_vect = tfidf.fit_transform(X)
all_vect


# # Random Forest

# In[76]:


RF = RandomForestClassifier(n_estimators=500)


# In[77]:


RF.fit(all_vect, y)


# In[105]:


# print ('Accuracy Score: ', accuracy_score(y, rf_pred))


# In[87]:


scoring = {'acc': 'accuracy',
           'precision': 'precision_macro',
           'recall': 'recall_macro',
           'f1_score' : 'f1_macro'}
rf_scores = cross_validate(RF, all_vect, y, scoring=scoring,
                         cv=10, return_train_score=True)


# In[89]:


print(rf_scores.keys())
print(rf_scores['test_acc'].mean())
print(rf_scores['test_precision'].mean())
print(rf_scores['test_recall'].mean())
print(rf_scores['test_f1_score'].mean())


# In[104]:


#Use cross validation predict to make prediction on the entire dataset
rf_pred = cross_val_predict(RF, all_vect, y, cv=10)

#Use classification report to get the precision & recall table, compare with prediction and actual labels
print(classification_report(y,rf_pred))


# In[122]:


#Assign the prediction back to dataframe
df['rf_prediction'] = rf_pred


# In[123]:


#Count the prediction labels
df.groupby('rf_prediction').count()


# In[124]:


#Count the actual labels
df.groupby('Label').count()


# # Gradient Boost

# In[106]:


GB = GradientBoostingClassifier(learning_rate = 0.25)


# In[107]:


GB.fit(all_vect, y)


# In[57]:


scoring = {'acc': 'accuracy',
           'precision': 'precision_macro',
           'recall': 'recall_macro',
           'f1_score' : 'f1_macro'}
gb_scores = cross_validate(GB, all_vect, y, scoring=scoring,
                         cv=10, return_train_score=True)


# In[58]:


print(gb_scores.keys())
print(gb_scores['test_acc'].mean())
print(gb_scores['test_precision'].mean())
print(gb_scores['test_recall'].mean())
print(gb_scores['test_f1_score'].mean())


# In[108]:


#Use cross validation predict to make prediction on the entire dataset
gb_pred = cross_val_predict(GB, all_vect, y, cv=10)

#Use classification report to get the precision & recall table, compare with prediction and actual labels
print(classification_report(y,gb_pred))


# # Naive Bayes

# In[109]:


clf = MultinomialNB().fit(all_vect, y)


# In[60]:


scoring = {'acc': 'accuracy',
           'precision': 'precision_macro',
           'recall': 'recall_macro',
           'f1_score' : 'f1_macro'}
nb_scores = cross_validate(clf, all_vect, y, scoring=scoring,
                         cv=10, return_train_score=True)


# In[61]:


print(nb_scores.keys())
print(nb_scores['test_acc'].mean())
print(nb_scores['test_precision'].mean())
print(nb_scores['test_recall'].mean())
print(nb_scores['test_f1_score'].mean())


# In[110]:


#Use cross validation predict to make prediction on the entire dataset
nb_pred = cross_val_predict(clf, all_vect, y, cv=10)

#Use classification report to get the precision & recall table, compare with prediction and actual labels
print(classification_report(y,nb_pred))


# # KNN

# In[111]:


modelknn = KNeighborsClassifier(n_neighbors=5)
modelknn.fit(all_vect,y)


# In[67]:


scoring = {'acc': 'accuracy',
           'precision': 'precision_macro',
           'recall': 'recall_macro',
           'f1_score' : 'f1_macro'}
knn_scores = cross_validate(modelknn, all_vect, y, scoring=scoring,
                         cv=10, return_train_score=True)


# In[68]:


print(knn_scores.keys())
print(knn_scores['test_acc'].mean())
print(knn_scores['test_precision'].mean())
print(knn_scores['test_recall'].mean())
print(knn_scores['test_f1_score'].mean())


# In[112]:


#Use cross validation predict to make prediction on the entire dataset
knn_pred = cross_val_predict(modelknn, all_vect, y, cv=10)

#Use classification report to get the precision & recall table, compare with prediction and actual labels
print(classification_report(y,knn_pred))


# # SVM Linear

# In[113]:


svm_clf=SVC(kernel="linear").fit(all_vect,y)


# In[71]:


scoring = {'acc': 'accuracy',
           'precision': 'precision_macro',
           'recall': 'recall_macro',
           'f1_score' : 'f1_macro'}
svm_scores = cross_validate(svm_clf, all_vect, y, scoring=scoring,
                         cv=10, return_train_score=True)


# In[72]:


print(svm_scores.keys())
print(svm_scores['test_acc'].mean())
print(svm_scores['test_precision'].mean())
print(svm_scores['test_recall'].mean())
print(svm_scores['test_f1_score'].mean())


# In[114]:


#Use cross validation predict to make prediction on the entire dataset
svm_pred = cross_val_predict(svm_clf, all_vect, y, cv=10)

#Use classification report to get the precision & recall table, compare with prediction and actual labels
print(classification_report(y,svm_pred))


# # SVM Non-Linear

# In[115]:


svm_non=SVC(kernel="rbf",cache_size=1,class_weight={1:2}).fit(all_vect,y)


# In[74]:


scoring = {'acc': 'accuracy',
           'precision': 'precision_macro',
           'recall': 'recall_macro',
           'f1_score' : 'f1_macro'}
non_scores = cross_validate(svm_non, all_vect, y, scoring=scoring,
                         cv=10, return_train_score=True)


# In[75]:


print(non_scores.keys())
print(non_scores['test_acc'].mean())
print(non_scores['test_precision'].mean())
print(non_scores['test_recall'].mean())
print(non_scores['test_f1_score'].mean())


# In[116]:


#Use cross validation predict to make prediction on the entire dataset
non_pred = cross_val_predict(svm_non, all_vect, y, cv=10)

#Use classification report to get the precision & recall table, compare with prediction and actual labels
print(classification_report(y,non_pred))


# # Manipulating Labels

# In[90]:


#counting labels from original dataframe
df.groupby('Label').count()


# In[91]:


#Create new dataframe that only contains label 0 & 8, both contains 74 labels each
d1 = df[df['Label']==0].sample(74)
d2 = df[df['Label']==8].sample(74)
#Append them into a new dataframe
d3 = d1.append(d2, ignore_index=True)


# In[92]:


#Create new dataframe that only contains label 1 to 7
d4 = df[(df['Label'] != 0 ) & (df['Label'] != 8)]
#Append them together
ds = d4.append(d3, ignore_index=True)


# In[93]:


#Counting labels from the new dataframe
ds.groupby('Label').count()


# In[94]:


#Randomize the whole dataframe
ds = ds.sample(frac=1)


# In[95]:


#Set the target feature column as X1
X1 = ds['train']
y1 = ds['Label']


# In[96]:


##Computing new dimension of the document term matrix, whole dataset
tfidf = TfidfVectorizer(max_features=1000, lowercase=True, analyzer='word',
stop_words= 'english',ngram_range=(1,1))
new_vect = tfidf.fit_transform(X1)
new_vect


# # Random Foest (New Data)

# In[97]:


RF = RF.fit(new_vect, ds['Label'])


# In[83]:


scoring = {'acc': 'accuracy',
           'precision': 'precision_weighted',
           'recall': 'recall_macro',
           'f1_score' : 'f1_macro'}
RF_scores = cross_validate(RF, new_vect, ds['Label'], scoring=scoring,
                         cv=10, return_train_score=True)


# In[84]:


print(RF_scores.keys())
print(RF_scores['test_acc'].mean())
print(RF_scores['test_precision'].mean())
print(RF_scores['test_recall'].mean())
print(RF_scores['test_f1_score'].mean())


# In[99]:


#Use cross validation predict to make prediction on the entire dataset
RF_pred = cross_val_predict(RF, new_vect, y1, cv=10)

#Use classification report to get the precision & recall table, compare with prediction and actual labels
print(classification_report(y1,RF_pred))


# In[125]:


#Assign the prediction back to dataframe
ds['RF_prediction'] = RF_pred


# In[128]:


#Count the prediction labels
ds.groupby('RF_prediction').count()


# In[127]:


#Count the actual labels
ds.groupby('Label').count()


# # Gradient Boost (New Data)

# In[85]:


GB.fit(new_vect, ds['Label'])


# In[86]:


scoring = {'acc': 'accuracy',
           'precision': 'precision_weighted',
           'recall': 'recall_macro',
           'f1_score' : 'f1_macro'}
GB_scores = cross_validate(GB, new_vect, ds['Label'], scoring=scoring,
                         cv=10, return_train_score=True)


# In[87]:


print(GB_scores.keys())
print(GB_scores['test_acc'].mean())
print(GB_scores['test_precision'].mean())
print(GB_scores['test_recall'].mean())
print(GB_scores['test_f1_score'].mean())


# In[117]:


#Use cross validation predict to make prediction on the entire dataset
GB_pred = cross_val_predict(GB, new_vect, y1, cv=10)

#Use classification report to get the precision & recall table, compare with prediction and actual labels
print(classification_report(y1,GB_pred))


# # Naive Bayes (New Data)

# In[88]:


clf = MultinomialNB().fit(new_vect, ds['Label'])


# In[89]:


scoring = {'acc': 'accuracy',
           'precision': 'precision_weighted',
           'recall': 'recall_macro',
           'f1_score' : 'f1_macro'}
NB_scores = cross_validate(clf, new_vect, ds['Label'], scoring=scoring,
                         cv=10, return_train_score=True)


# In[90]:


print(NB_scores.keys())
print(NB_scores['test_acc'].mean())
print(NB_scores['test_precision'].mean())
print(NB_scores['test_recall'].mean())
print(NB_scores['test_f1_score'].mean())


# In[118]:


#Use cross validation predict to make prediction on the entire dataset
NB_pred = cross_val_predict(clf, new_vect, y1, cv=10)

#Use classification report to get the precision & recall table, compare with prediction and actual labels
print(classification_report(y1,NB_pred))


# # KNN (New Data)

# In[91]:


modelknn = KNeighborsClassifier(n_neighbors=5).fit(new_vect, ds['Label'])


# In[92]:


scoring = {'acc': 'accuracy',
           'precision': 'precision_weighted',
           'recall': 'recall_macro',
           'f1_score' : 'f1_macro'}
KNN_scores = cross_validate(modelknn, new_vect, ds['Label'], scoring=scoring,
                         cv=10, return_train_score=True)


# In[93]:


print(KNN_scores.keys())
print(KNN_scores['test_acc'].mean())
print(KNN_scores['test_precision'].mean())
print(KNN_scores['test_recall'].mean())
print(KNN_scores['test_f1_score'].mean())


# In[119]:


#Use cross validation predict to make prediction on the entire dataset
KNN_pred = cross_val_predict(modelknn, new_vect, y1, cv=10)

#Use classification report to get the precision & recall table, compare with prediction and actual labels
print(classification_report(y1,KNN_pred))


# # SVM Linear (New Data)

# In[94]:


svm_clf=SVC(kernel="linear").fit(new_vect,ds['Label'])


# In[95]:


scoring = {'acc': 'accuracy',
           'precision': 'precision_weighted',
           'recall': 'recall_macro',
           'f1_score' : 'f1_macro'}
SVM_scores = cross_validate(svm_clf, new_vect, ds['Label'], scoring=scoring,
                         cv=10, return_train_score=True)


# In[96]:


print(SVM_scores.keys())
print(SVM_scores['test_acc'].mean())
print(SVM_scores['test_precision'].mean())
print(SVM_scores['test_recall'].mean())
print(SVM_scores['test_f1_score'].mean())


# In[120]:


#Use cross validation predict to make prediction on the entire dataset
SVM_pred = cross_val_predict(svm_clf, new_vect, y1, cv=10)

#Use classification report to get the precision & recall table, compare with prediction and actual labels
print(classification_report(y1,SVM_pred))


# # SVM Non-Linear (New Data)

# In[97]:


svm_non=SVC(kernel="rbf",cache_size=1,class_weight={1:2}).fit(new_vect,ds['Label'])


# In[98]:


scoring = {'acc': 'accuracy',
           'precision': 'precision_weighted',
           'recall': 'recall_macro',
           'f1_score' : 'f1_macro'}
NON_scores = cross_validate(svm_non, new_vect, ds['Label'], scoring=scoring,
                         cv=10, return_train_score=True)


# In[99]:


print(NON_scores.keys())
print(NON_scores['test_acc'].mean())
print(NON_scores['test_precision'].mean())
print(NON_scores['test_recall'].mean())
print(NON_scores['test_f1_score'].mean())


# In[121]:


#Use cross validation predict to make prediction on the entire dataset
NON_pred = cross_val_predict(svm_non, new_vect, y1, cv=10)

#Use classification report to get the precision & recall table, compare with prediction and actual labels
print(classification_report(y1,NON_pred))

