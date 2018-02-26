###########################################################################################
########################## Data Processing ################################################
###########################################################################################
########### Prepare Data for training set #################################################

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
import string
from sklearn.datasets import load_files
from sklearn.naive_bayes import MultinomialNB
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
import collections
from itertools import dropwhile

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import csv
import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import cross_validation 
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn import grid_search



###########################################################################################
train_data = load_files("/home/srichaitanna/Downloads/training",
                        load_content=True, encoding = "utf-8")

test_data = load_files("/home/srichaitanna/Downloads/test",
                        load_content=True, encoding = "utf-8")

all_data=train_data.data + test_data.data
train_data.target
test_data.target
# The returned dataset is a scikit-learn a simple holder object with fields that 
##### can be both accessed as python dict keys or object attributes for 
#convenience, for instance the target_names holds the list of the requested category names:
    
print(train_data.target_names)
print(train_data.data[0])
print(train_data.target_names[train_data.target[0]])

# the last two lines of codes gives the first text and corresponding class names

#1. Remove The Stop Words From The Raw Text After Tokenization And Then Map Each Token To Its
#Stem Using A Stemming Algorithm.

for i in range (0,len(all_data)):
     all_data[i] = all_data[i].lower()
    
print(all_data)
 
for i in range (0,len(all_data)):
     all_data[i] =  re.sub(r'\d+', '', all_data[i])
 
rawdata_docs = all_data
print(rawdata_docs)

import nltk
nltk.download('punkt')
nltk.download('stopwords')

tokenizeddata_docs = [word_tokenize(doc) for doc in rawdata_docs]

regex = re.compile('[%s]' % re.escape(string.punctuation))

tokenizeddata_docs_no_punctuation = []

for review in tokenizeddata_docs:
     data_review = []
     for token in review:
         new_token = regex.sub(u'', token)
         if not new_token == u'':
                 data_review.append(new_token)
     tokenizeddata_docs_no_punctuation.append(data_review)
  

len(tokenizeddata_docs_no_punctuation)
print(tokenizeddata_docs_no_punctuation)

tokenizeddata_docs_no_stopwords = []
for doc in tokenizeddata_docs_no_punctuation:
        data_term_vector = []
        for word in doc:
            if not word in stopwords.words('english'):
                data_term_vector.append(word)
        tokenizeddata_docs_no_stopwords.append(data_term_vector)
 
len(tokenizeddata_docs_no_stopwords)
print(tokenizeddata_docs_no_stopwords)

porter = PorterStemmer()
preprocesseddata_docs = []
for doc in tokenizeddata_docs_no_stopwords:
     finaldata_doc = []
     for word in doc:
         finaldata_doc.append(porter.stem(word))
     preprocesseddata_docs.append(finaldata_doc)
     
len(preprocesseddata_docs)
print(preprocesseddata_docs)

 
######################################################################################
#######################################################################################
######## parsing #########################################################################

nstemmed_data = []
for i in range(len(preprocesseddata_docs)): ## put in a single list
    for w in preprocesseddata_docs[i]:
        nstemmed_data.append(w)
        
print(nstemmed_data)

counter = collections.Counter(nstemmed_data)
print(counter.most_common)

for key, count in dropwhile(lambda key_count: key_count[1] > 20, counter.most_common()):
    del(counter[key])
    
print(counter.keys())

newlist = list()
for i in counter.keys():
    newlist.append(i)

print(newlist)
len(newlist)

parseddata_docs = []
for i in range(len(preprocesseddata_docs)):
    nparseddata_docs = []
    for w in preprocesseddata_docs[i]:
        if w in newlist:
             nparseddata_docs.append(w)
    parseddata_docs.append(nparseddata_docs)

print(parseddata_docs)   



#############################################################################################
##########################################################################################
########################## joining #######################################################
stemmed_data = []
for i in range(0,len(parseddata_docs)):
      d = " ".join(parseddata_docs[i])
      stemmed_data.append(d)
 
len(stemmed_data)

print(stemmed_data)
################ Try with both the tf matrix an tf-idf matrix and try to compare #########
################ these results or justify them ###########################################
###Create The Term-Document Matrix By Using An Efficient Tf-Idf Weighting Scheme.#########
###### try using lda and not using lda on them and try to justify them ###################
 
count_vect = CountVectorizer()
stemmed_data_counts = count_vect.fit_transform(stemmed_data)
stemmed_data_counts.shape

from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
stemmed_data_tfidf = tfidf_transformer.fit_transform(stemmed_data_counts)
stemmed_data_tfidf.shape

#data_feature = stemmed_data_tfidf  ###for tf-idf format
data_feature = stemmed_data_counts ###for tf format


from scipy.sparse import csc_matrix

m=data_feature 
data=pd.SparseDataFrame([ pd.SparseSeries(m[i].toarray().ravel()) 
                              for i in np.arange(m.shape[0]) ])
    
data.var()
data.mean()
data_feature=data.loc[:,data.var()>0.1]
data_feature=data_feature.loc[:,data_feature.mean()>0.001]

data_feature.shape



train_feature=data_feature[0:499]
test_feature=data_feature[499:]
train_target=train_data.target
test_target=test_data.target


#data_target = data_data.target
#print(data_target)
#print(data_data.target_names)





######################### Modeling ########################################################
##### 1. Naive-base #######################################################################
############ try with both tf and tf-idf matrix and compare them ##########################

clf = MultinomialNB().fit(train_feature, train_target)
predicted = clf.predict(train_feature)

np.mean(predicted == train_target)

pr=precision_score(train_target, predicted, average='micro')
print('\n Train Set Precision:'+str(pr))

re=recall_score(train_target, predicted, average='micro')
print('\n Train Set Recall:'+str(re))

fm=f1_score(train_target, predicted, average='micro')
print('\n Train Set F-Measure:'+str(fm))


########################prediction on test data

predicted = clf.predict(test_feature)

np.mean(predicted == test_target)

pr=precision_score(test_target, predicted, average='micro')
print('\n Train Set Precision:'+str(pr))

re=recall_score(test_target, predicted, average='micro')
print('\n Train Set Recall:'+str(re))

fm=f1_score(test_target, predicted, average='micro')
print('\n Train Set F-Measure:'+str(fm))


############################################################################
######################Gaussian #####################################

from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf=clf.fit(np.array(train_feature),np.array(train_target))
predicted = clf.predict(np.array(train_feature))

np.mean(predicted == train_target)
pr=precision_score(train_target, predicted, average='micro')
print('\n Train Set Precision:'+str(pr))

re=recall_score(train_target,predicted,average='micro')
print('\n Train Set Recall:'+str(re))

fm=f1_score(train_target, predicted, average='micro')
print('\n Train Set F-Measure:'+str(fm))


########################prediction on test data

predicted = clf.predict(np.array(test_feature))

np.mean(predicted == test_target)

pr=precision_score(test_target, predicted, average='micro')
print('\n Train Set Precision:'+str(pr))

re=recall_score(test_target, predicted, average='micro')
print('\n Train Set Recall:'+str(re))

fm=f1_score(test_target, predicted, average='micro')
print('\n Train Set F-Measure:'+str(fm))












#######################################################################################################
#############svm###########################################
###############svm with linear kernel##################
from sklearn import svm
clf=svm.SVC(C=50.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='auto', kernel='linear',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False).fit(train_feature, train_target)




# SVM Classifier 
svr = svm.SVC(kernel='linear')

# Grid Search 
#C=[]
#for i in range(1,100):
#    C.append(int(i))
    
param_grid =[{'C':[0.0001,001,0.5,1,2,3]},] 
#param_grid =[{'kernel':['linear'],C}]  #,{'kernel':['poly','rbf'],'C':[1,10,100]},]  # Sets of parameters

grid = grid_search.GridSearchCV(svr,param_grid,cv=10)          
grid.fit(train_feature,train_target)    
clf= grid.best_estimator_                   # Best grid
print '\n The best grid is as follows: \n'
print grid.best_estimator_ 



predicted = clf.predict(train_feature)

np.mean(predicted == train_target)

pr=precision_score(train_target, predicted, average='micro')
print('\n Train Set Precision:'+str(pr))

re=recall_score(train_target, predicted, average='micro')
print('\n Train Set Recall:'+str(re))

fm=f1_score(train_target, predicted, average='micro')
print('\n Train Set F-Measure:'+str(fm))


########################prediction on test data

predicted = clf.predict(test_feature)

np.mean(predicted == test_target)

pr=precision_score(test_target, predicted, average='micro')
print('\n Train Set Precision:'+str(pr))

re=recall_score(test_target, predicted, average='micro')
print('\n Train Set Recall:'+str(re))

fm=f1_score(test_target, predicted, average='micro')
print('\n Train Set F-Measure:'+str(fm))








###############svm with rbf kernel##################
from sklearn import svm
clf=svm.SVC(C=50.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False).fit(train_feature, train_target)


predicted = clf.predict(train_feature)

np.mean(predicted == train_target)

pr=precision_score(train_target, predicted, average='micro')
print('\n Train Set Precision:'+str(pr))

re=recall_score(train_target, predicted, average='micro')
print('\n Train Set Recall:'+str(re))

fm=f1_score(train_target, predicted, average='micro')
print('\n Train Set F-Measure:'+str(fm))


########################prediction on test data

predicted = clf.predict(test_feature)

np.mean(predicted == test_target)

pr=precision_score(test_target, predicted, average='micro')
print('\n Train Set Precision:'+str(pr))

re=recall_score(test_target, predicted, average='micro')
print('\n Train Set Recall:'+str(re))

fm=f1_score(test_target, predicted, average='micro')
print('\n Train Set F-Measure:'+str(fm))



###############svm with sigmoid kernel##################
from sklearn import svm
clf=svm.SVC(C=50.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='auto', kernel='sigmoid',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False).fit(train_feature, train_target)


predicted = clf.predict(train_feature)

np.mean(predicted == train_target)

pr=precision_score(train_target, predicted, average='micro')
print('\n Train Set Precision:'+str(pr))

re=recall_score(train_target, predicted, average='micro')
print('\n Train Set Recall:'+str(re))

fm=f1_score(train_target, predicted, average='micro')
print('\n Train Set F-Measure:'+str(fm))


########################prediction on test data

predicted = clf.predict(test_feature)

np.mean(predicted == test_target)

pr=precision_score(test_target, predicted, average='micro')
print('\n Train Set Precision:'+str(pr))

re=recall_score(test_target, predicted, average='micro')
print('\n Train Set Recall:'+str(re))

fm=f1_score(test_target, predicted, average='micro')
print('\n Train Set F-Measure:'+str(fm))






###############svm with polynimial kernel##################
from sklearn import svm
clf=svm.SVC(C=50.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='auto', kernel='polynomial',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False).fit(train_feature, train_target)


predicted = clf.predict(train_feature)

np.mean(predicted == train_target)

pr=precision_score(train_target, predicted, average='micro')
print('\n Train Set Precision:'+str(pr))

re=recall_score(train_target, predicted, average='micro')
print('\n Train Set Recall:'+str(re))

fm=f1_score(train_target, predicted, average='micro')
print('\n Train Set F-Measure:'+str(fm))


########################prediction on test data

predicted = clf.predict(test_feature)

np.mean(predicted == test_target)

pr=precision_score(test_target, predicted, average='micro')
print('\n Train Set Precision:'+str(pr))

re=recall_score(test_target, predicted, average='micro')
print('\n Train Set Recall:'+str(re))

fm=f1_score(test_target, predicted, average='micro')
print('\n Train Set F-Measure:'+str(fm))





################################################################################################################
######################logistic rgeression classifier##################################################
clf = sklearn.linear_model.LogisticRegressionCV(class_weight='balanced',max_iter=1000) 

param_grid = {'Cs':[1,5,10,20] }
grid= grid_search.GridSearchCV(clf, param_grid,cv=10)
clf=grid.fit(train_feature, train_target)
#clf.get_params().keys()
clf= grid.best_estimator_                   # Best grid
print '\n The best grid is as follows: \n'
print grid.best_estimator_ 




predicted = clf.predict(train_feature)

np.mean(predicted == train_target)

pr=precision_score(train_target, predicted, average='micro')
print('\n Train Set Precision:'+str(pr))

re=recall_score(train_target, predicted, average='micro')
print('\n Train Set Recall:'+str(re))

fm=f1_score(train_target, predicted, average='micro')
print('\n Train Set F-Measure:'+str(fm))


########################prediction on test data

predicted = clf.predict(test_feature)

np.mean(predicted == test_target)

pr=precision_score(test_target, predicted, average='micro')
print('\n Train Set Precision:'+str(pr))

re=recall_score(test_target, predicted, average='micro')
print('\n Train Set Recall:'+str(re))

fm=f1_score(test_target, predicted, average='micro')
print('\n Train Set F-Measure:'+str(fm))





############################################################################################
######################multilayer perceptron ##################################

####################### multilayer perceptron with logistic activation function#############################
clf=MLPClassifier(hidden_layer_sizes=(100,100 ), activation='logistic', solver='adam', alpha=0.1, batch_size='auto', 
 learning_rate='constant', learning_rate_init=0.01,power_t=0.5, max_iter=200, shuffle=True,
 random_state=1, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, 
 early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

#gs = grid_search.GridSearchCV(clf, param_grid={
 #   'learning_rate_init': [0.5,0.1,0.05, 0.01,0.001],
#})
#grid=gs.fit(train_feature, train_target) 
#clf=grid.best_estimator_




clf=clf.fit(train_feature, train_target) 
predicted = clf.predict(train_feature)

np.mean(predicted == train_target)

pr=precision_score(train_target, predicted, average='micro')
print('\n Train Set Precision:'+str(pr))

re=recall_score(train_target, predicted, average='micro')
print('\n Train Set Recall:'+str(re))

fm=f1_score(train_target, predicted, average='micro')
print('\n Train Set F-Measure:'+str(fm))


########################prediction on test data

predicted = clf.predict(test_feature)

np.mean(predicted == test_target)

pr=precision_score(test_target, predicted, average='micro')
print('\n Train Set Precision:'+str(pr))

re=recall_score(test_target, predicted, average='micro')
print('\n Train Set Recall:'+str(re))

fm=f1_score(test_target, predicted, average='micro')
print('\n Train Set F-Measure:'+str(fm))




####################### multilayer perceptron with hyperbolic tan activation function#############################
clf=MLPClassifier(hidden_layer_sizes=(100,100 ), activation='tanh', solver='adam', alpha=0.5, batch_size='auto', 
 learning_rate='constant', learning_rate_init=0.01, power_t=0.5, max_iter=2000, shuffle=True,
 random_state=1, tol=0.0000001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, 
 early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-10)

clf=clf.fit(train_feature, train_target) 

predicted = clf.predict(train_feature)

np.mean(predicted == train_target)

pr=precision_score(train_target, predicted, average='micro')
print('\n Train Set Precision:'+str(pr))

re=recall_score(train_target, predicted, average='micro')
print('\n Train Set Recall:'+str(re))

fm=f1_score(train_target, predicted, average='micro')
print('\n Train Set F-Measure:'+str(fm))


########################prediction on test data

predicted = clf.predict(test_feature)

np.mean(predicted == test_target)

pr=precision_score(test_target, predicted, average='micro')
print('\n Train Set Precision:'+str(pr))

re=recall_score(test_target, predicted, average='micro')
print('\n Train Set Recall:'+str(re))

fm=f1_score(test_target, predicted, average='micro')
print('\n Train Set F-Measure:'+str(fm))



####################### multilayer perceptron with hyperbolic tan activation function#############################
clf=MLPClassifier(hidden_layer_sizes=(100,100 ), activation='relu', solver='adam', alpha=0.015, batch_size='auto', 
 learning_rate='constant', learning_rate_init=0.01, power_t=0.5, max_iter=200, shuffle=True,
 random_state=1, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, 
 early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

clf=clf.fit(train_feature, train_target) 

predicted = clf.predict(train_feature)

np.mean(predicted == train_target)

pr=precision_score(train_target, predicted, average='micro')
print('\n Train Set Precision:'+str(pr))

re=recall_score(train_target, predicted, average='micro')
print('\n Train Set Recall:'+str(re))

fm=f1_score(train_target, predicted, average='micro')
print('\n Train Set F-Measure:'+str(fm))


########################prediction on test data

predicted = clf.predict(test_feature)

np.mean(predicted == test_target)
pr=precision_score(test_target, predicted, average='micro')
print('\n Train Set Precision:'+str(pr))
re=recall_score(test_target, predicted, average='micro')
print('\n Train Set Recall:'+str(re))
fm=f1_score(test_target, predicted, average='micro')
print('\n Train Set F-Measure:'+str(fm))




####################### multilayer perceptron with hyperbolic identity activation function#############################
clf=MLPClassifier(hidden_layer_sizes=(100,100 ), activation='identity', solver='adam', alpha=1, batch_size='auto', 
 learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True,
 random_state=1, tol=0.005, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, 
 early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

clf=clf.fit(train_feature, train_target) 

predicted = clf.predict(train_feature)

np.mean(predicted == train_target)

pr=precision_score(train_target, predicted, average='micro')
print('\n Train Set Precision:'+str(pr))

re=recall_score(train_target, predicted, average='micro')
print('\n Train Set Recall:'+str(re))

fm=f1_score(train_target, predicted, average='micro')
print('\n Train Set F-Measure:'+str(fm))


########################prediction on test data

predicted = clf.predict(test_feature)

np.mean(predicted == test_target)
pr=precision_score(test_target, predicted, average='micro')
print('\n Train Set Precision:'+str(pr))
re=recall_score(test_target, predicted, average='micro')
print('\n Train Set Recall:'+str(re))
fm=f1_score(test_target, predicted, average='micro')
print('\n Train Set F-Measure:'+str(fm))