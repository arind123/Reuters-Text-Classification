# arinML1
Data
We will consider the Reuters-21578 data set. It is new wire of 1987.The documents were originally
assembled and ordered with categories by Carnegi Group Inc. and Returns Ltd. The corpus
originally contains 135 categories and the categories are overlapped i.e., one document may exists
in several categories. Hence we will consider the Mod Apte version of Renters.The ModApte
version contains 12902 documents with 90 categories and the corpus is divided into training and
test sets. In this assignment we will consider only the following 10 categories of the ModApte
version of Returns-21578 corps:
alum, barley, coffee, dmk, fuel, livestock, palm-oil, retail, soybean, veg-oil
The class label of a document is the name of the directory to which it belongs to.

Tasks

1.Remove the stop words from the raw text after tokenization and then map each token to its
stem using a stemming algorithm.
2.Create the term-document matrix by using an efficient tf-idf weighting scheme.
3.Perform Naive bayes, logistic regression, support vector machine and multilayer perceptron
classifiers to categorize the documents in the test set of the given Reuters corpus. The aim
is to achieve the best performance for each of these classifiers by properly tuning its
parameters and by choosing an efficient version of the classifiers using the training set of the
given corpus. Properly explain your selection using experimental results. You may consider
different types of feature combinations e.g., unigrams , bigrams etc.
4. Evaluate and compare the performance of the classifiers using the actual class labels of the
test samples. Properly explain and analyze the results. Discuss about significant findings
from these results. Conclude with future scope of research (if any) of this assignment.
