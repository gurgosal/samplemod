import pandas as pd
from pymongo import MongoClient

# client = MongoClient()
# client = MongoClient('mongodb://spark:hg7iib8w@34.207.228.12:27017/')
# client.the_database.authenticate('user', 'password', mechanism='SCRAM-SHA-1')
# uri = "mongodb://user:password@example.com/the_database?authMechanism=SCRAM-SHA-1"
# client = MongoClient(uri)

# client = MongoClient('34.207.228.12',27017)
client = MongoClient('mongodb://spark:hg7iib8w@34.207.228.12:27017')
# print client.database_names()
db = client.dg
# db.authenticate('spark', 'hg7iib8w', mechanism ='MONGODB-CR')
collections_db = db.collection_names()
collec_1 = db.events_live2
cursor = collec_1.find()
df_database = pd.DataFrame(list(cursor))

import sys
#import print from future
sys.path.append('/Users/gurpreetgosal/Dropbox/Work_DG/Packages')
import re
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'figure.autolayout': True})

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn import metrics
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier

from Data_Preprocessing.DataCleaningRaw import DataCleaningRaw
from Data_Preprocessing.ParsingNLP import ParsingNLP
from Data_Postprocessing.export_results import export_to_file
from Data_Postprocessing.export_results import model_performance
from Data_Preprocessing.data_collection import collection_ML



# import other files
watch_data = pd.read_csv("/Users/gurpreetgosal/Dropbox/Work_DG/Product_category/Data/ali_watches.csv")
watch_data.loc[:,'Category'] = pd.Series(np.repeat('Watches', len(watch_data.iloc[:,1])), index=watch_data.index)
#
menfashion_data = pd.read_csv("/Users/gurpreetgosal/Dropbox/Work_DG/Product_category/Data/ali_mensfashion.csv")
menfashion_data.loc[:,'Category'] = pd.Series(np.repeat('Mens Clothing', len(menfashion_data.iloc[:,1])),
                                              index=menfashion_data.index)
#
jewelry_data = pd.read_csv("/Users/gurpreetgosal/Dropbox/Work_DG/Product_category/Data/ali_jewelry.csv")
jewelry_data.loc[:,'Category'] = pd.Series(np.repeat('Jewelry', len(jewelry_data.iloc[:,1])), index=jewelry_data.index)
#
sportsoutdoors1_data = pd.read_csv("/Users/gurpreetgosal/Dropbox/Work_DG/Product_category/Data/ali_sportsoutdoors1.csv")
sportsoutdoors2_data = pd.read_csv("/Users/gurpreetgosal/Dropbox/Work_DG/Product_category/Data/ali_sportsoutdoors2.csv")
sportsoutdoors3_data = pd.read_csv("/Users/gurpreetgosal/Dropbox/Work_DG/Product_category/Data/ali_sportsoutdoors3.csv")
sportsoutdoors4_data = pd.read_csv("/Users/gurpreetgosal/Dropbox/Work_DG/Product_category/Data/ali_sportsoutdoors4.csv")
sportsoutdoors_comb_data = pd.concat([sportsoutdoors1_data, sportsoutdoors2_data, sportsoutdoors3_data, sportsoutdoors4_data])
sportsoutdoors_comb_data.loc[:,'Category'] = pd.Series(np.repeat('Sports & Outdoors',
                                        len(sportsoutdoors_comb_data.iloc[:,1])), index=sportsoutdoors_comb_data.index)

healthandbeauty_data = pd.read_csv("/Users/gurpreetgosal/Dropbox/Work_DG/Product_category/Data/ali_healthbeauty.csv")
healthandbeauty_data.loc[:,'Category'] = pd.Series(np.repeat('Health & Beauty',
                                                             len(healthandbeauty_data.iloc[:,1])), index=healthandbeauty_data.index)

wish_men_data = pd.read_csv("/Users/gurpreetgosal/Dropbox/Work_DG/Product_category/Data/wish_male_onetime.csv")
wish_women_data = pd.read_csv("/Users/gurpreetgosal/Dropbox/Work_DG/Product_category/Data/wish_female_onetime.csv")
wish_sports = pd.read_csv("/Users/gurpreetgosal/Dropbox/Work_DG/Product_category/Data/wish_hobby_onetime.csv")

#join
join_all = pd.concat([ watch_data, menfashion_data, jewelry_data, sportsoutdoors_comb_data, healthandbeauty_data ])

# remove duplicates
filtered_data = join_all.drop_duplicates('product_id')

# rename columns
filtered_data = filtered_data.rename(columns={'title': 'Name', 'specifics': 'Description'})
df_database = df_database.rename(columns={'name': 'Name', 'description': 'Description', 'category': 'Category'})
wish_women_data = wish_women_data.rename(columns={'title': 'Name', 'description': 'Description', 'category':'Category'})
#wish_women_data = wish_women_data.rename(columns={'title': 'Name', 'description': 'Description', 'category': 'Category'})

#drop columns
# for col_name in filtered_data:
#     if (col_name not in ['Name', 'Category', 'Description', 'Sub Category']):
#         filtered_data = DataCleaningRaw.drop_columns_pd(filtered_data, col_name)
for col_name in df_database:
    if (col_name not in ['Name', 'Category', 'Description', 'Sub Category']):
        df_database = DataCleaningRaw.drop_columns_pd(df_database, col_name)
for col_name in wish_men_data:
    if (col_name not in ['Name', 'Category', 'Description', 'Sub Category']):
        wish_men_data = DataCleaningRaw.drop_columns_pd(wish_men_data, col_name)
for col_name in wish_women_data:
    if (col_name not in ['Name', 'Category', 'Description', 'Sub Category']):
        wish_women_data = DataCleaningRaw.drop_columns_pd(wish_women_data, col_name)
for col_name in wish_sports:
    if (col_name not in ['Name', 'Category', 'Description', 'Sub Category']):
        wish_sports = DataCleaningRaw.drop_columns_pd(wish_sports, col_name)

# join with data from mongodb
# complete_unprocessed_data = pd.concat([filtered_data, df_database, wish_men_data])
for col_name in filtered_data.columns:
    if (col_name not in ['Name', 'Description','Category']):
        filtered_data = DataCleaningRaw.drop_columns_pd(filtered_data, col_name)
#complete_unprocessed_data = pd.concat([wish_women_data, df_database, wish_men_data, wish_sports, filtered_data])
complete_unprocessed_data = pd.concat([df_database, filtered_data])
# clean, parse, pos, lemmatize
data_features, categories = collection_ML.data_prep(complete_unprocessed_data)

# data_features = pickle.load(open('/Users/gurpreetgosal/Dropbox/Work_DG/Product_category/Processed_Data/data1.pkl','rb'))
# categories = pickle.load(open('/Users/gurpreetgosal/Dropbox/Work_DG/Product_category/Processed_Data/categories1.pkl','rb'))
# label encoder
le = preprocessing.LabelEncoder()
le.fit(list(set(categories)))
y_target = le.transform(categories)
# save encoder's classes


# save processed data as pickle

# train-test split
X_train, X_test, y_train, y_test = train_test_split(data_features, y_target, test_size = 0.20, random_state=0)

#ML Pipeline

# pipeline = Pipeline([
#     ('vect', TfidfVectorizer(max_df=0.5, ngram_range= (1,2))),
#     ('tfidf', TfidfTransformer()),
#     ('clf', SGDClassifier(loss='hinge', penalty='elasticnet', n_iter=5, random_state= 31, n_jobs= -1))
# ])
#
# parameters = {
#     #'vect__max_df': (0.5),
#     'vect__max_features': (5000, 10000),
#     #'vect__ngram_range': ((1, 2)), # unigrams or bigrams
#     'tfidf__use_idf': (True, False),
#     'tfidf__norm': ('l1', 'l2'),
#     'clf__alpha': ( 1e-2, 0.1, 1 )
# }

# # multiprocessing
# if __name__ == "__main__":
#
#     # find the best parameters for both the feature extraction and the classifier
#     grid_search = GridSearchCV(pipeline, parameters, verbose=1, cv =5)
#     print("Performing grid search...")
#     print("pipeline:", [name for name, _ in pipeline.steps])
#     print("parameters:")
#     print(parameters)
#
#     grid_search.fit(data_features, y_target)
#
#     print("Best score: %0.3f" % grid_search.best_score_)
#     print("Best parameters set:")
#     best_parameters = grid_search.best_estimator_.get_params()
#     for param_name in sorted(parameters.keys()):
#         print("\t%s: %r" % (param_name, best_parameters[param_name]))
#
# # Code to export the results
# data_export = pd.read_csv("Product_Seller_Data.csv")
# result = list(le.inverse_transform(grid_search.predict(test_features)))
# missing = missing_labels
#
# for col_name in data_export:
#     if (col_name not in ['Name','Sub Category', 'Category']) :
#         DataCleaningRaw.drop_columns_pd(data_export, col_name)
#
# result.reverse()
# missing = map(int, missing)
# data_export = data_export.iloc[missing,:]
#
# data_export.loc[:,'predicted category'] = pd.Series(result, index=data_export.index)
# #data_export.loc[:,'original row no.'] = pd.Series(missing, index=data_export.index)
# data_export.to_csv('test_with_labels.csv',sep = ',')
#
# joblib.dump(grid_search.best_estimator_, 'model1_SVC_OneVsRest.pkl', compress = 1)
#
# # to load model
# model = joblib.load(open('model1_SVC_OneVsRest.pkl'))


#--------------------------------------------------------------------
#  Random Forests Model

# ML Pipeline
pipeline = Pipeline([
    ('vect', TfidfVectorizer(max_df = 0.5, ngram_range = (1,2), max_features = 5000 )),
    ('tfidf', TfidfTransformer()),
    ('clf', RandomForestClassifier(n_jobs=-1, class_weight='balanced_subsample',
                                    max_depth=None,n_estimators = 200 )),
])

parameters = {
    #'vect__max_df': (0.5),
    #'vect__max_features': (5000, 10000),
    #'vect__ngram_range': ((1, 2)), # unigrams or bigrams
    'tfidf__use_idf': (True, False),
    'tfidf__norm': ('l1', 'l2'),
    'clf__max_features': ( 250, 500)
}
# multiprocessing
if __name__ == "__main__":
    # find the best parameters for both the feature extraction and the classifier
    grid_search = GridSearchCV(pipeline, parameters, verbose=20, cv = 5, scoring= 'f1_micro')
    print("Performing grid search...")
    print("pipeline:", [name for name, _ in pipeline.steps])
    print("parameters:")
    print(parameters)
    grid_search.fit(X_train, y_train)
    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))
y_pred = np.asarray(list(le.inverse_transform(grid_search.predict(X_test))))
y_test = np.asarray(list(le.inverse_transform(y_test)))

# Code to export the results

#data_export = pd.read_csv("Product_Seller_Data.csv")
#export_to_file.unseen_test_no_labels_present(missing_labels, data_export, y_pred)

# Analyze model performance

# Confusion Matrix
cnf_matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)
print('Confusion matrix, without normalization')
print(cnf_matrix)
class_names = list(set(categories))
f1 = f1_score(y_test, y_pred, average='micro')
print(f1)
print(metrics.classification_report(y_test, y_pred, class_names))

# Plot non-normalized confusion matrix
# plt.figure()
# model_performance.plot_confusion_matrix(cnf_matrix, classes=class_names,
#                       title='Confusion matrix, without normalization')
# # Plot normalized confusion matrix
# plt.figure()
# model_performance.plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
#                       title='Normalized confusion matrix')

# Store model in a file

# to load model
#model = joblib.load(open('model1_SVC_OneVsRest.pkl'))
joblib.dump(grid_search.best_estimator_, 'model_for_ali_trained_with_ali_only_RF_.pkl', compress = 1)
np.save('classes_RF_ali_only_feb13_2.npy', le.classes_)
model_1 = joblib.load(open('test_saving_loading.pkl'))
y_pred_1 = np.asarray(list(le.inverse_transform(model_1.predict(X_test))))






