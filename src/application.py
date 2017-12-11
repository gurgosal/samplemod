import sys
import importlib
import pandas as pd
import numpy as np
import matplotlib
from sklearn.externals import joblib

matplotlib.rcParams.update({'figure.autolayout': True})
from sklearn import preprocessing
from pymongo import MongoClient
import sys

sys.path.append('./src/Data_Preprocessing')
from Data_Preprocessing.DataCleaningRaw import DataCleaningRaw
from Data_Preprocessing.ParsingNLP import ParsingNLP
from Data_Preprocessing.data_collection import collection_ML

# get unlabeled products from mongo
client = MongoClient('mongodb://bridge:Tfm_321@54.165.83.71/dg')
# print client.database_names()
db = client.dg
# db.authenticate('spark', 'hg7iib8w', mechanism ='MONGODB-CR')
collections_db = db.collection_names()
collec_1 = db.unlabeled_products
cursor = collec_1.find()
df_database = pd.DataFrame(list(cursor))
print("data fetcehd from db")

if __name__ == '__main__':
    print("inside main")
    import csv


    # load trained model from Models directory
    model_1 = joblib.load(open('./src/Models/ali_model_with wishali_data_RF/'
                               'model_for_ali_trained_with_wish_and_ali_RF_.pkl'))
    # model_2 = joblib.load(open('/Users/gurpreetgosal/Dropbox/Work_DG/Product_category/Models/wish_model/'
    #                            'model_runwith_wishdata_RF2_.pkl'))
    models = [model_1]
    print("got models")
    # import new data
    test_new = df_database
    data_arch = df_database

    # preprocess test data
    test_new_X = collection_ML.data_prep(test_new, ali=0, wish=1)
    print("preped test data")
    le1 = preprocessing.LabelEncoder()
    le1.classes_ = np.load('./src/Models/ali_model_with wishali_data_RF/classes_RF_ali_wish_feb12_1.npy')
    le = [le1]
    print("compute ypred")
    y_pred = np.asarray(list(le[0].inverse_transform(models[0].predict(test_new_X))))
    y_pred_code = np.zeros(len(y_pred), dtype='|S1')
    print("ypred computed")
    # read category code file
    with open('./src/category.csv', mode='r') as infile:
        reader = csv.reader(infile)
        category_code_dict = {rows[0]: rows[1] for rows in reader}
    print("got categories from file")
    for i in range(len(y_pred)):
        item_cat = y_pred[i]
        if item_cat == "Auto Accessories":
            item_cat = "Sports & Outdoors"
        code = category_code_dict.keys()[category_code_dict.values().index(item_cat)]
        y_pred_code[i] = code
    print("assigned codes to ypred")
    out_df = data_arch
    out_df['PredictedCategoryCode'] = y_pred_code
    out_df['PredictedCategoryName'] = y_pred
    print("update collection start")
    for idx, row in out_df.iterrows():
        result_update = db.unlabeled_products.update_one({'product_id': row.product_id},
                                                         {'$set': {'predicted_category_code': row.PredictedCategoryCode,
                                                                   'predicted_category_name': row.PredictedCategoryName}})
    print("finished updating")


