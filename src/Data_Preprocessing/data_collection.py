import sys
#sys.path.append('/Users/gurpreetgosal/Dropbox/Work_DG/Packages')
import matplotlib
matplotlib.rcParams.update({'figure.autolayout': True})
from sklearn import preprocessing
from DataCleaningRaw import DataCleaningRaw
from ParsingNLP import ParsingNLP


class collection_ML:
   def __init__(self):
       print('using collection_ML Class')
       # self.data = data

   @classmethod
   def data_prep(cls, data_products, ali, wish):
       # get proposed category list
       fields = list(data_products)

       if ('Category' or 'category') not in fields:

           if ali == 1:

               # proposed_categories = [number.rstrip("\n") for number in proposed_categories]
               # clean_instance = DataCleaningRaw.DataCleaningRaw(data_products)
               for col_name in data_products.columns:
                   if (col_name not in ['Name', 'Description', 'name', 'description', 'title','specifics']):
                       DataCleaningRaw.drop_columns_pd(data_products, col_name)

               if 'name' in data_products.columns:
                   data_products.rename(columns={'name': 'Name'}, inplace=True)

               if 'title' in data_products.columns:
                   data_products.rename(columns={'title': 'Name'}, inplace=True)

               if 'description' in data_products.columns:
                   data_products.rename(columns={'description': 'Description'}, inplace=True)
               if 'Description' not in data_products.columns:
                   if 'specifics' in data_products.columns:
                       data_products.rename(columns={'specifics': 'Description'}, inplace=True)
                   data_initial = DataCleaningRaw.JoinCol(data_products, ['Name', 'Description'])
               elif 'Description' and 'specifics' in data_products.columns:
                   data_initial = DataCleaningRaw.JoinCol(data_products, ['Name', 'Description','specifics'])

           elif wish == 1:
               for col_name in data_products.columns:
                   if (col_name not in ['Name', 'Description', 'name', 'description', 'title', 'specifics']):
                       DataCleaningRaw.drop_columns_pd(data_products, col_name)

               if 'name' in data_products.columns:
                   data_products.rename(columns={'name': 'Name'}, inplace=True)

               if 'title' in data_products.columns:
                   data_products.rename(columns={'title': 'Name'}, inplace=True)

               if 'description' in data_products.columns:
                   data_products.rename(columns={'description': 'Description'}, inplace=True)

               if 'Description' not in data_products.columns:
                   if 'specifics' in data_products.columns:
                       data_products.rename(columns={'specifics': 'Description'}, inplace=True)
                   data_initial = DataCleaningRaw.JoinCol(data_products, ['Name', 'Description'])
               else:
                   data_initial = DataCleaningRaw.JoinCol(data_products, ['Name', 'Description'])

           # data_products['joined'] = data_products.loc[:, ['Name', 'Description']].apply(lambda x: '\n'.join(str(x)),
           #                                                                               axis=1)
           data_cleaned = ParsingNLP.FilterHTMLStopWordsLemmatize(data_initial)


           return data_cleaned

       else:

           with open('/Users/gurpreetgosal/Dropbox/Work_DG/Product_category/Data/category_list.txt', 'r') as f:
               proposed_categories = f.read().splitlines()
           # proposed_categories = [number.rstrip("\n") for number in proposed_categories]
           # clean_instance = DataCleaningRaw.DataCleaningRaw(data_products)

           # data_products['joined'] = data_products.loc[:, ['Name', 'Description']].apply(lambda x: '\n'.join(str(x)),axis=1)
           data_initial = DataCleaningRaw.JoinCol(data_products, ['Name', 'Description'])
           # for i in xrange(0, len(data_initial)):
           #     if isinstance(data_initial[i], unicode):
           #         try:
           #             # print data_initial[i]
           #             data_initial[i] = str(data_initial[i])
           #             # print data_initial[i]
           #             # print '\n\n\n'
           #         except:
           #             # data_initial[i] = data_initial[i].encode('utf-8')
           #             print data_initial[i]

           data_targets = data_products.loc[:, ['Category']]

           categories = data_targets['Category'].tolist()
           # if  category missing add that data point to missing data which can be later used as test set
           categories, missing_labels, unwanted_category_id = DataCleaningRaw.get_category_2(categories,
                                                                                           proposed_categories)

           data_cleaned, missing_test_data = DataCleaningRaw.delete_na_items(data_initial, missing_labels)
           categories, missing_test_false  = DataCleaningRaw.delete_na_items(categories, missing_labels)

           # perform parsing

           # remove html tags
           data_cleaned = ParsingNLP.FilterHTMLStopWordsLemmatize(data_cleaned)
           # test_cleaned = ParsingNLP.FilterText(missing_test_data)

           # pos tagging
           # data_pos = ParsingNLP.StopWords_PosTag(data_cleaned)
           # test_pos = ParsingNLP.StopWords_PosTag(test_cleaned)
           # lemmatization
           # data_features = ParsingNLP.LemmatizeData(data_pos)
           # test_features = ParsingNLP.LemmatizeData(test_pos)

           return data_cleaned, categories






