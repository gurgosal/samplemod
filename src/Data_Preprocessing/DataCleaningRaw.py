import re
import string
import pandas as pd


class DataCleaningRaw:
   def __init__(self):
       print('using DataCleaningRaw Class')
       # self.data = data

   @classmethod
   def drop_columns_pd(cls, data, col_name):
       df = data
       df = df.drop(col_name, axis=1)
       return df



   @classmethod
   def get_category(cls, category_list, proposed_categories):
       category_refined = []
       missing_labels = []
       unwanted_category_id = []
       i = 0
       for item in category_list:
           try:
               if item.find('|') != -1:
                   text = item.split("|")
                   flag = 0
                   for resolved_category in text:
                       if resolved_category in proposed_categories:
                           category_refined.append(resolved_category)
                           flag = 1 + flag
                           break
                   if flag ==0:
                       missing_labels.append(i)
                       category_refined.append('na')
               else:
                   if item in proposed_categories:
                       category_refined.append(item)
                   elif item == 'Earrings':
                       category_refined.append('Jewelry')
                   else:
                       unwanted_category_id.append(i)

           except:
               print 'error occured in refining categories', item, i
               missing_labels.append(i)
               category_refined.append('na')
           i = i + 1
       return category_refined, missing_labels, unwanted_category_id

   @classmethod
   def get_category_2(cls, category_list, proposed_categories):
       missing_labels = []
       unwanted_category_id = []
       i = 0
       for item in category_list:
           try:
               if isinstance(item,str):
                   item = [item]

               flag = 0
               if 'Watches' in item:
                    item = []
                    item.append('Watches')

               if 'Fashion' in item:
                   indexF = item.index('Fashion')
                   item[indexF] = 'Fashion Accessories'

               if 'Auto Accessories ' in item:
                    item = []
                    item.append('Auto Accessories')

               if 'Phone Accessories ' in item:
                    item = []
                    item.append('Phone Accessories')

               if ('Gadgets' in item) & ('Electronics' not in item) & ('Phone Accessories ' not in item) :
                    item = []
                    item.append('Electronics')

               if 'Earrings' in item:
                    item = []
                    item.append('Jewelry')

               if 'Kitchen' in item:
                   item = []
                   item.append('Kitchen')

               if 'Health & Personal Care' in item:
                   item = []
                   item.append('Health & Beauty')

               if 'Beauty' in item:
                   item = []
                   item.append('Health & Beauty')

               if 'Mens Clothing' in item:
                   item = []
                   item.append('Mens Clothing')

               if 'Womans Clothing' in item:
                   item = []
                   item.append('Womans Clothing')

               for j in xrange(0, len(item)):
                   if item[j] in proposed_categories:
                       category_list[i] = item[j]
                       flag = 1 + flag
                       break

               if flag == 0:
                   unwanted_category_id.append(i)
                   missing_labels.append(i)
                   category_list[i] = 'na'

           except:
               print 'error occured in refining categories', item, i
               missing_labels.append(i)
               category_list[i] = 'na'
           i = i + 1
       return category_list, missing_labels, unwanted_category_id

   @classmethod
   def delete_na_items(cls, data, indices):
       test_data = []
       for i in sorted(indices, reverse=True):
           test_data.append(data[i])
           del data[i]
       return data, test_data

   @classmethod
   def InheritSubCategory(cls, data, proposed_category):
       category    = data['Category'].tolist()
       subcategory = data['Sub Category'].tolist()
       joined_categ = zip(category, subcategory)
       filled_categories = category
       for i in xrange(0,len(category)):
           item = joined_categ[i]
           if (str(item[0]) == 'nan') & (str(item[1]) != 'nan'):
               filled_categories[i] = item[1]
           else:
               sub_item = str(item[0]).split('|')
               sub_item2 = str(item[1]).split('|')
               sub_item_presence = []
               for ele in sub_item:
                   sub_item_presence.append(ele in proposed_category)
               if True not in sub_item_presence:
                   filled_categories[i] = item[1]
               if ('Electronics' in str(sub_item).strip()) & ('Phone Accessories' in str(sub_item2).strip()):
                   filled_categories[i] = 'Phone Accessories '
       return filled_categories

   @classmethod
   def JoinCol(cls, data_products, colnames):
       Name = data_products[colnames[0]].tolist()
       Description = data_products[colnames[1]].tolist()
       joined = []

       for i in xrange(0, len(Name)):
           try:
               item = str((Name[i])) + ' ' + str((Description[i]))
               joined.append(item)
           except:
               item = unicode((Name[i])) + unicode(' ') + unicode((Description[i]))
               joined.append(item)
       return joined











