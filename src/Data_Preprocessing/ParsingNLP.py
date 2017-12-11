from nltk.corpus import stopwords
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
import re
import string
# import sys
# reload(sys)
# sys.setdefaultencoding('utf8')

class ParsingNLP:

   def __init__(self):
       print('using ParsingNLP Class')

   @classmethod
   def StopWords_PosTag(cls, data):
       data_processed = []
       stop = set(stopwords.words('english'))
       stop = stop | {'nan'}
       for item in data:
           item = ''.join([i for i in item if not i.isdigit()])
           item = ([i for i in item.lower().split() if i not in stop])
           data_processed.append(nltk.pos_tag(item))
       return data_processed

   @classmethod
   def FilterText(cls,data):
       raw_html = data
       cleanr = re.compile('<.*?>')
       cleantext = []
       for item in raw_html:
           item = re.sub(cleanr, '', str(item))
           item = item.replace('\n', ' ')
           item.translate(None, string.punctuation)
           item = re.sub(r'[^\w\s]', '', item)
           cleantext.append(item)
       return cleantext



   @classmethod
   def LemmatizeData(cls, data_processed):
       data_processed2 = []
       lemmatizer = WordNetLemmatizer()
       for item in (data_processed):
           lemat_item = []
           for word in item:
               word = lemmatizer.lemmatize(word[0], ParsingNLP.get_wordnet_pos(word[1]))
               lemat_item.append(word.lower())
           data_processed2.append(' '.join(lemat_item))
       return data_processed2

   @classmethod
   def get_wordnet_pos(cls, treebank_tag):
       if treebank_tag.startswith('J'):
           return u'a'
       elif treebank_tag.startswith('V'):
           return u'v'
       elif treebank_tag.startswith('N'):
           return u'n'
       elif treebank_tag.startswith('R'):
           return u'r'
       elif treebank_tag.startswith('S'):
           return u's'
       else:
           return u'n'

   @classmethod
   def FilterHTMLStopWordsLemmatize(cls, data_df):
       data_processed = []
       index = 0
       for raw_html in data_df:

           # if index > 2000:
           #     print index
           # clean html
           # cleanr = re.compile('<.*?>')
           # print type(raw_html)
           # raw_html = re.sub(cleanr, ' ', unicode(str(raw_html)))
           raw_html = re.sub('[a-z]+:', '', raw_html)
           raw_html = re.sub('<[^>]*>', ' ', raw_html)
           raw_html = raw_html.replace(r'\n', ' ')
           # raw_html = raw_html.translate(None, string.punctuation)
           punc_re = re.compile('[%s]' % re.escape(string.punctuation))
           raw_html = punc_re.sub(' ', raw_html)
           clean_html = re.sub(r'[^\w\s]', ' ', raw_html)
           url_re = re.compile('https?://(www.)?\w+\.\w+(/\w+)*/?')
           alpha_num_re = re.compile("^[a-z0-9_.]+$")
           num_re = re.compile('(\\d+)')
           clean_html = url_re.sub(' ', clean_html)
           clean_html = num_re.sub(' ', clean_html)


           # clean_html = BeautifulSoup(clean_html).get_text()
           # # stop words

           # remove non a-z 0-9 characters and words shorter than 3 characters
           list_pos = 0
           cleaned_str = ''
           for word in clean_html.split():
               if list_pos == 0:
                   if len(word) > 2:
                       cleaned_str = word
                   else:
                       cleaned_str = ' '
               else:
                   if len(word) > 2:
                       cleaned_str = cleaned_str + ' ' + word
                   else:
                       cleaned_str += ' '
               list_pos += 1

           clean_html = cleaned_str

           stop = set(stopwords.words('english'))
           stop = stop | {'nan', 'MSRP', 'msrp', 'price', 'Price', 'nbsp', 'msp'}
           clean_html = ''.join([i for i in clean_html if not i.isdigit()])
           clean_stophtml = ([i for i in clean_html.lower().split() if i not in stop])
           clean_stophtml = nltk.pos_tag(clean_stophtml)
           # print type(clean_stophtml)
           # Lemmatization
           lemmatizer = WordNetLemmatizer()
           lemat_item = []
           for word in clean_stophtml:
               word = lemmatizer.lemmatize(word[0], ParsingNLP.get_wordnet_pos(word[1]))
               lemat_item.append(word.lower())
           # print type(lemat_item)
           clean_lemmatized = ' '.join(lemat_item)

           # print type(clean_lemmatized)
           data_processed.append(clean_lemmatized)
           index += 1
       return data_processed




