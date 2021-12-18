import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import codecs
from model.crf import *
from model.lm_lstm_crf import *
import model.utils as utils
from model.predictor import predict_wc

import pandas as pd
import json
import re
import ast
from os import listdir, walk
from os.path import isfile, join
from sklearn.datasets import load_svmlight_file
from nltk import sent_tokenize


def report2dict(cr):
    """
    This function is to transform the free text result from classification_result to indexed format dataframe
    :param cr: the text result from classification_result()
    :return: dataframe format for the result
    """
    # Parse rows
    rows = cr.split("\n")
    df = pd.DataFrame(columns=['metric'] + rows[0].split())
    for r in rows[1:]:
        if r != '':
            print(r.split())
            df.loc[df.shape[0]] = r.split()[-5:]
    return df


def read_csv(input_file):
    """
    Deal with reading big csv files
    :param input_file:
    :return:
    """
    with open(input_file, 'r') as f:
        a = f.readline()
    csv_list = a.split(',')
    tsv_list = a.split('\t')
    if len(csv_list) > len(tsv_list):
        sep = ','
    else:
        sep = '\t'

    reader = pd.read_csv(input_file, iterator=True, low_memory=False, delimiter=sep, encoding='ISO-8859-1')
    loop = True
    chunk_size = 100000
    chunks = []
    while loop:
        try:
            chunk = reader.get_chunk(chunk_size)
            chunks.append(chunk)
        except StopIteration:
            loop = False
    df = pd.concat(chunks, ignore_index=True)
    return df


def get_filenames(path, format_files, include='', exclude='#DUMMYSTRING'):
    """
    Get file names in a folder
    :param path: folder path
    :param format_files: file extension
    :param include: include a specific string
    :param exclude: exclude a specific string
    :return: a list of file names with full path
    """
    return sorted([join(path, f) for f in listdir(path) if
            isfile(join(path, f)) and f[-len(format_files):] == format_files and f.find(include) > -1 and f.find(exclude) < 0])


def get_filenames_from_root(path, format_files, filtered_with=''):
    """
    Get all file names from a root path, including files in sub-directories
    :param path: root path
    :param format_files: file extension
    :param filtered_with: filter with specific values
    :return: a list of file names with full path
    """
    filenames = []
    for path, subdirs, files in walk(path):
        for name in files:
            if name[-3:] == format_files and name.find(filtered_with) > -1:
                filenames.append(join(path, name))

    return filenames


def get_regex_patterns(file_path):
    file = open(file_path)
    patterns = ''
    for line in file:
        pa = line.replace('\n', '').split('\t')[0]
        if pa[0] != '#':
            patterns = patterns + line.replace('\n', '').split('\t')[0] + '|'

    return patterns[:-1]


def load_config(path):
    with open(path) as json_file:
        data = json.load(json_file)
        pii_entity = data['Feature']
        constant_variable = data['ConstantVariable']

        return constant_variable, pii_entity


def read_dataset_1(filename):
    with open(filename, "r") as f:
        for line in f:
            sents = sent_tokenize(line)
            yield [s for s in sents]


def read_dataset_2(filename):
    with open(filename, "r") as f:
        for line in f:
            yield line.split('\t')


def get_labels_1(filename):
    """
    Get labels from a file where each keyphrase is per row without document id
    :param filename:
    :return: a list of keyphrases
    """
    with open(filename, "r") as f:
        yield [line[:-1] for line in f]


def get_labels_2(filename):
    """
    Get labels from a file where each row contains [doc_id] and a list if keyphrases
    :param filename:
    :return: a dictionary of doc ids and keyphrases ('doc_id': 'list of keyphrases')
    """

    dic = dict()
    with open(filename, "r") as f:
        for line in f:
            k, v = line.split('\t')
            dic[k] = ast.literal_eval(v.lower())

    return dic


def get_data(file_name):
    data = load_svmlight_file(file_name)
    return data[0], data[1]


def update_pos_pattern(filename, docs, labels, model="en_core_web_sm", N=4):
    import spacy

    nlp = spacy.load(model)
    r = re.compile(get_regex_patterns(filename), flags=re.I)
    patterns = []
    pattern_exam = dict()
    for doc in docs:
        sents = sent_tokenize(doc[1])
        for sen in sents:
            s = nlp(sen)
            for n in range(N):
                for i in range(len(s) - n):
                    tags = [s[j].tag_ for j in range(i, i + n + 1)]
                    candidate = str(s[i:(i + n + 1)]).lower()
                    if r.match(' '.join(tags)) is None and candidate in labels[doc[0]]:
                        p = ' '.join(tags) + '$'
                        if p not in patterns:
                            pattern_exam[p] = candidate
                            patterns.append(p)

    with open(filename, 'a') as f:
        for pa in patterns:
            f.write(pa + '\t' + pattern_exam[pa])
            f.write('\n')


def plot_ROC(fpr, tpr, index, title):
    import matplotlib.pyplot as plt

    plt.figure(index)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr)
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title(title)
    plt.show()


def report2dict(cr):
    """
    This function is to transform the free text result from classification_result to indexed format dataframe
    :param cr: the text result from classification_result()
    :return: dataframe format for the result
    """
    # Parse rows
    rows = cr.split("\n")
    df = pd.DataFrame(columns=['label'] + rows[0].split())
    for r in rows[1:4]:
        if r != '':
            df.loc[df.shape[0]] = r.split()[-5:]
    return df


def read_data_from_sequence_tagging_format(file_path):
    texts = []
    tags = []

    with open(file_path, 'r') as f:
        _text = []
        _tag = []
        for r in f:
            if '-DOCSTART-' not in r:
                if r == '\n':
                    texts.append(_text)
                    tags.append(_tag)
                    _text = []
                    _tag = []
                else:
                    l = r.replace('\n','').split(' ')
                    _text.append(l[0])
                    _tag.append(l[-1])
    return texts, tags


def combine_predicted_lists(l1, l2):
    l = []
    O_token = True

    for i in range(len(l1)):
        if l1[i]==l2[i]:
            l.append(l1[i])
        else:
            if O_token:
                l.append('B-KEY')
            else:
                l.append('I-KEY')

        O_token = True if l[i]=='O' else False
    
    return l


def combine_two_true_predictions(pred1, pred2):
    true_predictions_combined = []
    for idx in range(len(pred1)):
        true_predictions_combined.append(combine_predicted_lists(pred1[idx], pred2[idx]))
        
    return true_predictions_combined


def refine_prediction_list(predictions):
    new_predictions = []
    for j, _l in enumerate(predictions):
        pre_token = 'O'
        new_l = []
        for i in range(len(_l)):
            if (pre_token=='I-KEY' or pre_token=='B-KEY') and _l[i]=='B-KEY':
                new_l.append('I-KEY')
            else:
                new_l.append(_l[i])
            pre_token = _l[i]
        new_predictions.append(new_l)
        
    return new_predictions

def refine_prediction(_l):
    pre_token = 'O'
    new_l = []
    for i in range(len(_l)):
        if (pre_token=='I-KEY' or pre_token=='B-KEY') and _l[i]=='B-KEY':
            new_l.append('I-KEY')
        else:
            new_l.append(_l[i])
        pre_token = _l[i]
  
    return new_l


def get_predictions(model_name, model_arg, _features, gpu = -1):
    with open(model_arg, 'r') as f:
        jd = json.load(f)
    jd = jd['args']
    
    checkpoint_file = torch.load(model_name, map_location=lambda storage, loc: storage)
    f_map = checkpoint_file['f_map']
    l_map = checkpoint_file['l_map']
    c_map = checkpoint_file['c_map']
    in_doc_words = checkpoint_file['in_doc_words']

    if gpu >= 0:
        torch.cuda.set_device(gpu)
        
    # build model
#     print('loading model')
    ner_model = LM_LSTM_CRF(len(l_map), len(c_map), jd['char_dim'], jd['char_hidden'], jd['char_layers'], jd['word_dim'], jd['word_hidden'], jd['word_layers'], len(f_map), jd['drop_out'], large_CRF=jd['small_crf'], if_highway=jd['high_way'], in_doc_words=in_doc_words, highway_layers = jd['highway_layers'])
    ner_model.load_state_dict(checkpoint_file['state_dict'])
    
    if gpu >= 0:
        if_cuda = True
        torch.cuda.set_device(gpu)
        ner_model.cuda()
    else:
        if_cuda = False
        
    decode_label = ('label' == 'label')
    predictor = predict_wc(if_cuda, f_map, c_map, l_map, f_map['<eof>'], c_map['\n'], l_map['<pad>'], l_map['<start>'], decode_label, 16, jd['caseless'])
    
    predictions = predictor.output_batch(ner_model, _features)
    
    return predictions


def filtering_title_concepts(_output_id_concept):
    '''
    Remove concepts that contain articles, conjunctions or prepositions
    '''
    
     # list of articles
    articles = ["a", "an", "the"]
      
    # list of coordinating conjunctins
    conjunctions = ["and", "but",
                    "for", "nor",
                    "or", "so",
                    "yet"]
      
    # list of some short articles
    prepositions = ["in", "to", "for", 
                    "with", "on", "at",
                    "from", "by", "about",
                    "as", "into", "like",
                    "through", "after", "over",
                    "between", "out", "against", 
                    "during", "without", "before",
                    "under", "around", "among",
                    "of"]
      
    # merging the 3 lists
    lower_case = articles + conjunctions + prepositions
            
    for k, v in _output_id_concept.items():
        new_concept_list = []
        for c in v:
            # variable declaration for the output text 
            output_string = ""

            # separating each word in the string
            if type(c)==str:
                input_list = c.split(" ")
            else: ## NOT PROCESS THE CONCEPTS WITH POSITIONS FOR NOW. WHEN DOING IT, NEED TO UPDATE THE POSITIONS AS WELL
                input_list = c[0].split(" ")
                
            # checking each word
            for idx, word in enumerate(input_list):

                # if the word exists in the list
                # then no need to capitalize it
                if word in lower_case:
                    if idx != 0 and idx != len(input_list):
#                     print(word)
                        output_string += word.lower() + " "

                # if the word does not exists in
                # the list, then capitalize it
                else:
                    output_string += word + " "
                    
            if len(output_string) > 1:
                new_concept_list.append(output_string[:-1])
        _output_id_concept[k] =  new_concept_list
    return _output_id_concept


def generateTitleCase(input_string):
      
    # list of articles
    articles = ["a", "an", "the"]
      
    # list of coordinating conjunctins
    conjunctions = ["and", "but",
                    "for", "nor",
                    "or", "so",
                    "yet"]
      
    # list of some short articles
    prepositions = ["in", "to", "for", 
                    "with", "on", "at",
                    "from", "by", "about",
                    "as", "into", "like",
                    "through", "after", "over",
                    "between", "out", "against", 
                    "during", "without", "before",
                    "under", "around", "among",
                    "of"]
      
    # merging the 3 lists
    lower_case = articles + conjunctions + prepositions
      
    # variable declaration for the output text 
    output_string = ""
      
    # separating each word in the string
    input_list = input_string.split(" ")
      
    # checking each word
    for word in input_list:
          
        # if the word exists in the list
        # then no need to capitalize it
        if word.lower() in lower_case:
            output_string += word.lower() + " "
              
        # if the word does not exists in
        # the list, then capitalize it
        else:
            temp = word.title()
            output_string += word + " "
              
      
    return output_string[:-1]


def extraction_concepts_from_token_label(sub_s, idx_temp, token_label):
    extracted_concepts = []

    ## Retrieve extracted concept indices (combining separate labeled tokens into complete concepts)
    concept_indices = []
    start_idx = -1
    end_idx = -1
    flag = False
    for idx in range(0, len(token_label)):
        if token_label[idx][2] not in ['O', 'B-LOC', 'I-LOC']:
            flag = True
            if start_idx==-1:
                start_idx = token_label[idx][1][0]
            end_idx = token_label[idx][1][1]
        elif flag==True:
            while start_idx!=0 and sub_s[start_idx-1].isalpha():
                start_idx -= 1
            while sub_s[end_idx].isalpha():
                end_idx += 1
            
            if sub_s[start_idx]==' ': # Handle the case the concept starts with a space (e.g., ' problem analysis')
                start_idx += 1
            concept_indices.append((start_idx, end_idx))
            start_idx = -1
            end_idx = -1
            flag = False

    ## Retrieve the concept strings from the indices
    for idx_pair in concept_indices:
        if idx_pair[1] - idx_pair[0] > 1:
            extracted_concepts.append((sub_s[idx_pair[0]:idx_pair[1]], (idx_pair[0]+idx_temp, idx_pair[1]+idx_temp)))
        
    return extracted_concepts


def initialize_models(model_name, model_arg, gpu = -1):
    with open(model_arg, 'r') as f:
        jd = json.load(f)
    jd = jd['args']
    
    checkpoint_file = torch.load(model_name, map_location=lambda storage, loc: storage)
    f_map = checkpoint_file['f_map']
    l_map = checkpoint_file['l_map']
    c_map = checkpoint_file['c_map']
    in_doc_words = checkpoint_file['in_doc_words']

    if gpu >= 0:
        torch.cuda.set_device(gpu)
        
    # build model
#     print('loading model')
    ner_model = LM_LSTM_CRF(len(l_map), len(c_map), jd['char_dim'], jd['char_hidden'], jd['char_layers'], jd['word_dim'], jd['word_hidden'], jd['word_layers'], len(f_map), jd['drop_out'], large_CRF=jd['small_crf'], if_highway=jd['high_way'], in_doc_words=in_doc_words, highway_layers = jd['highway_layers'])
    ner_model.load_state_dict(checkpoint_file['state_dict'])
    
    if gpu >= 0:
        if_cuda = True
        torch.cuda.set_device(gpu)
        ner_model.cuda()
    else:
        if_cuda = False
        
    decode_label = ('label' == 'label')
    predictor = predict_wc(if_cuda, f_map, c_map, l_map, f_map['<eof>'], c_map['\n'], l_map['<pad>'], l_map['<start>'], decode_label, 16, jd['caseless'])
    
    return ner_model, predictor
