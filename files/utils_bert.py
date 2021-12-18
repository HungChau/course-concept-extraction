import pandas as pd
import json
import pickle
import re
import ast
import torch
from os import listdir, walk
from os.path import isfile, join
from sklearn.datasets import load_svmlight_file
from nltk import sent_tokenize
from scipy import stats
from scipy.stats import pearsonr
import numpy as np
import random

import seaborn as sns
import matplotlib.pyplot as plt


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


class WNUTDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


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


def find_str(s, exclude_list):
    
    for ex_s in exclude_list:
        if s.find(ex_s) >= 0:
            return True
    
    return False


def get_filenames_from_root(path, format_files, filtered_with='', exclude_list=[]):
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
            if name[-3:] == format_files and name.find(filtered_with) > -1 and find_str(name, exclude_list)==False:
                filenames.append(join(path, name))

    return filenames


def write_pickle(obj, filename):
    pickle.dump(obj, open(filename, 'wb'))
    
    
def read_pickle(filename):
    return pickle.load(open(filename, 'rb'))


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


def update_pos_pattern(filename, docs, labels, N = 4):
    import spacy

    # nlp = spacy.load("en_core_web_sm")
    nlp = spacy.load("en_ner_bionlp13cg_md")
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


def data_summary(data):
    print(pd.Series(list(data)).describe())
    sns.distplot(data)
    plt.show()


def summarize_target(data, target):
    """
    Summarize the target's properties
    :param data: DataFrame
    :param target: name of the dependent variable
    :return: None
    """
    print('Data summary:')
    print(data[target].describe())

    k2, p = stats.normaltest(data[target])
    alpha = 1e-3
    print("\nNormality test: p = {:g}".format(p))

    if p < alpha:  # null hypothesis: x comes from a normal distribution
        print("The null hypothesis can be rejected")
    else:
        print("The null hypothesis cannot be rejected")

    data.hist([target], bins=50)

    fig, ax = plt.subplots(figsize=(6, 4),dpi=100)

    ax=sns.boxplot(y=target, data=data)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    plt.show()


def standardize(data, _columns=[]):
    """
    Standardize features of a list of DataFrames
    :param data_k: <list> of DataFrames
    :return: <list> of new DataFrames
    """
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    data_temp = data.copy()
    scaler.fit(data_temp.loc[:, _columns])
    data_temp.loc[:, _columns] = scaler.transform(data_temp.loc[:, _columns])
    return data_temp


def bin_box_plots_corr_analysis(data, x_col, y_col, bins=20, title=None):
    """
    Scatter plot and boxplot and calculate the correlation and p value

    :param data: DataFrame
    :param x_col: column name of the independent variable
    :param y_col: column name of the dependent variable
    :param bins: number of bins
    :param title: name of the market
    :return: None
    """

    print('rows: {}'.format(len(data)))
    data = data[[x_col, y_col, 'score_diff']].fillna(0.0)

    x = data[x_col].values
    y = data[y_col].values
    x_bin = pd.cut(x, bins=bins, precision=0)
    data['bin'] = x_bin
    plt.figure(figsize=(9, 6))
    g = sns.boxplot(x_bin, y)
    g.set_xticklabels(g.get_xticklabels(), rotation=45, horizontalalignment='right')
    g.set_xlabel(x_col)
    g.set_ylabel(y_col)
    if title:
        g.set_title(title)

    plt.show()

    ax = sns.pairplot(data[['dwa', 'bin', 'score_diff']], hue="bin", height=5, kind="reg", palette=sns.color_palette("tab20"), plot_kws=dict(scatter_kws=dict(s=3)))
    for lh in ax._legend.legendHandles: 
        lh.set_alpha(1)
        lh._sizes = [50] 
        
    plt.show()

#     corr, corr_p = pearsonr(x, y)
#     print('correlation: {}, with p value: {}'.format(corr, corr_p))


def subset_train_data(texts, tags, ratio_1=0.5, ratio_2=0.1, verbose=False):
    '''
    Select randomly 'ratio_1' of data sentences which has less than 50 tokens and has at least one concepts, plus random 'raito_2' of sentences which has no concepts
    '''
    indices = []
    for idx in range(len(texts)):
        if len(texts[idx]) < 50 and 'B-KEY' in tags[idx]:
            indices.append(idx)
    if verbose:
        print(len(indices))

    # This block is to add sentences that do not have any concept, and select a half of the sentences (to speed up training)
    indices = random.sample(indices, int(len(indices)*ratio_1))
    if verbose:
        print(len(indices))

    random.seed(7)
    indices = indices + random.sample(list(set(range(len(texts))) - set(indices)), int(len(indices)*ratio_2))
    if verbose:
        print(len(indices))
    
    texts = [texts[i] for i in indices]
    tags = [tags[i] for i in indices]
    
    return texts, tags


def subset_val_data(texts, tags, ratio=0.1, verbose=False):
    '''
    Select randomly 'ratio' of data sentences which has less than 50 tokens
    '''
    indices = []
    for idx in range(len(texts)):
        if len(texts[idx]) < 50:
            indices.append(idx)
    
    if verbose:
        print(len(indices))

    subset_indices = random.sample(indices, int(len(indices)*ratio))
    texts = [texts[i] for i in subset_indices]
    tags = [tags[i] for i in subset_indices]
    
    return texts, tags


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    
    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


class WNUTDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
    

def encode_tags(_tags, encodings, _tag2id):

    labels = [[_tag2id[tag] for tag in doc] for doc in _tags]
    encoded_labels = []
    indices = []
    count = 0
    for doc_labels, doc_offset in zip(labels, encodings.offset_mapping):
        
        # create an empty array of -100
        doc_enc_labels = np.ones(len(doc_offset),dtype=int) * -100
        arr_offset = np.array(doc_offset)

#         set labels whose first offset position is 0 and the second is not 0
        if len(doc_enc_labels[(arr_offset[:,0] == 0) & (arr_offset[:,1] != 0)])==len(doc_labels):
            doc_enc_labels[(arr_offset[:,0] == 0) & (arr_offset[:,1] != 0)] = doc_labels
            encoded_labels.append(doc_enc_labels.tolist())
            indices.append(count)
            
        count += 1  
    
    new_encodings = {key: torch.tensor([val[i] for i in indices]) for key, val in encodings.items()}
    
    return encoded_labels, new_encodings, indices


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


def extraction_concepts_from_token_label(sub_s, idx_temp, token_label):
    extracted_concepts = []

    ## Retrieve extracted concept indices (combining separate labeled tokens into complete concepts)
    concept_indices = []
    start_idx = -1
    end_idx = -1
    flag = False
    for idx in range(1, len(token_label)-1):
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
            concept_indices.append((start_idx, end_idx))
            start_idx = -1
            end_idx = -1
            flag = False

    ## Retrieve the concept strings from the indices
    for idx_pair in concept_indices:
        if idx_pair[1] - idx_pair[0] > 1:
            extracted_concepts.append((sub_s[idx_pair[0]:idx_pair[1]], (idx_pair[0]+idx_temp, idx_pair[1]+idx_temp)))
        
    return extracted_concepts


def resolve_overlapping(l):
    '''
    l: sorted list
    '''
    if len(l)==0:
        return l
    
    new_l = []
    cur_item = l[0]
    for i in range(1, len(l)):
        new_item = l[i]
        if cur_item[1][0] == new_item[1][0] and cur_item[1][1] < new_item[1][1]:
            cur_item = new_item
        else:
            if cur_item[1][1] < new_item[1][1]:
                new_l.append(cur_item)
                cur_item = new_item
    
    new_l.append(cur_item)
    
    return new_l


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
      
    # list of common phrases
    common_phrases = ["classrooms", "classroom", "faculty", "student", "students", "lab", "labs", "topic", "topics", "semester", "semesters", "instructor", "instructors", "lecture", "lectures"]
    
    # merging the 3 lists
    lower_case = articles + conjunctions + prepositions + common_phrases
            
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
