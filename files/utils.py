import pandas as pd
import json
import pickle
import re
import ast
from os import listdir, walk
from os.path import isfile, join
from sklearn.datasets import load_svmlight_file
from nltk import sent_tokenize
from scipy import stats
from scipy.stats import pearsonr

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


def get_filenames_from_root(path, file_format, filtered_with='', exclude_list=[]):
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
            if name[-len(file_format):] == file_format and name.find(filtered_with) > -1 and find_str(name, exclude_list)==False:
                filenames.append(join(path, name))

    return sorted(filenames)


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
