from collections import Counter
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
import nltk, re, string, collections
from nltk.util import ngrams
from tqdm import tqdm
import os
import util
import operator
import pickle
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import LatentDirichletAllocation

malware_classes = ["Agent", "AutoRun", "FraudLoad", "FraudPack", "Hupigon", "Krap",
           "Lipler", "Magania", "None", "Poison", "Swizzor", "Tdss",
           "VB", "Virut", "Zbot"]

# https://towardsdatascience.com/understanding-feature-engineering-part-3-traditional-methods-for-text-data-f6f7d70acd41
# https://github.com/dipanjanS/practical-machine-learning-with-python/blob/master/bonus%20content/feature%20engineering%20text%20data/Feature%20Engineering%20Text%20Data%20-%20Traditional%20Strategies.ipynb
# https://medium.com/@lettier/how-does-lda-work-ill-explain-using-emoji-108abf40fa7d
def topic_models(tree):
    c = {}

    labels = ["label" + str(i) for i in range(15)]

    # text preprocessing
    root = tree.getroot()
    text = ET.tostring(root, encoding='utf8', method='xml')
    tokenized = text.split()
    joined = " ".join(tokenized)

    cv = CountVectorizer(min_df=0., max_df=1.)
    cv_matrix = cv.fit_transform([joined])
    cv_matrix = cv_matrix.toarray()

    # fit per document
    lda = LatentDirichletAllocation(n_components=15, max_iter=10, random_state=97)
    dt_matrix = lda.fit_transform(cv_matrix)
    features = pd.DataFrame(dt_matrix, columns=malware_classes)

    for i in range(15):
        c[labels[i]] = dt_matrix[0][i]
    return c

def get_avg_max_min(lists):
    avg = sum(lists) * 1.0 / len(lists)
    max_list = max(lists)
    min_list = min(lists)
    return avg, max_list, min_list

# https://www.kaggle.com/rtatman/tutorial-getting-n-grams
def get_word_counts(tree):
    sorted_words = pickle.load( open( "sorted_words.p", "rb" ) )
    sorted_bigrams = pickle.load( open( "sorted_bigrams.p", "rb" ) )
    sorted_trigrams = pickle.load( open( "sorted_trigrams.p", "rb" ) )

    all_keywords = sorted_words + sorted_bigrams + sorted_trigrams

    ## BIGRAM
    words = Counter()
    # get string version
    root = tree.getroot()
    text = ET.tostring(root, encoding='utf8', method='xml')
    text = re.sub("=", " ", text)
    tokenized = text.split()

    # update counters
    word_count = Counter(tokenized)
    words.update(word_count)  

    bi = ngrams(tokenized, 2)
    words.update(bi)

    tri = ngrams(tokenized, 3)
    words.update(tri)
    
    # find common
    words_keys = set(words.keys())
    all_keywords = set(all_keywords)
    intersection = words_keys & all_keywords

    # print intersection
    dict_you_want = { k: words[k] for k in intersection }

    return dict_you_want

# preprocessing to get top grams
def find_top_100(direc="train"):
    fds = []
    words = Counter()
    bigrams = Counter()
    trigrams = Counter()

    for datafile in tqdm(os.listdir(direc)):
        # parse file as an xml document
        tree = ET.parse(os.path.join(direc,datafile))

        # get word counts
        root = tree.getroot()
        text = ET.tostring(root, encoding='utf8', method='xml')
        text = re.sub("=", " ", text)
        tokenized = text.split()

        # update words
        word_count = Counter(tokenized)
        words.update(word_count)  

        # update bigrams
        bi = ngrams(tokenized, 2)
        bigrams.update(bi)
        tri = ngrams(tokenized, 3)
        trigrams.update(tri)

    sorted_words = dict(sorted(words.items(), key=operator.itemgetter(1), reverse = True)[:200]).keys()
    sorted_bigrams = dict(sorted(bigrams.items(), key=operator.itemgetter(1), reverse = True)[:200]).keys()
    sorted_trigrams = dict(sorted(trigrams.items(), key=operator.itemgetter(1), reverse = True)[:200]).keys()

    pickle.dump( sorted_words, open( "sorted_words.p", "wb" ) )
    pickle.dump( sorted_bigrams, open( "sorted_bigrams.p", "wb" ) )
    pickle.dump( sorted_trigrams, open( "sorted_trigrams.p", "wb" ) )
    return sorted_words, sorted_bigrams, sorted_trigrams

# combine all helper functions
def merge_functions(tree):
    c = {}
    grams = {}
    c.setdefault("startreason", 0)
    c.setdefault("executionstatus", 0)
    c.setdefault("terminationreason", 0)
    c.setdefault("username", 0)
    c.setdefault("srcfile", 0)

    times = []
    starts = []
    ends = []
    sizes = []

    tags = ""
    header_grams = pickle.load( open( "header_grams.p", "rb" ) )

    for el in tree.iter():
        c.setdefault(el.tag, 0)
        tags = tags + " " + str(el.tag)
        if el.tag == "process":
            user = el.get('startreason')

            s = el.get('starttime')
            e = el.get("terminationtime")

            # calculate times
            start = int(re.sub(r"[:\.]+", "", s))
            end = int(re.sub(r"[:\.]+", "", e))
            t = end - start

            # append
            times.append(t)
            starts.append(start)
            ends.append(end)

            sz = int(el.get('filesize'))
            if sz == -1:
                c.setdefault("invalid_file_sz", 0)
                c["invalid_file_sz"] += 1
            else:
                sizes.append(int(sz))

            if user != "CreateProcess" and user != "AnalysisTarget" and user != "DCOMService" and user != "SCM":
                c["startreason"]+=1

            user = el.get('executionstatus')
            if user != "OK":
                c["executionstatus"]+=1

            user = el.get('terminationreason')
            if user != "NormalTermination" and user != "Timeout":
                c["terminationreason"]+=1

            user = el.get('username')
            if user != "Administrator" and user != "SYSTEM" and user:
                c["username"]+=1

        elif el.tag == "open_file":
            user = el.get('srcfile')
            if user and user[0] != "c" and user[0] != "C" and user[0] != "\\":
                c["srcfile"]+=1

        elif el.tag[0:4] == "load":
            if el.get('filesize'):
                sz = int(el.get('filesize'))
                if sz == -1:
                    c.setdefault("invalid_file_sz", 0)
                    c["invalid_file_sz"] += 1
                else:
                    sizes.append(int(sz))
        c[el.tag]+=1

    # grams only
    bi_dict = dict(Counter(ngrams(tags.split(" "), 2)))
    grams.update(bi_dict)
    tri_dict = dict(Counter(ngrams(tags.split(" "), 3)))
    grams.update(tri_dict)

    grams_keys = set(grams.keys())
    header_keywords = set(header_grams)
    intersection = grams_keys & header_keywords

    dict_you_want = { k: grams[k] for k in intersection }
    c.update(dict_you_want)

    # process time and file size
    c["avg_process_time"], c["max_process_time"], c["min_process_time"] = get_avg_max_min(times)
    c["avg_startprocesstime"], c["max_startprocesstime"], c["min_startprocesstime"]= get_avg_max_min(starts)
    c["avg_endprocesstime"], c["max_endprocesstime"], c["min_endprocesstime"]= get_avg_max_min(ends)
    c["avgfilesize"], c["maxfilesize"], c["minfilesize"] = get_avg_max_min(sizes)
    return c

# preprocessing to get tags
def get_header_tags():
    words = {}
    direc = "train"

    for datafile in tqdm(os.listdir(direc)):
        # parse file as an xml document
        tree = ET.parse(os.path.join(direc,datafile))
        tags = ""
        for el in tree.iter():
            tags = tags + " " + str(el.tag)

        bi_dict = dict(Counter(ngrams(tags.split(" "), 2)))
        words.update(bi_dict)
        tri_dict = dict(Counter(ngrams(tags.split(" "), 3)))
        words.update(tri_dict)

    sorted_words = dict(sorted(words.items(), key=operator.itemgetter(1), reverse = True)[:200]).keys()
    print len(sorted_words)
    pickle.dump( sorted_words, open( "header_grams.p", "wb" ) )
    return

# inspired by https://www.kaggle.com/rtatman/tutorial-getting-n-grams
def get_bigram_data(tree):
    root = tree.getroot()
    text = ET.tostring(root, encoding='utf8', method='xml')
    text = re.sub("=", " ", text)
    tokenized = text.split()

    # and get a list of all the bi-grams
    bigrams = ngrams(tokenized, 3)
    bigrams_dict = Counter(bigrams)
    return bigrams_dict

# not helpful
def get_host_by_name(tree):
    c = {}
    for el in tree.iter():
        if el.tag == "get_host_by_name":
            user = el.get('get_host_by_name')
            c.setdefault("host", 0)
            c["host"]+=1
    return c