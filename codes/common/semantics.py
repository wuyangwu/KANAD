'''
Be aware that a log sequence inside a chunk has been padded.
'''
import re
def tokenize(log):
    word_lst_tmp = re.findall(r"[a-zA-Z]+", log)
    word_lst = []
    for word in word_lst_tmp:
        res = list(filter(None, re.split("([A-Z][a-z][^A-Z]*)", word)))
        if len(res) == 0: word_lst.append(word.lower())
        else: word_lst.extend([w.lower() for w in res])
    return word_lst

from gensim.models import Word2Vec, FastText
from gensim.models import KeyedVectors
import os
import dgl
import torch
import torch.nn as nn
import math
class Vocab:
    def __init__(self, **kwargs):
        self.embedding_dim = kwargs["word_embedding_dim"]
        self.save_dir = kwargs["word2vec_save_dir"]
        self.model_type = kwargs["word2vec_model_type"]
        self.epochs = kwargs["word2vec_epoch"]
        self.word_window = kwargs["word_window"]
        self.log_lenth = 0
        self.save_path = os.path.join(self.save_dir, self.model_type+"-"+str(self.embedding_dim)+".model")

    def get_word2vec(self, logs):
        if os.path.exists(self.save_path): #load
            if self.model_type == "naive" or self.model_type == "skip-gram": 
                model = Word2Vec.load(self.save_path)
            elif self.model_type == "fasttext": 
                model = FastText.load(self.save_path)
        else:
            ####### Load log corpus #######
            sentences = [["padding"]]
            for log in logs:
                word_lst = tokenize(log)
                if len(set(word_lst)) == 1 and word_lst[0] == "padding": continue
                self.log_lenth = max(self.log_lenth, len(word_lst))
                sentences.append(word_lst)
            
            ####### Build model #######
            if self.model_type == "naive": 
                model = Word2Vec(window=self.word_window, min_count=1, vector_size=self.embedding_dim)
            elif self.model_type == "skip-gram": 
                model = Word2Vec(sg=1, window=self.word_window, min_count=1, vector_size=self.embedding_dim)
            elif self.model_type == "fasttext": 
                model = FastText(window=self.word_window, min_count=1, vector_size=self.embedding_dim)
            model.build_vocab(sentences)
            
            ####### Train and Save#######
            model.train(sentences, total_examples=len(sentences), epochs=self.epochs)
            os.makedirs(self.save_dir, exist_ok=True)
            model.save(self.save_path)

        self.word2vec = model
        self.wv = model.wv; del model.wv

from sklearn.base import BaseEstimator   
import numpy as np
import logging    
import itertools

class FeatureExtractor(BaseEstimator):
    def __init__(self, **kwargs):
        self.feature_type = kwargs["feature_type"]
        self.data_type = kwargs["data_type"]
        self.log_window_size = kwargs["log_window_size"]
        self.model_type = kwargs["word2vec_model_type"]
        self.embedding_dim = kwargs["word_embedding_dim"]
        self.vocab = Vocab(**kwargs)
        self.meta_data = {"num_labels":2, "max_log_lenth":1}
        self.oov = set()
    

    def __log2vec(self, log): 
        word_lst = tokenize(log)
        feature = []
        for word in word_lst:
            if word in self.known_words:
                feature.append(self.word_vectors[word])
            else:
                self.oov.add(word)
                if self.model_type == "naive" or self.model_type == "skip-gram": 
                    feature.append(np.random.rand(self.word_vectors["padding"].shape)-0.5) #[-0.5, 0.5]
                else: 
                    feature.append(self.word_vectors[word])
        return np.array(feature).mean(axis=0).astype("float32") #[embedding_dim]
        
    def __seqs2feat(self, seqs): #seqs in a chunk, with the number of chunk_length
        if self.feature_type == "word2vec":
            return np.array([[self.__log2vec(log) for log in seq] for seq in seqs]).astype("float32")
        if self.feature_type == "sequential":
            return np.array([self.log2id_train.get(log, 1) for log in seqs]).astype("float32")
    

    def fit(self, chunks):
        total_logs = list(itertools.chain(*[v["logs"] for _, v in chunks.items()]))
        self.ulog_train = set(total_logs)
        self.id2log_train = {0: "oovlog"}
        self.id2log_train.update({idx: log for idx, log in enumerate(self.ulog_train, 1)})
        self.log2id_train = {v: k for k, v in self.id2log_train.items()}
        logging.info("{} tempaltes are found.".format(len(self.log2id_train)-1))
        
        if self.feature_type == "word2vec":
            self.vocab.get_word2vec(total_logs)
            self.word_vectors = self.vocab.wv
            self.known_words = self.vocab.wv.key_to_index #known word list
            self.meta_data["vocab_size"] = len(self.known_words)
            self.meta_data["max_log_lenth"] = self.vocab.log_lenth if self.vocab.log_lenth>0 else 50
        
        elif self.feature_type == "sequentials":
            self.meta_data["vocab_size"] = len(self.log2id_train)
        
        else: raise ValueError("Unrecognized feature type {}".format(self.feature_type))

    def transform(self, chunks, datatype="train"):
        logging.info("Transforming {} data.".format(datatype))
        
        if not ("train" in datatype): # handle new logs
            total_logs = list(itertools.chain(*[v["logs"] for _, v in chunks.items()]))
            ulog_new = set(total_logs) - self.ulog_train
            logging.info(f"{len(ulog_new)} new templates show.")
            for u in ulog_new: print(u)
        
        for id, item in chunks.items():
            chunks[id]["log_features"] = self.__seqs2feat(item["seqs"])
            
        if len(self.oov) > 0: 
            logging.info("{} OOV words: {}".format(len(self.oov), ",".join(list(self.oov))))
        
        return chunks
    def construct_graph(self, bytes, w_size, k=1):
        # word co-occurence with context windows
        window_size = w_size
        windows = []  # [[], [], [], ..., []]

        words = bytes  # ['A', 'B', 'C']
        length = len(words)
        if length <= window_size:
            windows.append(words)
        else:
            # print(length, length - window_size + 1)
            for j in range(length - window_size + 1):
                window = words[j: j + window_size]
                windows.append(window)

        word_window_freq = {}
        for window in windows:
            appeared = set()
            for i in range(len(window)):
                if window[i] in appeared:
                    continue
                if window[i] in word_window_freq:
                    word_window_freq[window[i]] += 1
                else:
                    word_window_freq[window[i]] = 1
                appeared.add(window[i])

        word_pair_count = {}
        for window in windows:
            for i in range(1, len(window)):
                for j in range(0, i):
                    word_i = window[i]
                    word_i_id = word_i
                    word_j = window[j]
                    word_j_id = word_j
                    if word_i_id == word_j_id:
                        continue
                    word_pair_str = str(word_i_id) + ',' + str(word_j_id)
                    if word_pair_str in word_pair_count:
                        word_pair_count[word_pair_str] += 1
                    else:
                        word_pair_count[word_pair_str] = 1
                    # two orders
                    word_pair_str = str(word_j_id) + ',' + str(word_i_id)
                    if word_pair_str in word_pair_count:
                        word_pair_count[word_pair_str] += 1
                    else:
                        word_pair_count[word_pair_str] = 1

        src = []
        dst = []
        weight = []

        # pmi as weights

        num_window = len(windows)

        for key in word_pair_count:
            temp = key.split(',')
            i = temp[0]
            j = temp[1]
            count = word_pair_count[key]
            word_freq_i = word_window_freq[i]
            word_freq_j = word_window_freq[j]
            pmi = math.log((1.0 * count / num_window) ** k /
                           (1.0 * word_freq_i * word_freq_j / (num_window * num_window)))
            if pmi <= 0:
                continue
            src.append(i)
            dst.append(j)
            weight.append(pmi)

        bytes2id = {}
        feat = []
        id_count = 0
        for byte in src:
            if byte in bytes2id:
                continue
            bytes2id[byte] = id_count
            id_count += 1
            feat.append([byte])

        src = [bytes2id[i] for i in src]
        dst = [bytes2id[i] for i in dst]

        g = dgl.graph((src, dst))
        g.ndata['feat'] = torch.tensor([self.word_vectors[feats].reshape(-1).tolist() for feats in feat],
                                       dtype=torch.float32)
        return dgl.add_self_loop(g)
    def transformGraph(self,chunks,datatype="train"):
        logging.info("Transforming {} log Graph construct data.".format(datatype))
        for id, item in chunks.items():
           if  item["logs"][0] not in 'padding':
               words = []
               for log in item["logs"]:
                   # if log not in 'padding':
                    words.extend(tokenize(log))
               gLog = self.construct_graph(words,w_size=6)
               item['glog'] = gLog
           else:
               src = []  # 没有边的起始节点
               dst = []  # 没有边的目标节点

               g = dgl.graph((src, dst))
               g.add_nodes(1)  # 这里我们显式地添加一个节点
               g.ndata['feat'] = torch.tensor([self.word_vectors['padding'].reshape(-1).tolist()],
                                              dtype=torch.float32)
               gLog = dgl.add_self_loop(g)
               item['glog'] = gLog
               # logging.info("全是padding重新构图")
        return chunks
                
    
        

