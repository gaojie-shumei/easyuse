from gensim.models import Word2Vec
import itertools
import numpy as np
import random
import os.path as ospath
import os
from bert import tokenization
class NLPDataUtil:
    def __init__(self,use_for_bert=False,label_setlist:list=None, pad_word="<pad>",word2vec_path="word2vecmodel/",word2vec_model:[dict]=None, **word2vec_params):
        '''
        :use_for_bert: use this to generate data can feed to bert model
        :label_setlist: the ner label or classfication(the number needed same to index)
                        ner example: ['B_N','I_N','E_N','O','S','B_S'....]
                        classfication example:[0,1,2,3,4,5,6,7,8,9]
        :pad_word: use for padding data,if the data is need padding,use this to pad
        :word2vec_path: the word2vec_model store path
        :word2vec_model: the word2vec_model
        :word2vec_params: the function of word2vec's paramers
        [X] is the one word split to many word's label except the first word after split
        [CLS],[SEP] is the label for [CLS],[SEP]
        '''
        self.use_for_bert = use_for_bert
        self.pad_word = pad_word
        self.word2vec_path = word2vec_path
        self.word2vec_params = word2vec_params
        self.word2vec_model = word2vec_model  
        if label_setlist is not None:
            if use_for_bert:
                if pad_word not in label_setlist:
                    self.label_setlist = [pad_word]+label_setlist+["[X]","[CLS]","[SEP]"]
                else:
                    self.label_setlist = label_setlist+["[X]","[CLS]","[SEP]"]
            else:
                if pad_word not in label_setlist:
                    self.label_setlist = [pad_word] + label_setlist
                else:
                    self.label_setlist = label_setlist
        else:
            self.label_setlist = None

    '''
        data_x: 通常是二维数组
        data_y: None 或者 跟data_x的维度一致 
    '''
    def next_batch(self,batch_size,data_x:list,data_y:list=None,position=0,shuffle=True,random_state=random.randint(0,1000)):
        temp_data_x = data_x[position:]
        if data_y is not None:
            temp_data_y = data_y[position:]
        if shuffle:
            random.seed(random_state)
            random.shuffle(temp_data_x)
            if data_y is not None:
                random.seed(random_state)
                random.shuffle(temp_data_y)
        data_x = data_x[0:position] + temp_data_x
        if data_y is not None:
            data_y = data_y[0:position] + temp_data_y
        if batch_size>=len(temp_data_x):
            batch_x = temp_data_x
            if data_y is not None:
                batch_y = temp_data_y
            else:
                batch_y = None
        else:
            batch_x = temp_data_x[0:batch_size]
            if data_y is not None:
                batch_y = temp_data_y[0:batch_size]
            else:
                batch_y = None
        position += batch_size
        return data_x,data_y,batch_x,batch_y,position
    '''
    batch_data :二维数组  str
    '''
    def padding(self,batch_data:list,batch_y_data:list=None):
        pad_word = self.pad_word
        max_len = 0
        actual_lengths = []
        for i in range(len(batch_data)):
            if len(batch_data[i])>max_len:
                max_len = len(batch_data[i])
            actual_lengths.append(len(batch_data[i]))
        pad_data = batch_data
        pad_y_data = batch_y_data
        for i in range(len(pad_data)):
            pad_data[i] = pad_data[i] + [pad_word]*(max_len-len(batch_data[i]))
            if batch_y_data is not None:
                pad_y_data[i] = pad_y_data[i] + [pad_word]*(max_len-len(batch_y_data[i]))
#             print(len(pad_data[i]),len(pad_y_data[i]))
        actual_lengths = np.array(actual_lengths)
        return pad_data, pad_y_data, actual_lengths
    
    def ner_bert_data_convert(self,batch_data:list,tokenizer:tokenization.FullTokenizer,
                              batch_label:list=None):
        '''
        :batch_data: a two dim list,if English,then the data is splited by space,
                     if Chinese,the list is a char list
        :batch_label: the label for the batch_data
        :tokenizer: the bert spliter for the word embedding,it maybe split a word into 
                    multi-word. such as 'unexpected' maybe split to 'un##','expect','##ed'  
        '''
        def convert_single_sample(sample:list,tokenizer:tokenization.FullTokenizer,label:list=None):
            '''
            :sample: a one dim list,one of batch_data
            :label: a one dim list, one of batch_label
            :tokenizer: same to ner_bert_data_convert paramer of tokenizer
            '''
            tokens = []
            segment_ids = []
            labels = []
            tokens.append("[CLS]")
            segment_ids.append(0)  #use to token_type_ids,tag the sentence is the first or second,first is 0,second is 1
            if label is not None:
                labels.append("[CLS]")
            for i in range(len(sample)):
                token = tokenizer.tokenize(sample[i])
                for j in range(len(token)):
                    tokens.append(token[j])
                    segment_ids.append(0)
                    if label is not None:
                        if j==0:
                            labels.append(label[i])
                        else:
                            labels.append("[X]")
            tokens.append("[SEP]")
            segment_ids.append(0)
            if label is not None:
                labels.append("[SEP]")
            input_ids = tokenizer.convert_tokens_to_ids(tokens)#set word to ids
            
            #create mask,if mask=1,it show the position has word,if mask=0,then not
            input_mask = [1] * len(input_ids)
            return tokens,input_ids,input_mask,segment_ids,labels
        batch_tokens,batch_input_ids,batch_input_mask,\
        batch_segment_ids,batch_labels = [],[],[],[],[]
        max_len = 0
        actual_lengths = []
        for i in range(len(batch_data)):
            tokens,input_ids,input_mask,segment_ids,\
            labels = convert_single_sample(batch_data[i], batch_label[i], tokenizer)
            if len(input_ids)>max_len:
                max_len = len(input_ids)
            actual_lengths.append(len(input_ids))
            batch_tokens.append(tokens)
            batch_input_ids.append(input_ids)
            batch_input_mask.append(input_mask)
            batch_segment_ids.append(segment_ids)
            if batch_label is not None:
                batch_labels.append(labels)
        for i in range(len(batch_tokens)):
            batch_tokens[i] = batch_tokens[i] + [self.pad_word]*(max_len-len(batch_tokens[i]))
            batch_input_ids[i] = batch_input_ids[i] + [0]*(max_len-len(batch_input_ids[i]))
            batch_input_mask[i] = batch_input_mask[i] + [0]*(max_len-len(batch_input_mask[i]))
            batch_segment_ids[i] = batch_segment_ids[i] + [0]*(max_len-len(batch_segment_ids[i]))
            if batch_label is not None:
                batch_labels[i] = batch_labels[i] + [self.pad_word]*(max_len-len(batch_labels[i]))
        
        actual_lengths = np.array(actual_lengths)
        batch_tokens = np.array(batch_tokens)
        batch_input_ids = np.array(batch_input_ids)
        batch_input_mask = np.array(batch_input_mask)
        batch_segment_ids = np.array(batch_segment_ids)
        if batch_label is not None:
            batch_labels_index = self.label2index(batch_labels)
        else:
            batch_labels_index = np.array([])
        return batch_tokens,batch_input_ids,batch_input_mask,\
            batch_segment_ids,batch_labels_index,actual_lengths
        
    
    def label2index(self,batch_labels):
        '''
        :batch_labels: a two dim list, the content is in self.label_setlist
        '''
        label_setlist = self.label_setlist
        batch_indexs = []
        for i in range(len(batch_labels)):
            indexs = []
            for j in range(len(batch_labels[i])):
                indexs.append(label_setlist.index(batch_labels[i][j]))
            batch_indexs.append(np.array(indexs))
        batch_indexs = np.array(batch_indexs)
        return batch_indexs
    
    def index2label(self,batch_indexs:np.ndarray,actual_lengths:np.ndarray):
        '''
        :batch_indexs: a two dim np.ndarray, the content is the label's index in the self.label_setlist
        :actual_lengths: a one dim np.ndarray, the actual_lengths of the input data
        '''
        batch_labels = []
        label_setlist = self.label_setlist
        for i in range(batch_indexs.shape[0]):
            labels = []
            for j in range(actual_lengths[i]):
                if label_setlist[batch_indexs[i,j]] not in [self.pad_word,"[X]","[CLS]","[SEP]"]:
                    labels.append(label_setlist[batch_indexs[i,j]])
            batch_labels.append(np.array(labels).tolist())
        return batch_labels
    
    '''
    pad_data:二维数组 str
    '''
    def format(self,pad_data:list,pad_y_data:list=None):
        batch_x = []
        if pad_y_data is not None:
            batch_y = []
        else:
            batch_y = None
        word2vec_model = self.word2vec_model
        if word2vec_model is None:
            try:
                word2vec_model = self.word2vec()
            except:
                raise RuntimeError("the word2vec model not find, can't convert the str to vector")
        if word2vec_model is not None:
            for i in range(len(pad_data)):
                x, y = [], []
                for j in range(len(pad_data[i])):
                    x.append(word2vec_model[pad_data[i][j]])
                    if pad_y_data is not None:
                        y.append(self.label_setlist.index(pad_y_data[i][j]))
                batch_x.append(np.array(x))
                if batch_y is not None:
                    batch_y.append(np.array(y))
            batch_x = np.array(batch_x)
            if batch_y is not None:
                batch_y = np.array(batch_y)
        return batch_x,batch_y

    '''
    sentences：可以是一个·ist，对于大语料集，建议使用BrownCorpus,Text8Corpus或·ineSentence构建。
    ·  sg： 用于设置训练算法，默认为0，对应CBOW算法；sg=1则采用skip-gram算法。
    ·  size：是指特征向量的维度，默认为100。大的size需要更多的训练数据,但是效果会更好. 推荐值为几十到几百。
    ·  window：表示当前词与预测词在一个句子中的最大距离是多少
    ·  alpha: 是学习速率
    ·  seed：用于随机数发生器。与初始化词向量有关。
    ·  min_count: 可以对字典做截断. 词频少于min_count次数的单词会被丢弃掉, 默认值为5
    ·  max_vocab_size: 设置词向量构建期间的RAM限制。如果所有独立单词个数超过这个，则就消除掉其中最不频繁的一个。每一千万个单词需要大约1GB的RAM。设置成None则没有限制。
    ·  sample: 高频词汇的随机降采样的配置阈值，默认为1e-3，范围是(0,1e-5)
    ·  workers参数控制训练的并行数。
    ·  hs: 如果为1则会采用hierarchica·softmax技巧。如果设置为0（defau·t），则negative sampling会被使用。
    ·  negative: 如果>0,则会采用negativesamp·ing，用于设置多少个noise words
    ·  cbow_mean: 如果为0，则采用上下文词向量的和，如果为1（defau·t）则采用均值。只有使用CBOW的时候才起作用。
    ·  hashfxn： hash函数来初始化权重。默认使用python的hash函数
    ·  iter： 迭代次数，默认为5
    ·  trim_rule： 用于设置词汇表的整理规则，指定那些单词要留下，哪些要被删除。可以设置为None（min_count会被使用）或者一个接受()并返回RU·E_DISCARD,uti·s.RU·E_KEEP或者uti·s.RU·E_DEFAU·T的函数。
    ·  sorted_vocab： 如果为1（defau·t），则在分配word index 的时候会先对单词基于频率降序排序。
    ·  batch_words：每一批的传递给线程的单词的数量，默认为10000
    '''
    def word2vec(self, sentences=None, size=128, alpha=0.025, window=5, min_count=5,max_vocab_size=None, sample=1e-3,
                 seed=1, workers=3, min_alpha=0.0001,sg=0, hs=0, negative=5, cbow_mean=1, iter=5):
        if sentences is not None:
            if isinstance(sentences,list)==False:
                sentences = list(sentences)
            sentences = [[self.pad_word]] + sentences
#         print(sentences)
        keys = self.word2vec_params.keys()
        if "size" in keys:
            size = self.word2vec_params["size"]
        if "alpha" in keys:
            alpha = self.word2vec_params["alpha"]
        if "window" in keys:
            window = self.word2vec_params["window"]
        if "min_count" in keys:
            min_count = self.word2vec_params["min_count"]
        if "max_vocab_size" in keys:
            max_vocab_size = self.word2vec_params["max_vocab_size"]
        if "sample" in keys:
            sample = self.word2vec_params["sample"]
        if "seed" in keys:
            seed = self.word2vec_params["seed"]
        if "workers" in keys:
            workers = self.word2vec_params["workers"]
        if "min_alpha" in keys:
            min_alpha = self.word2vec_params["min_alpha"]
        if "sg" in keys:
            sg = self.word2vec_params["sg"]
        if "hs" in keys:
            hs = self.word2vec_params["hs"]
        if "negative" in keys:
            negative = self.word2vec_params["negative"]
        if "cbow_mean" in keys:
            cbow_mean = self.word2vec_params["cbow_mean"]
        if "iter" in keys:
            iter = self.word2vec_params["iter"]
        try:
            if self.word2vec_model is not None:
                model = self.word2vec_model
            else:
                model = Word2Vec.load(self.word2vec_path)
            if sentences is not None:
                flag = 0
                sentences_set = set(list(itertools.chain.from_iterable(sentences)))
                for word in sentences_set:
                    if word not in model.wv.vocab:
                        flag = 1
                        break
                if flag==1:
                    tte = model.corpus_count + len(sentences)
                    model.build_vocab(sentences, update=True)
                    model.train(sentences,total_examples=tte,epochs=model.iter)
        except:
            if sentences is not None:
                model = Word2Vec(size=size, alpha=alpha, window=window, min_count=min_count, max_vocab_size=max_vocab_size, sample=sample,
                                 seed=seed, workers=workers, min_alpha=min_alpha,sg=sg, hs=hs, negative=negative, cbow_mean=cbow_mean, iter=iter)
                model.build_vocab(sentences)
                model.train(sentences,total_examples=model.corpus_count,epochs=model.iter)
            else:
                raise RuntimeError("sentences is None and model not exists!")
        if self.word2vec_path is not None and self.word2vec_path!="":
            if ospath.exists(self.word2vec_path)==False:
                os.mkdir(self.word2vec_path)
            model.save(self.word2vec_path)
        self.word2vec_model = model
        return model
