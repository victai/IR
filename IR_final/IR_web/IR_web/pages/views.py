from django.http import HttpResponse
from django.shortcuts import render

import pickle
import argparse
from gensim.models import KeyedVectors

import jieba
import jieba.analyse
import re
import numpy as np
import math
import networkx
import matplotlib.pyplot as plt
import os

cur_path = os.path.dirname(os.path.abspath(__file__))

print(cur_path)
class TextRank_S():
    def __init__( self):
        self.sentence_list = []
        self.word_list = []
        self.stopwords = []
        self.delimiter = ['?',  '!',  '；',  '？',  '！',  '。',  '；',  '……',  '…']
        #self.punctuation = ['?',  '!',  ';',  '？',  '！',  '。',  '；',  '……',  '…',  '\n', '，']
        
        with open(os.path.join(cur_path, 'stopword.txt'),'r') as stopword:
            for data in stopword.readlines():
                data = data.strip()
                self.stopwords.append(data)
        #print(stopwords)
        
    def Segmentation( self, corpus):
        # cut the article into sentence( unit(node) in the textrank graph)
        # store the word in a sentence in a word_list
        #corpus is a list of sentence
        for line in corpus:
            for delimiter in self.delimiter:
                line = line.replace(' ', '')
                line = line.replace('\u3000', '')
                line = line.replace(delimiter,'@')
            ans = line.split('@')[:-1]
            #print(ans)
            for elem in ans:
                self.sentence_list.append(elem)
        
        for sentence in self.sentence_list:
            self.word_list.append(jieba.lcut(sentence))
    
    def Analyze( self):
        #build the DAG in textrank
        self.sentences_num =  len(self.sentence_list)
        self.graph = np.zeros((self.sentences_num , self.sentences_num ) )
        for x in range( self.sentences_num):
            for y in range( x, self.sentences_num):
                similarity = self.Similarity( x, y)
                self.graph[x, y] = similarity
                self.graph[y, x] = similarity
                #print(similarity)
        self.networkx_graph = networkx.from_numpy_matrix(self.graph)
        scores = networkx.pagerank(self.networkx_graph, alpha=0.85)
        self.sorted_scores = sorted(scores.items(), key = lambda item: item[1], reverse=True)
        print(self.sorted_scores)
            
        
    def Similarity(self,x,y):
        #compute the similarity which assigned as the weight between two nodes in the textrank DAG  
        
        #print("index1 = " + str(x))
        #print("index2 = " + str(y))
        
        words   = list(set(self.word_list[x] + self.word_list[y]))        
        vector1 = [float(self.word_list[x].count(word)) for word in words]
        vector2 = [float(self.word_list[y].count(word)) for word in words]

        vector3 = [vector1[index]*vector2[index]  for index in range(len(vector1))]
        vector4 = [1 for num in vector3 if num > 0.]
        co_occur_num = sum(vector4)
        if abs(co_occur_num) <= 1e-12:
            return 0.
        
        log_sum =  math.log(float(len(self.word_list[x]))) + math.log(float(len(self.word_list[y])))
        if abs(log_sum) <= 1e-12:
            return 0
        
        return float(co_occur_num/log_sum)

    def Get_top_three( self):
        pass

with open(os.path.join(cur_path, 'data/code2yahootextall.pickle'), 'rb') as f:
    code2news = pickle.load(f)
with open('/tmp2/b04902105/IR_final/IR_howard/code2name.pickle', 'rb') as f:
    code2name = pickle.load(f)


component2idx = pickle.load(open('/tmp2/b04902105/IR_final/IR_howard/component2idx.pickle', 'rb'))
code2components =  pickle.load(open('/tmp2/b04902105/IR_final/IR_howard/code2components.pickle', 'rb'))
expansion_model = KeyedVectors.load_word2vec_format(os.path.join(cur_path, 'data/zh_wiki_word2vec_300.bin'), binary = True)
print('+'*100)

def expansion(word):

    related_enterprise_code = []

    print(word, 'most similar: ', expansion_model.most_similar(word, topn=5))
    expanded_words = expansion_model.most_similar(word, topn = 500)

    for i in expanded_words:
        syn = i[0]
        if syn not in component2idx:
            continue
        for code, components in code2components.items():
            if components[component2idx[syn]] > 0:
                related_enterprise_code.append(code)
    return related_enterprise_code


def home_view(request, *args, **kwargs):
    if request.method == 'POST':
        #query = input('query: ')
        query = request.POST.get('query', '').strip()

        try:
            related_enterprise = expansion(query) # get 3 similar words
            if len(related_enterprise) == 0:
                print('please input another query')
                return render(request, 'index.html')
        except KeyError:
            print('please input another query')
            return render(request, 'index.html')

        print('related enterprises')
        for i, enterprise_code in enumerate(related_enterprise):
            related_enterprise[i] = (code2name[enterprise_code], enterprise_code)
            print('enterprise name: {}, enterprise code: {}'.format(code2name[enterprise_code], enterprise_code))

        selected_enterprise = ''
        if request.POST.get('related_enterprise', '') != '':
            print(related_enterprise[int(request.POST['related_enterprise'])])
            selected_enterprise = related_enterprise[int(request.POST['related_enterprise'])]

        News = []
        if selected_enterprise != '':
            #for enterprise in related_enterprise:
            #print('enterprise name: {}, enterprise code: {}'.format(code2name[enterprise], enterprise))
            enterprise_news = code2news[selected_enterprise[1]]
            for title, date, context in enterprise_news:
                print('title:', title)
                print('date:', date)
                #print('context:', context)
                if context.strip() is not '':
                    model = TextRank_S()
                    model.Segmentation(context.split('\n'))
                    model.Analyze()
                    #networkx.draw_networkx(model.networkx_graph)
                    #plt.show()
                    tmp = []
                    for index in range(len(model.sentence_list)):
                        if index >= 1:
                            break
                        tmp.append(model.sentence_list[index])
                        print(index, model.sentence_list[index])
                    News.append((title, date, tmp))

                print('-'*100)
            print('+'*100)

        return render(request, 'index.html', {'query': query,
                                              'related_enterprise': related_enterprise,
                                              'selected_enterprise': selected_enterprise,
                                              'News': News})
    elif request.method == 'GET':
        return render(request, 'index.html')
