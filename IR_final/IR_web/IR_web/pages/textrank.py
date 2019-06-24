import jieba
import jieba.analyse
import re
import numpy as np
import math
import networkx
import matplotlib.pyplot as plt

class TextRank_S():
    def __init__( self):
        self.sentence_list = []
        self.word_list = []
        self.stopwords = []
        self.delimiter = ['?',  '!',  '；',  '？',  '！',  '。',  '；',  '……',  '…']
        #self.punctuation = ['?',  '!',  ';',  '？',  '！',  '。',  '；',  '……',  '…',  '\n', '，']
        
        with open('stopword.txt','r') as stopword:
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
'''
if __name__ == '__main__':
    model = TextRank_S()
    with open('test.txt','r') as f:
        corpus = f.readlines()
        #print(str(corpus))
    model.Segmentation(corpus)
    model.Analyze()
    #networkx.draw_networkx(model.networkx_graph)
    #plt.show()
   
    for index in range( len(model.sentence_list)):
        print("\n")
        print(index)
        print( model.sentence_list[index])
        print(model.word_list[index])
'''
