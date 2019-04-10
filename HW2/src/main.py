import argparse
import ipdb
import os
import re
import math
import numpy as np
from collections import Counter

from utils import gen_vocab_dict, gen_file_id_dict, gen_inverted_file_dict
from metrics import MAP

def getChinese(context):
    filtrate = re.compile(u'[^\u4E00-\u9FA5]')
    context = filtrate.sub(r'', context)
    return context

def splitChinese(context):
    filtrate = re.compile(u'[^\u4E00-\u9FA5]')
    context = filtrate.split(context)
    context = [i for i in context if i != '']
    return context

def get_doc_len(file_id_dict):
    doc_len = {}
    for id, file in file_id_dict.items():
        with open(os.path.join('data', file), 'r') as f:
            data = getChinese(f.read())
            doc_len[id] = len(data)
    return doc_len


def get_answer(args):
    answer = {}
    with open(args.answer_path, 'r') as f:
        next(f)
        for line in f:
            id, ans = line.split(',')
            id = id.strip()
            ans = ans.split()
            answer[id] = ans
    return answer


def parse_query(args, query_dict, section):
    with open(args.query_path, 'r') as f:
        data = f.read().split('<xml>')[1].split('</xml>')[0].strip().split('<topic>')[1:]

    queries = []
    for d in data:
        d = d.split('</topic>')[0].strip()
        queries.append(d)
    
    numbers = []
    for query in queries:
        number = query.split('<number>')[1].split('</number>')[0].strip()[-3:]
        numbers.append(number)

    texts = []
    for query in queries:
        text = query.split('<'+section+'>')[1].split('</'+section+'>')[0].strip()
        texts.append(splitChinese(text))
    
    for i, text in enumerate(texts):
        if numbers[i] not in query_dict.keys():
            query_dict[numbers[i]] = {}

        unigram, bigram = set(), set()
        for q in text:
            q = q.strip()
            unigram |= set(list(q))

            start = 0
            while start+2 <= len(q):
                bigram.add(q[start:start+2])
                start += 1

        query_dict[numbers[i]][section] = (unigram, bigram)


def calculate_score(tfidf_score, query_dict, inv_vocab_dict, file_id_dict, inverted_file_dict, doc_len, section):
    file_cnt = len(file_id_dict.keys())
    avdl = np.mean([v for k, v in doc_len.items()])
    okapi_b = 0.75
    okapi_k = 2.0
    for query_id, tmp in query_dict.items():
        unigram, bigram = tmp[section]
        if query_id not in tfidf_score.keys():
            tfidf_score[query_id] = {}
        for u in unigram:
            vocab_id = inv_vocab_dict.get(u, None)
            if vocab_id != None:
                inverted = inverted_file_dict.get(vocab_id, None)
                if inverted != None and 'unigram' in inverted.keys():
                    df = len(inverted['unigram'].items())
                    for file, cnt in inverted['unigram'].items():
                        if file not in tfidf_score[query_id].keys():
                            tfidf_score[query_id][file] = 0
                        #tfidf_score[query_id][file] += math.log(1+cnt) * (math.log(file_cnt / len(inverted['unigram'].items())))**2
                        tfidf_score[query_id][file] += (cnt * (okapi_k + 1) / (cnt + okapi_k * (1-okapi_b + okapi_b*doc_len[file]/avdl)) *\
                                                        (math.log((file_cnt-df+0.5) / (df+0.5))))
        for b1, b2 in bigram:
            b1_id = inv_vocab_dict.get(b1, None)
            b2_id = inv_vocab_dict.get(b2, None)
            if b1_id != None and b2_id != None:
                inverted = inverted_file_dict.get(b1_id, None)
                if inverted != None and 'bigram' in inverted.keys():
                    if b2_id in inverted['bigram'].keys():
                        df = len(inverted['bigram'][b2_id].items())
                        for file, cnt in inverted['bigram'][b2_id].items():
                            if file not in tfidf_score[query_id].keys():
                                tfidf_score[query_id][file] = 0
                            #tfidf_score[query_id][file] += math.log(1+cnt) * (math.log(file_cnt / len(inverted['bigram'][b2_id].items())))**2
                            tfidf_score[query_id][file] += cnt * (okapi_k + 1) / (cnt + okapi_k * (1-okapi_b + okapi_b*doc_len[file]/avdl)) *\
                                                            (math.log((file_cnt-df+0.5) / (df+0.5)))


def main(args):
    query_dict = {}
    L = ['question', 'concepts']
    for q in L:
        parse_query(args, query_dict, q)

    vocab_dict = gen_vocab_dict(args.vocab_path)
    inv_vocab_dict = {v: k for k, v in vocab_dict.items()}
    file_id_dict = gen_file_id_dict(args.file_list)
    inverted_file_dict = gen_inverted_file_dict(args.inverted_file, args.inverted_file_pkl)
    doc_len = get_doc_len(file_id_dict)
    
    tfidf_score = {}
    for q in L:
        calculate_score(tfidf_score, query_dict, inv_vocab_dict, file_id_dict, inverted_file_dict, doc_len, q)
    result = {}
    for query_id in query_dict.keys():
        res = sorted(tfidf_score[query_id].items(), key=lambda kv:kv[1], reverse=True)[:100]
        result[query_id] = [os.path.split(file_id_dict[i[0]])[1].lower() for i in res]

    with open(args.output, 'w') as f:
        f.write('query_id,retrieved_docs\n')
        for q_id, res in result.items():
            f.write(q_id + ',' + ' '.join(res) + '\n')

    if args.query_path == 'data/query-train.xml':
        answer = get_answer(args)
        score = 0
        for query_id in result.keys():
            score += MAP(result[query_id], answer[query_id])
        score /= len(result)
        print(score)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--query_path', default='data/query-train.xml')
    parser.add_argument('--vocab_path', default='data/vocab.all')
    parser.add_argument('--file_list', default='data/file-list')
    parser.add_argument('--inverted_file', default='data/inverted-file')
    parser.add_argument('--inverted_file_pkl', default='data/inverted_file_dict.pkl')
    parser.add_argument('--answer_path', default='data/ans_train.csv')
    parser.add_argument('--output', default='out.csv')
    args = parser.parse_args()
    main(args)
