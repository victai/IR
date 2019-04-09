import argparse
import ipdb
import os
from collections import Counter

from utils import gen_vocab_dict, gen_file_id_dict, gen_inverted_file_dict
from metrics import MAP

'''
    Just a simple baseline.
    Training score: 0.72450
    Kaggle public score: 0.75279

'''


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


def parse_query(args):
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

    concepts = []
    for query in queries:
        concept = query.split('<concepts>')[1].split('</concepts>')[0].strip().strip('。').split('、')
        concepts.append(concept)

    Q = []
    for concept in concepts:
        unigram, bigram = [], []
        for q in concept:
            q = q.strip()
            if len(q) == 1:
                unigram.append(q)
            else:
                start = 0
                while start+2 <= len(q):
                    bigram.append(q[start:start+2])
                    start += 1
        Q.append((unigram, bigram))
    return list(zip(numbers, Q))
        

def main(args):
    queries = parse_query(args)
    vocab_dict = gen_vocab_dict(args.vocab_path)
    inv_vocab_dict = {v: k for k, v in vocab_dict.items()}
    file_id_dict = gen_file_id_dict(args.file_list)
    inverted_file_dict = gen_inverted_file_dict(args.inverted_file, args.inverted_file_pkl)
    
    result = {}
    for query_id, (unigram, bigram) in queries:
        res = []
        for u in unigram:
            vocab_id = inv_vocab_dict.get(vocab_id, None)
            if vocab_id != None:
                inverted = inverted_file_dict.get(vocab_id, None)
                if inverted != None and 'unigram' in inverted.keys():
                    for k, v in inverted['unigram'].items():
                        res += [k]
        for b1, b2 in bigram:
            b1_id = inv_vocab_dict.get(b1, None)
            b2_id = inv_vocab_dict.get(b2, None)
            if b1_id != None and b2_id != None:
                inverted = inverted_file_dict.get(b1_id, None)
                if inverted != None and 'bigram' in inverted.keys():
                    if b2_id in inverted['bigram'].keys():
                        for k, v in inverted['bigram'][b2_id].items():
                            res += [k]
        counter = Counter(res)
        m_c = counter.most_common(100)
        result[query_id] = [os.path.split(file_id_dict[i[0]])[1].lower() for i in m_c]

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
