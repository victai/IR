import pickle
from gensim.models import KeyedVectors

component2idx = pickle.load(open('/tmp2/b04902105/IR_final/IR_howard/component2idx.pickle', 'rb'))
code2components =  pickle.load(open('/tmp2/b04902105/IR_final/IR_howard/code2components.pickle', 'rb'))
model = KeyedVectors.load_word2vec_format('data/zh_wiki_word2vec_300.bin', binary = True)

def expansion(word):

    related_enterprise_code = []

    expanded_words = model.most_similar(word, topn = 500)

    for i in expanded_words:
        syn = i[0]
        if syn not in component2idx:
            continue
        for code, components in code2components.items():
            if components[component2idx[syn]] > 0:
                related_enterprise_code.append(code)
    return related_enterprise_code

# expansion('水泥')
