import os
import pickle

def gen_vocab_dict(path):
    vocab_dict = {}
    with open(path, 'r') as f:
        next(f) # skip encoding line
        for i, line in enumerate(f, 1):
            vocab_dict[i] = line.strip()
    return vocab_dict

def gen_file_id_dict(path):
    file_dict = {}
    with open(path, 'r') as f:
        for i, line in enumerate(f):
            file_dict[i] = line.strip()
    return file_dict

'''
inverted_file_dict{}:
    word_id: {
        'unigram': {
            file_id: cnt,
            ...
            ...
        }
        'bigram': {
            second_word_1: {
                file_id: cnt,
                ...
                ...
            }
            second_word_2: {
            
            }
        }
    }
'''
def gen_inverted_file_dict(path, out_path):
    if os.path.exists(out_path): 
        with open(out_path, 'rb') as f:
            inverted_file_dict = pickle.load(f)
    else:
        inverted_file_dict = {}
        with open(path, 'r') as f:
            for line in f:
                first, second, next_n_lines = line.split()
                first, second, next_n_lines = int(first), int(second), int(next_n_lines)
                if inverted_file_dict.get(first, None) == None:
                    inverted_file_dict[first] = {}

                if second == -1:
                    uni = {}
                    for i in range(next_n_lines):
                        line = next(f)
                        f_id, cnt = line.split()
                        uni[int(f_id)] = int(cnt)
                    inverted_file_dict[first]['unigram'] = uni
                else:
                    if inverted_file_dict[first].get('bigram', None) == None:
                        inverted_file_dict[first]['bigram'] = {}
                    bi = {}
                    for i in range(next_n_lines):
                        line = next(f)
                        f_id, cnt = line.split()
                        bi[int(f_id)] = int(cnt)
                        inverted_file_dict[first]['bigram'][second] = bi

        with open(out_path, 'wb') as f:
            pickle.dump(inverted_file_dict, f)

    return inverted_file_dict
