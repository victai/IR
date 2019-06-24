import pickle
import argparse
import ipdb

from textrank import TextRank_S
from expansion import expansion


if __name__ == '__main__':
    #parser = argparse.ArgumentParser()
    #parser.add_argument('--query', '-q', required=True, help='what do you want to know about?, ex')
    #args = parser.parse_args()

    with open('data/code2yahootextall.pickle', 'rb') as f:
        code2news = pickle.load(f)
    with open('IR_howard/code2name.pickle', 'rb') as f:
        code2name = pickle.load(f)

    while True:
        # Step 1: Query Expansion
        query = input('query: ')
        if query.strip() == 'q':
            exit()
        try:
            related_enterprise = expansion(query) # get 3 similar words
            if len(related_enterprise) == 0:
                print('please input another query')
                continue
        except KeyError:
            print('please input another query')
            continue

        print('related enterprises')
        for enterprise in related_enterprise:
            print('enterprise name: {}, enterprise code: {}'.format(code2name[enterprise], enterprise))

        for enterprise in related_enterprise:
            print('enterprise name: {}, enterprise code: {}'.format(code2name[enterprise], enterprise))
            enterprise_news = code2news[enterprise]
            for title, date, context in enterprise_news:
                ipdb.set_trace()
                print('title:', title)
                print('date:', date)
                #print('context:', context)
                if context.strip() is not '':
                    model = TextRank_S()
                    model.Segmentation(context.split('\n'))
                    model.Analyze()
                    #networkx.draw_networkx(model.networkx_graph)
                    #plt.show()

                    for index in range(len(model.sentence_list)):
                        if index >= 2:
                            break
                        print(index, model.sentence_list[index])
                        #print(model.word_list[index])
                print('-'*100)
            print('+'*100)
