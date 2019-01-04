import itertools
import os
import re
from os import listdir
from os.path import isfile, join
from nltk.tokenize import sent_tokenize

import gensim
import morfeusz2

POLISH_STOP_WORDS = './polish_stopwords.txt'


def load_stop_words():
    with open(POLISH_STOP_WORDS, 'r', encoding='utf8') as f:
        stop_words = f.read().splitlines()
    return stop_words;


def make_common_txt(sources_paths, target_path):
    asterix_pattern = re.compile('(^\**$)|(^Rozdział ([a-z]* ?){1,2}$)|(^[VIX]*$)')
    book_finish_pattern = re.compile('^koniec ?([a-z]* ?){1,2}$', flags=re.IGNORECASE)
    stop_words = set(load_stop_words())

    if os.path.exists(target_path):
        os.remove(target_path)
    target_f = open(target_path, 'a', encoding="utf8")

    for path in sources_paths:
        with open(path, 'r', encoding='utf8') as f:
            # text = f.read()
            lines = []
            for line in f:
                line = line.strip()
                if len(line) == 0:
                    continue
                match = asterix_pattern.match(line)
                if match is not None:
                    continue
                sents = sent_tokenize(line)
                lines.extend(sents)

        last_line = lines[len(lines) - 1]
        match = book_finish_pattern.match(last_line)
        if match is not None:
            del lines[-1]

        lines = preprocess_sents(lines, stop_words)

        text = '\n'.join(lines)
        target_f.write(text)

    target_f.close()


def preprocess_sents(sents, stop_words):
    morf = morfeusz2.Morfeusz(generate=False)
    res = []

    for sent in sents:
        analysis = morf.analyse(sent)
        brief_list = [next(t) for _, t in itertools.groupby(analysis, lambda x: x[0])]
        words = list(filter(lambda x: x.isalpha() and x not in stop_words, map(lambda x: x[2][1].lower(), brief_list)))
        if len(words) > 0:
            res.append(' '.join(words))

    return res


def train_model(sents):
    """
        training a gensim model, see also: https://radimrehurek.com/gensim/models/word2vec.html
    """
    # min-count: only include words in the model with a min-count
    return gensim.models.Word2Vec(sents, min_count=5, size=150, workers=4, window=12, sg=1, negative=5, iter=25)


class Sentences:
    def __init__(self, path):
        self.path = path

    def __iter__(self):
        with open(self.path, 'r', encoding='utf8') as f:
            for line in f:
                yield line.split()


def test():
    word_model = gensim.models.KeyedVectors.load_word2vec_format('../models/witcher.model', binary=False)
    print('leo + bonhart:', word_model.similarity('leo', 'bonhart'))
    print('skellen + puszczyk:', word_model.similarity('skellen', 'puszczyk'))
    print('geralt + wiedźmin:', word_model.similarity('geralt', 'wiedźmin'))
    print('dijkstra + sigismund:', word_model.similarity('dijkstra', 'sigismund'))
    print('emhyr:', word_model.most_similar('emhyr', topn=6))
    print('emreis:', word_model.most_similar('emreis', topn=6))
    print('skellen:', word_model.most_similar('skellen', topn=6))
    print('geralt:', word_model.most_similar('geralt', topn=6))
    print('ciri:', word_model.most_similar('ciri', topn=6))
    print('puszczyk:', word_model.most_similar('puszczyk', topn=6))
    print('dijkstra:', word_model.most_similar('dijkstra', topn=6))


def create_preprocessed_text():
    path = './text/'
    texts_paths = [join(path, f) for f in listdir(path) if isfile(join(path, f))]
    make_common_txt(texts_paths, './text/whole_story.txt')


def create_model():
    sentences = Sentences('./text/whole_story.txt')
    model = train_model(sentences)
    model.wv.save_word2vec_format("../models/witcher.model", binary=False)  # text / vec format

def main():
    # create_preprocessed_text()

    # create_model()

    test()


if __name__ == '__main__':
    main()
