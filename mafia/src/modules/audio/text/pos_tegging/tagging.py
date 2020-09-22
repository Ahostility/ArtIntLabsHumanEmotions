from .....dirs import DIR_DATA_RAW, DIR_DATA_PROCESSED


def get_tag(text: str):
    from nltk.tag.api import TaggerI
    from nltk.tag.util import str2tuple, tuple2str, untag
    from nltk.tag.sequential import (
        SequentialBackoffTagger,
        ContextTagger,
        DefaultTagger,
        NgramTagger,
        UnigramTagger,
        BigramTagger,
        TrigramTagger,
        AffixTagger,
        RegexpTagger,
        ClassifierBasedTagger,
        ClassifierBasedPOSTagger,
    )
    from nltk.tag.brill import BrillTagger
    from nltk.tag.brill_trainer import BrillTaggerTrainer
    from nltk.tag.tnt import TnT
    from nltk.tag.hunpos import HunposTagger
    from nltk.tag.stanford import StanfordTagger, StanfordPOSTagger, StanfordNERTagger
    from nltk.tag.hmm import HiddenMarkovModelTagger, HiddenMarkovModelTrainer
    from nltk.tag.senna import SennaTagger, SennaChunkTagger, SennaNERTagger
    from nltk.tag.mapping import tagset_mapping, map_tag
    from nltk.tag.crf import CRFTagger
    from nltk.tag.perceptron import PerceptronTagger
    from nltk.tokenize import word_tokenize

    from nltk.data import load, find

    import pandas as pd
    import numpy as np

    RUS_PICKLE = (
        "taggers/averaged_perceptron_tagger_ru/averaged_perceptron_tagger_ru.pickle"
    )


    def _get_tagger(lang=None):
        if lang == "rus":
            tagger = PerceptronTagger(False)
            ap_russian_model_loc = "file:" + str(find(RUS_PICKLE))
            tagger.load(ap_russian_model_loc)
        else:
            tagger = PerceptronTagger()
        return tagger


    def _pos_tag(tokens, tagset=None, tagger=None, lang=None):

        if lang not in ["eng", "rus"]:
            raise NotImplementedError(
                "Currently, NLTK pos_tag only supports English and Russian "
                "(i.e. lang='eng' or lang='rus')"
            )
        else:
            tagged_tokens = tagger.tag(tokens)
            if tagset: 
                if lang == "eng":
                    tagged_tokens = [
                        (token, map_tag("en-ptb", tagset, tag))
                        for (token, tag) in tagged_tokens
                    ]
                elif lang == "rus":
                    tagged_tokens = [
                        (token, map_tag("ru-rnc-new", tagset, tag.partition("=")[0]))
                        for (token, tag) in tagged_tokens
                    ]
            return tagged_tokens


    def pos_tag(tokens, tagset=None, lang="rus"):
        """
        Use NLTK's currently recommended part of speech tagger to
        tag the given list of tokens..

        :param tokens: Sequence of tokens to be tagged
        :type tokens: list(str)
        :param tagset: the tagset to be used, e.g. universal, wsj, brown
        :type tagset: str
        :param lang: the ISO 639 code of the language, e.g. 'eng' for English, 'rus' for Russian
        :type lang: str
        :return: The tagged tokens
        :rtype: list(tuple(str, str))
        """
        tagger = _get_tagger(lang)
        return _pos_tag(tokens, tagset, tagger, lang)


    tags = []
    for index, i in enumerate(text):
        tags.append(pos_tag(word_tokenize(i), lang="rus"))

    data_text = []
    for index, sample in enumerate(tags):
        data_text.append('')
        for i in sample:
            data_text[index] += str(i[1] + ' ')

    return data_text


if __name__ == '__main__': get_tag()