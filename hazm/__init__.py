# ruff: noqa
"""entry point for the package."""

from typing import List

from hazm.chunker import Chunker, RuleBasedChunker, tree2brackets
from hazm.lemmatizer import Conjugation, Lemmatizer
from hazm.normalizer import Normalizer
from hazm.pos_tagger import POSTagger
from hazm.sentence_tokenizer import SentenceTokenizer
from hazm.sequence_tagger import IOBTagger, SequenceTagger
from hazm.stemmer import Stemmer
from hazm.utils import (NUMBERS, abbreviations, default_verbs, default_words,
                        informal_verbs, informal_words, maketrans,
                        regex_replace, stopwords_list, words_list)
from hazm.word_tokenizer import WordTokenizer


def sent_tokenize(text: str) -> List[str]:
    """Sentence Tokenizer."""
    if not hasattr(sent_tokenize, "tokenizer"):
        sent_tokenize.tokenizer = SentenceTokenizer()
    return sent_tokenize.tokenizer.tokenize(text)


def word_tokenize(sentence: str) -> List[str]:
    """Word Tokenizer."""
    if not hasattr(word_tokenize, "tokenizer"):
        word_tokenize.tokenizer = WordTokenizer()
    return word_tokenize.tokenizer.tokenize(sentence)


from hazm.corpus_readers import (ArmanReader, BijankhanReader, DadeganReader,
                                 DegarbayanReader, FaSpellReader,
                                 HamshahriReader, MirasTextReader, MizanReader,
                                 NaabReader, NerReader, PersianPlainTextReader,
                                 PersicaReader, PeykareReader, PnSummaryReader,
                                 QuranReader, SentiPersReader, TNewsReader,
                                 TreebankReader, UniversalDadeganReader,
                                 VerbValencyReader, WikipediaReader)
from hazm.dependency_parser import DependencyParser, MaltParser, TurboParser
from hazm.informal_normalizer import InformalLemmatizer, InformalNormalizer
from hazm.token_splitter import TokenSplitter

# from hazm.embedding import SentEmbedding
# from hazm.embedding import WordEmbedding
