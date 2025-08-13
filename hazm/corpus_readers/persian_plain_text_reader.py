"""این ماژول، پیکره‌های متنی خام را می‌خواند."""

from typing import Callable, List

from nltk.corpus import PlaintextCorpusReader
from nltk.corpus.reader import StreamBackedCorpusView, read_blankline_block

from hazm import SentenceTokenizer, WordTokenizer


class PersianPlainTextReader(PlaintextCorpusReader):
    CorpusView = StreamBackedCorpusView

    def __init__(
        self: "PersianPlainTextReader",
        root: str,
        fileids: List,
        word_tokenizer: Callable = WordTokenizer.tokenize,
        sent_tokenizer: Callable = SentenceTokenizer.tokenize,
        para_block_reader: Callable = read_blankline_block,
        encoding: str = "utf8",
    ) -> None:
        super().__init__(
            root,
            fileids,
            word_tokenizer,
            sent_tokenizer,
            para_block_reader,
            encoding,
        )
