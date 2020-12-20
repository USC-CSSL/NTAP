
from .bagofwords import DocTerm, LDA
from .embedding import DDR
from .parse import TextProcessor


__all__ = ['Pipeline', 'WordCount', 'DicSims', 'Topics', 'FineTune']


class Pipeline:

    def __init__(self, steps):

        self.steps = steps


class WordCount(Pipeline):

    steps = [TextProcessor(), 

    def __init__(self):
        super().__init__(
