import pytest
from ntap.bagofwords import DocTerm

"""
FIXTURE_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    'test_files',
    )

LABEL_DATASETS = {'imdb': os.path.join(FIXTURE_DIR, 
                                       'imdb_sentiment_train.zip'),
                  'reviews': os.path.join(FIXTURE_DIR, 
                                          'review_polarity.tar.gz')}

@pytest.mark.datafiles(LABEL_DATASETS['imdb'])
def test_load_imdb(datafiles):
    for fn in datafiles.listdir():
        data = load_imdb(fn)
        assert len(data['pos']) == 12500
        assert len(data['neg']) == 12500
"""

@pytest.fixture
def text8_docterm():
    from gensim import downloader
    text8 = downloader.load('text8')
    text8 = [' '.join(words) for words in text8]
    text8 = text8[:10]
    dt = DocTerm(text8)
    return dt

def test_docterm_attributes(text8_docterm):

    assert isinstance(text8_docterm, DocTerm)
    assert text8_docterm.N == 10
    assert text8_docterm.K == 357
    assert text8_docterm.__str__() == ("DocTerm Object (documents: 10, terms: 357)\n"
                            "Doc Lengths (165-193): mean 177.00, median 178.50\n"
                            "Top Terms: academy oil aircraft appears illinois metal "
                            "bc territory characters roman property ground data isbn "
                            "method local ethical france agricultural republican")

def test_docterm_exceptions():
    with pytest.raises(AssertionError):
        text8_docterm = DocTerm([0,1,2])
    with pytest.raises(AssertionError):
        text8_docterm = DocTerm(["some text"], tokenizer='some_tokenizer')

    """
    #lda_m = LDA(df.body, method='mallet', mallet_path="~/mallet-2.0.8/bin/mallet")
    ddr = DDR(df.body, 
              dic='/Users/brendan/PycharmProjects/Facebook/data/dictionaries/mfd2.dic', 
              embedding_name="glove-twitter-50")
    """

    # TODO(brendan): diagnostic functions (via gensim, others) for high-level eval and tuning of ddr centers; handle NaNs in DocTerm

