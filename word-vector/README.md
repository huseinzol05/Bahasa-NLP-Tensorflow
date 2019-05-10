## How-to

1. Download wiki dump from here, https://dumps.wikimedia.org/mswiki/latest/mswiki-latest-pages-articles.xml.bz2

2. Run [make-corpus.py](make-corpus.py),
```bash
python make-corpus.py mswiki-latest-pages-articles.xml.bz2 wiki-ms.txt
```

3. Run any notebook using Jupyter Notebook.

## Notes

All these word vectors already implemented in Malaya.

For word2vec, https://malaya.readthedocs.io/en/latest/Word2vec.html

For ELMO, https://malaya.readthedocs.io/en/latest/Elmo.html

For Fast-text, https://malaya.readthedocs.io/en/latest/Fasttext.html
