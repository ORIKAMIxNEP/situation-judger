# procon33_RemoteTravelers_python

## 実行環境

- [Ubuntu 20.04.5](https://jp.ubuntu.com/)
- [Python 3.10.5](https://www.python.org/)

## 使用ユーティリティ

- [Flask 2.1.2](https://flask.palletsprojects.com/en/2.2.x/)：API の構築
- [ClipCap](https://github.com/rmokady/CLIP_prefix_caption)：画像キャプショニング
  - [Vision Transformer](https://github.com/google-research/vision_transformer)：画像認識
  - [Conceptual Captions](https://ai.google.com/research/ConceptualCaptions)：画像キャプショニングデータセット
- [NLTK 3.7](https://www.nltk.org/)：形態素解析(英語)
- [Gensim 4.2.0](https://radimrehurek.com/gensim/)
  - [Word2Vec](https://radimrehurek.com/gensim/models/word2vec.html)：単語ベクトルによる類似度測定
  - [LexVec](https://github.com/alexandres/lexvec)：単語埋め込みモデル
