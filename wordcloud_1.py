#%%
from wordcloud import WordCloud
from collections import Counter
from janome.tokenizer import Tokenizer
import random
import re
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def make_wordcloud(text, additional_stop_words):
    if not text.strip():
        raise ValueError("入力されたテキストが空です。")

    stop_words = set(["%)","これ", "それ", "あれ", "この", "その", "あの", "ここ", "そこ", "あそこ", "こちら", "どこ", "だれ", "なに", "なん", "何", "私", "こと", "もの"])
    stop_words.update(additional_stop_words)

    tokenizer = Tokenizer(udic='user_dic.csv', udic_enc='utf8')
    tokens = tokenizer.tokenize(text)
    nouns = [token.surface for token in tokens if token.part_of_speech.split(",")[0] == "名詞" and token.surface not in stop_words and len(token.surface) > 1 and not re.search(r'\d', token.surface)]

    if not nouns:
        raise ValueError("テキストから名詞を抽出できませんでした。")

    word_nouns = " ".join(nouns)
    fpath = "MEIRYO.TTC"
    wordcloud = WordCloud(font_path=fpath, width=900, height=500, background_color="white", collocations=False, max_words=500, max_font_size=150).generate(word_nouns)

    return wordcloud

