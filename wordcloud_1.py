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
    stop_words = set(["%)","これ", "それ", "あれ", "この", "その", "あの", "ここ", "そこ", "あそこ", "こちら", "どこ", "だれ", "なに", "なん", "何", "私", "こと", "もの"])
    stop_words.update(additional_stop_words)

    # 強制的にカウントしたい単語リスト（オプション）
    forced_words = []

    # Janomeで名詞を抽出
    tokenizer = Tokenizer(udic='user_dic.csv', udic_enc='utf8')
    
    tokens = tokenizer.tokenize(text)
    nouns = []
    
    for token in tokens:
        pos = token.part_of_speech.split(",")[0]
        if pos == "名詞":
            word = token.surface
            if word not in stop_words and len(word) > 1 and not re.search(r'\d', word):
                nouns.append(word)
                
    # 名詞をスペースで連結
    word_nouns = " ".join(nouns)

    fpath = "MEIRYO.TTC"

    # WordCloud オブジェクトを生成
    wordcloud = WordCloud(font_path=fpath, width=900, height=500, 
                        background_color="white", collocations=False, 
                        max_words=500, max_font_size=150).generate(word_nouns)
    return wordcloud
#%%

