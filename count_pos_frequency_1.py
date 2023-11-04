#%%
from collections import Counter
import re
from janome.tokenizer import Tokenizer
import os

tokenizer = Tokenizer(udic='user_dic.csv', udic_enc='utf8')

def count_pos_frequency(text, selected_pos):
    # ストップワードの定義（オプション）
    stop_words = set(["%)","これ", "それ", "あれ", "この", "その", "あの", "ここ", "そこ", "あそこ", "こちら", "どこ", "だれ", "なに", "なん", "何", "私", "こと", "もの"])

    # 強制的にカウントしたい単語リスト
    forced_words = []
    
    nodes = []
    tokens = tokenizer.tokenize(text)

    for token in tokens:
        pos = token.part_of_speech.split(',')[0]
        if pos == selected_pos:
            word = token.surface if selected_pos != '名詞' else (token.base_form if token.base_form in forced_words else token.surface)
            # 単語が1文字であるか、数字である場合、またはストップワードに含まれる場合はスキップ
            if len(word) == 1 or re.match(r'^\d+$', word) or word in stop_words:
                continue
            nodes.append(word)

    # 選択された品詞の出現頻度をカウント
    pos_freq = Counter(nodes)
    
    # 上位5単語を取得
    top_words = pos_freq.most_common()
    
    return top_words

#%%
