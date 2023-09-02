import re
from collections import Counter
from sudachipy import Dictionary

# SudachiPyの辞書を初期化
dictionary = Dictionary().create()

def count_pos_frequency(text, selected_pos):
    # ストップワードの定義（オプション）
    stop_words = set(["これ", "それ", "あれ", "この", "その", "あの", "ここ", "そこ", "あそこ", "こちら", "どこ", "だれ", "なに", "なん", "何", "私", "こと", "もの"])

    text = text.translate(str.maketrans({chr(0xFF01 + i): chr(0x21 + i) for i in range(len(text))}))
    text = text.replace("キャッシュ・フロー","キャッシュフロー")

    nodes = []
    tokens = dictionary.tokenize(text)

    for token in tokens:
        pos = token.part_of_speech()[0]
        if pos == selected_pos:
            word = token.surface()

            # 単語が1文字であるか、数字である場合、またはストップワードに含まれる場合はスキップ
            if len(word) == 1 or re.match(r'^\d+$', word) or word in stop_words:
                continue

            nodes.append(word)

    # 選択された品詞の出現頻度をカウント
    pos_freq = Counter(nodes)

    return pos_freq
