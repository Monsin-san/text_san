#%%
from janome.tokenizer import Tokenizer
import re
import os

tokenizer = Tokenizer(udic='user_dic.csv', udic_enc='utf8')
    
def identify_gosyu(text):
    tokens = tokenizer.tokenize(text)
    
    kango = []      # 漢語: 漢字のみ
    wago = []       # 和語: ひらがなのみ、または漢字とひらがな
    katakana = []   # カタカナ語: カタカナのみ
    alphabet = []   # アルファベット: アルファベットのみ
    mixed = []      # 混種語: 上記以外の混合
    symbols_numbers = []  # 記号・アラビア数字

    for token in tokens:
        surface = token.surface
        # 中黒のみの場合、記号・アラビア数字として扱う
        if all(char == '\u30fb' for char in surface):
            symbols_numbers.append(surface)
        # 漢字のみチェック
        elif all('\u4e00' <= char <= '\u9fff' or char == '\u3005' for char in surface):
            kango.append(surface)
        # ひらがなチェック（和語含む）
        elif all('\u3040' <= char <= '\u309f' or '\u4e00' <= char <= '\u9fff' for char in surface):
            wago.append(surface)
        # カタカナのみチェック（中黒も許容）
        elif all('\u30a0' <= char <= '\u30ff' or char == '\u30fb' for char in surface):
            katakana.append(surface)
        # アルファベットのみチェック
        elif all('A' <= char <= 'Z' or 'a' <= char <= 'z' for char in surface):
            alphabet.append(surface)
        # 記号・アラビア数字チェック
        elif all(not('\u4e00' <= char <= '\u9fff' or '\u3040' <= char <= '\u309f' or '\u30a0' <= char <= '\u30ff' or 'A' <= char <= 'Z' or 'a' <= char <= 'z') for char in surface):
            symbols_numbers.append(surface)
        # 混種語
        else:
            mixed.append(surface)

    return {
        '漢語': kango,
        '和語': wago,
        'カタカナ': katakana,
        'アルファベット': alphabet,
        '混種語': mixed,
        '記号・数字': symbols_numbers
    }

def calculate_gosyu_ratios(text):
    # 語種識別
    gosyu_counts = identify_gosyu(text)

    # symbols_numbersを除外
    gosyu_counts.pop('記号・数字', None)

    # 各語種の出現数を計算（symbols_numbersを除外した後）
    gosyu_lengths = {gosyu: len(words) for gosyu, words in gosyu_counts.items()}

    # symbols_numbersを除いた全出現数の計算
    total_words = sum(gosyu_lengths.values())

    # 各語種の割合を計算（symbols_numbersを除外した後）
    gosyu_ratios = {gosyu: count / total_words for gosyu, count in gosyu_lengths.items()}

    return gosyu_ratios

import re

def readability(text):
    sentences = re.split('。', text)
    sentences = [sentence for sentence in sentences if len(sentence) > 30]  # 30文字以下の文を除外
    nofsent = len(sentences)  # 文の数を再計算

    if nofsent == 0:  # 除外後に文がない場合は、可読性スコアを計算できない
        return None, None, None, None, None, None

    # トークン化と語種識別は、30文字を超える文に対してのみ行う
    allwords = []
    verb = []
    ppp = []
    for sentence in sentences:
        tokens_gen = tokenizer.tokenize(sentence + '。')  # 文末の。を付け直す
        tokens = [token for token in tokens_gen]

        allwords.extend([token.surface for token in tokens])
        verb.extend([token.surface for token in tokens if token.part_of_speech.split(',')[0] == '動詞'])
        ppp.extend([token.surface for token in tokens if token.part_of_speech.split(',')[0] == '助詞'])

    nofallwords = len(allwords)
    nofverb = len(verb)
    nofppp = len(ppp)

    # calculate_gosyu_ratios を使用して漢語率と和語率を取得
    gosyu_ratios = calculate_gosyu_ratios(text)
    kangoritu = gosyu_ratios.get('漢語', 0)
    wagoritu = gosyu_ratios.get('和語', 0)
    
    heikinbuncho = nofallwords / nofsent if nofsent > 0 else 0
    dousiritu = nofverb / nofallwords if nofallwords > 0 else 0
    jyosiritu = nofppp / nofallwords if nofallwords > 0 else 0
    
    readability = heikinbuncho * -0.056 + kangoritu * 100 * -0.126 + wagoritu * 100 * -0.042 \
        + dousiritu * 100 * -0.145 + jyosiritu * 100 * -0.044 + 12.724
    readability_score = round(readability, 3)

    return readability_score, heikinbuncho, kangoritu, wagoritu, dousiritu, jyosiritu


# %%
