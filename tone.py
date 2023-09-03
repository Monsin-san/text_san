#%%
import pandas as pd
from janome.tokenizer import Tokenizer
from collections import Counter

def tone_score(text):
    with open("mlwordlist.csv", 'r', encoding="utf-8") as csv_file:
        pdlist = pd.read_csv(csv_file)
    plist = pdlist['positive'].values
    nlist = pdlist['negative'].values

    tokenizer = Tokenizer()
    
    # 強制的にカウントしたい単語リスト
    forced_words = []

    tokens = tokenizer.tokenize(text)
    s = [token.surface for token in tokens if '名詞' in token.part_of_speech.split(',')[0] and '数' not in token.part_of_speech and '記号' not in token.part_of_speech and '助詞' not in token.part_of_speech]
    
    # forced_wordsに含まれる単語を強制的にリストsに追加
    s += [word for word in forced_words if word in text]

    pword = []
    nword = []
    for word in s:
        if word in plist:
            pword.append(word)
        if word in nlist:
            nword.append(word)

    # Get top 5 most frequent positive and negative words
    top_pwords = Counter(pword).most_common(5)
    top_nwords = Counter(nword).most_common(5)

    nofpword = len(pword)
    nofnword = len(nword)
    
    try:
        tone = (nofpword - nofnword) / (nofnword + nofpword)
    except ZeroDivisionError:
        tone = 0
    
    tone = round(tone, 5)
    return tone, nofpword, nofnword, top_pwords, top_nwords

def tone_eval(tone):
    if tone <-0.8:
        return "超ネガティブ"
    elif tone <-0.4:
        return "結構ネガティブ"
    elif tone < 0:
        return "ややネガティブ"
    elif tone == 0:
        return "ニュートラル"
    elif tone < 0.4:
        return "ややポジティブ"
    elif tone < 0.8:
        return "結構ポジティブ"
    else:
        return "超ポジティブ"
