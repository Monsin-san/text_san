#%%
import pandas as pd
from janome.tokenizer import Tokenizer
from collections import Counter

def tone_score(text):
    with open("mlwordlist_2.csv", 'r', encoding="utf-8") as csv_file:
        pdlist = pd.read_csv(csv_file)
    plist = pdlist['positive'].values
    nlist = pdlist['negative'].values

    tokenizer = Tokenizer(udic='user_dic.csv', udic_enc='utf8')
    
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
    top_pwords = Counter(pword).most_common()
    top_nwords = Counter(nword).most_common()

    # Ensure that there are always 5 words (or <NA> placeholders)
    while len(top_pwords) < 5:
        top_pwords.append(("<NA>", 0))
    while len(top_nwords) < 5:
        top_nwords.append(("<NA>", 0))

    nofpword = len(pword)
    nofnword = len(nword)

    # 両方の単語数が0の場合、特別な処理を行う
    if nofpword == 0 and nofnword == 0:
        return "データなし", 0, 0, [("<NA>", 0)], [("<NA>", 0)]

    # トーンスコアの計算
    try:
        tone = (nofpword - nofnword) / (nofnword + nofpword)
    except ZeroDivisionError:
        tone = 0
    
    tone = round(tone, 3)
    return tone, nofpword, nofnword, top_pwords, top_nwords

def tone_eval(tone):
    if tone <-0.8:
        return "超ネガティブ"
    elif tone <-0.4:
        return "かなりネガティブ"
    elif tone < 0:
        return "ややネガティブ"
    elif tone == 0:
        return "ニュートラル"
    elif tone < 0.4:
        return "ややポジティブ"
    elif tone < 0.8:
        return "かなりポジティブ"
    else:
        return "超ポジティブ"

#%%

