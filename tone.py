#%%
import pandas as pd
from sudachipy import Dictionary

# %%
def tone_score(text):
    with open("mlwordlist.csv", 'r', encoding="utf-8") as csv_file:
        pdlist = pd.read_csv(csv_file)
    plist = pdlist['positive'].values
    nlist = pdlist['negative'].values

    tokenizer = Dictionary().create()
    
    tokens = tokenizer.tokenize(text)
    s = [token.surface() for token in tokens if '名詞' in token.part_of_speech() and '数' not in token.part_of_speech() and '記号' not in token.part_of_speech() and '助詞' not in token.part_of_speech()]

    pword = []
    nword = []
    for word in s:
        if word in plist:
            pword.append(word)
        if word in nlist:
            nword.append(word)
    
    nofpword = len(pword)
    nofnword = len(nword)
    
    try:
        tone = (nofpword - nofnword) / (nofnword + nofpword)
    except ZeroDivisionError:
        tone = 0
    
    tone = round(tone, 3)
    return tone, nofpword, nofnword

def tone_eval(tone):
    if tone <-0.8:
        return "超ネガティブ"
    elif tone <-0.4:
        return "結構ネガティブ"
    elif tone < 0:
        return "ややネガティブ"
    elif tone < 0.4:
        return "ややポジティブ"
    elif tone < 0.8:
        return "結構ポジティブ"
    else:
        return "超ポジティブ"
    
#%%

