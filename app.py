#%%
'''
streamlit run D:\GoogleDrive\python\python_code\streamlit_app_2\app.py
'''
import pandas as pd
import streamlit as st
from PIL import Image
from janome.tokenizer import Tokenizer
import base64
import re
from count_pos_frequency_1 import count_pos_frequency
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from networkx_1 import make_network
from tone import tone_score,tone_eval
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from janome.tokenizer import Tokenizer
from janome.analyzer import Analyzer
from janome.charfilter import *
from janome.tokenfilter import *
import os
from wordcloud_1 import make_wordcloud

st.title("ネコでも使える！テキスト分析（β版）") # タイトル
st.write("少しずつ機能を追加していきたいと思います。")

#image = Image.open("title.png")
image = Image.open("title.png")
st.image(image,use_column_width=True)

st.write("青山学院大学矢澤研究室では「会計・財務研究におけるテキスト分析」に取り組んでいます。研究活動の一環として、テキスト分析の魅力を体感できるウェブサイトを作成しました。肩の力を抜いてお楽しみください！")
st.write("最終更新日：2023年10月1日")
st.write("本サイトの特徴")
st.write("・ネコでもわかるくらい簡単です")
st.write("・オリジナルのユーザー辞書（2,487単語）およびネガポジ判定にはLoughran-McDonald Dictionaryを使用しています（金先生、伊藤先生に感謝）")
st.write("・こんな機能があったらいいな、→連絡ください。yazawa(at)aoyama.ac.jp")

# %%
st.title("はじめに")
st.write("テキストマイニングとは！？")

st.write("テキストマイニングとは、簡単に言えば、大量のテキストデータの中に埋もれている「意味のある情報」を自然言語処理（Natural Language Processing）という技術を用いて取り出すことです。本当はプログラミング言語でコードを書いたりなどちょっと面倒な作業があるのですが、ここではそんな過程をすっとばして、すぱっとさっくりテキスト分析を楽しんでみましょう。")

st.sidebar.title("準備")
st.sidebar.write("下記のボックスに文章を入力してみましょう！サンプルデータを用意していますのでコピー＆ペーストしてください。")

# ユーザーに文章を入力してもらう
user_input_text = st.sidebar.text_area("文章を入力してください:")

# テキストファイルを読み込む
with open("sample_A.txt", "r", encoding="utf-8") as f:
    text_content = f.read()

# テキストファイルをダウンロードするための関数
def get_text_download_link(text, filename):
    """Generates a link allowing the text to be downloaded"""
    b64 = base64.b64encode(text.encode()).decode()  # 文字をbase64エンコードする
    return f'<a href="data:file/txt;base64,{b64}" download="{filename}">Download text file</a>'

# テキストファイルのダウンロードリンクを作成
st.sidebar.write("サンプルテキスト")
st.sidebar.markdown(f"【トヨタ自動車　2023年3月期　決算短信】　 {get_text_download_link(text_content, 'sample_A.txt')} ", unsafe_allow_html=True)

st.title("ステップ１　文字数、単語数、文章数")
st.write("まずは入力した文章の文字数、単語数、文章数が出力されますので確認してください。文章は長いですか、それとも短いですか？文字数と単語数の割合などどうなっていますか？")

# JanomeのTokenizerを初期化
#tokenizer = Tokenizer()
tokenizer = Tokenizer(udic='user_dic.csv', udic_enc='utf8')

# テキストが入力された場合の処理
if user_input_text:
    user_input_text = user_input_text.translate(str.maketrans({chr(0xFF01 + i): chr(0x21 + i) for i in range(94)}))
    user_input_text = user_input_text.replace("キャッシュ・フロー", "キャッシュフロー")

    # 強制的にカウントしたい単語リスト
    forced_words = []

    # 文字数をカウント
    char_count = len(user_input_text)
    st.write(f'文字数： {char_count} 字')

    # 単語数をカウント
    tokens = []
    for token in tokenizer.tokenize(user_input_text):
        word = token.surface if token.part_of_speech.split(',')[0] != '名詞' else (token.base_form if token.base_form in forced_words else token.surface)
        tokens.append(word)
    word_count = len(tokens)
    st.write(f'単語数： {word_count} 語')

    # 文章数をカウント
    sentence_count = len(re.split('[。.!?]', user_input_text)) - 1
    st.write(f'文章数： {sentence_count} 文')

# Streamlitのインターフェイス
st.title("ステップ２　単語の出現頻度")
st.write("続いて単語の出現頻度を分析してみましょう。下のボックスからカウントしたい品詞を選択してください。どのような単語が多く使われているでしょうか？")

selected_pos = st.selectbox("カウントする品詞を選んでください:", ("名詞", "動詞", "形容詞"), key='my_unique_selectbox_key')

# グラフのフォントを設定
fontprop = FontProperties(fname="MEIRYO.TTC")  # フォントのパスを適宜変更

if user_input_text:
    pos_freq = count_pos_frequency(user_input_text, selected_pos)
    st.write(f'{selected_pos}の出現頻度')
    st.write(pos_freq)

    # グラフの描画
    if pos_freq:
        fig, ax = plt.subplots()

        # 頻度でソート
        sorted_pos_freq = pos_freq.most_common(10)
        words = [item[0] for item in sorted_pos_freq]
        counts = [item[1] for item in sorted_pos_freq]

        # y軸を逆順にして描画
        ax.barh(words[::-1], counts[::-1])

        ax.set_xlabel('出現回数', fontproperties=fontprop)

        # タイトルにフォントを設定
        ax.set_title(f'{selected_pos}の出現頻度', fontproperties=fontprop)

        # グラフにフォントを設定して描画
        ax.set_yticklabels(words[::-1], fontproperties=fontprop)  # y軸のラベルに日本語フォントを設定
        plt.tight_layout()

        # Streamlitで表示
        st.pyplot(fig)

user_input = user_input_text

st.title("ステップ３　ワードクラウド")
#st.write("タダイマ開発中デス　m(_ _)m。")
#ワードクラウド
st.write("ワードクラウドは単語の出現頻度をイラストにしたもので、どのような単語が多く使われているかを視覚的にわかりやすく表現できます。")
st.write("下の図は、名詞のみ抽出してワードクラウドを作成しています。なお、場合によっては「連結会計年度」など意味のない単語が大きく描画されることがあります。これらの単語を削除したい場合は下記のストップワードを設定してください。")

additional_stop_words = st.text_area("追加するストップワードを入力してください（スペースで区切って複数入力可能）").split()

if user_input:
    wordcloud = make_wordcloud(user_input, additional_stop_words)
    fig, ax = plt.subplots(figsize=(15, 12))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    st.pyplot(fig)

#共起ネットワーク
st.title("ステップ４　共起ネットワーク")
st.write("共起ネットワークは、単語同士のつながりをイラストにしたもので、どのような単語の組み合わせがみられるかを視覚的にわかりやすく表現できます。")
st.write("綺麗に図式化するためにはパラメータの細かな調整が必要なのですが、ここでは共起の閾値（＝共起としてカウントする最低値）のみ変更可能にしています。閾値を変更するとどのように変わるのか試してみてください。")

# 1から10までのスライダーを作成。初期値は5。
slider_value = st.slider('共起の閾値:', min_value=1, max_value=10, value=2)

if user_input:
    network = make_network(user_input,slider_value)
    st.pyplot(network)

st.write("続いて、文章がどの程度わかりやすいか（＝可読性）、そしてどのようなニュアンス（＝トーン）で書かれているかを判断する指標が算出されます。")
st.title("ステップ５　可読性")
st.write("タダイマ開発中デス　m(_ _)m。")

#%%
# トーンスコア
st.title("ステップ６　トーン")
st.write("トーンは－1（超ネガティブ）から1（超ポジティブ）で計算されます。0は中立（ニュートラル）となります。 文章のトーンはどのくらいポジティブ（ネガティブ）でしょうか？")

if user_input:
    score, nofpword, nofnword, top_pwords, top_nwords = tone_score(user_input)
    evaluation = tone_eval(score)
    st.write(f"トーンスコア： {score}")
    st.write(f"トーンレベル： {evaluation}")

    # Convert to DataFrame and display as table
    df_pwords = pd.DataFrame(top_pwords, columns=["Positive Word", "Frequency"])
    df_nwords = pd.DataFrame(top_nwords, columns=["Negative Word", "Frequency"])

    st.write("最も頻繁なポジティブワード:")
    st.table(df_pwords)

    st.write("最も頻繁なネガティブワード:")
    st.table(df_nwords)
    
# タイトルを設定
st.title('ステップ７　文章の類似度')
st.write("下記のボックスに2つの文章を入力すると、文章Aと文章Bが似ているかどうかを計算することができます。類似度は0（まったく似ていない）から1（完全に同じ）で計算されます。 ")

st.write("それでは文章Aには先ほどのテキストデータ（トヨタ自動車2023年3月期）、文章Bには同社の前決算期（2022年3月期）のデータを下記のダウンロードリンクからコピー＆ペーストしてください。 ")

# ユーザーに文章Aを入力してもらう
text_A = st.text_area("文章Aを入力してください:")

# ユーザーに文章Bを入力してもらう
text_B = st.text_area("文章Bを入力してください:")

# テキストファイルを読み込む
with open("sample_B.txt", "r", encoding="utf-8") as B:
    text_content_b = B.read()

# テキストファイルをダウンロードするための関数
def get_text_download_link_b(text, filename):
    """Generates a link allowing the text to be downloaded"""
    b64 = base64.b64encode(text.encode()).decode()  # 文字をbase64エンコードする
    return f'<a href="data:file/txt;base64,{b64}" download="{filename}">Download text file</a>'

# テキストファイルのダウンロードリンクを作成
st.write("サンプルテキスト")
st.markdown(f"【トヨタ自動車　2022年3月期　決算短信】　 {get_text_download_link(text_content_b, 'sample_B.txt')} ", unsafe_allow_html=True)

# Function to evaluate readability score
def evaluate_sim(cos_sim):
    if cos_sim < 0.2:
        return "全然似てない"
    elif cos_sim < 0.4:
        return "あんまり似てない"
    elif cos_sim < 0.6:
        return "ちょっと似てる？"
    elif cos_sim < 0.8:
        return "まあまあ似ている"
    else:
        return "超そっくり"
    
# 両方のテキストが入力された場合、類似度を計算
if text_A and text_B:
    # TF-IDFベクトルを計算
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text_A, text_B])

    # コサイン類似度を計算
    cos_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    cos_sim = round(cos_sim,3)
    
    # 類似度を表示
    st.write(f'類似度: {cos_sim}')
    evaluation = evaluate_sim(cos_sim)
    st.write(f'そっくりレベル: {evaluation}')

st.write("２つの文章はどのくらい似ていますか？なぜ似ている（似ていない）のか自分でも考えてみましょう。")

st.title('おわりに')
st.write("テキスト分析はいかがでしたでしょうか？次はぜひ自分の興味のある文章を入れて結果を確かめてみましょう！")

st.write("【サイト運営者】")
st.write("青山学院大学　経営学部　矢澤憲一研究室")

st.write("【免責事項】")
st.write("このウェブサイトおよびそのコンテンツは、一般的な情報提供を目的としています。このウェブサイトの情報を使用または適用することによって生じるいかなる利益、損失、損害について、当ウェブサイトおよびその運営者は一切の責任を負いません。情報の正確性、完全性、時宜性、適切性についても、一切保証するものではありません。")
