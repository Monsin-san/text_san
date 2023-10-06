#%%
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
import collections

st.title("ネコでも使える！テキスト分析（β版）") # タイトル

image = Image.open("title.png")
st.image(image,use_column_width=True)

st.write("青山学院大学矢澤研究室では「会計・財務研究におけるテキスト分析」に取り組んでいます。研究活動の一環として、テキスト分析の魅力を体感できるウェブサイトを作成しました。肩の力を抜いてお楽しみください！")
st.write("最終更新日：2023年10月6日")
st.write("本サイトの特徴")
st.write("・「コピペ」でテキスト分析ができます。")
st.write("・会計・財務専門用語を登録したユーザー辞書（2,487単語）およびネガポジ判定にはLoughran-McDonald Dictionary（和訳版）を使用しています（金先生、伊藤先生に感謝）")
st.write("・こんな機能があったらいいな、→連絡ください。yazawa(at)aoyama.ac.jp")

st.title("はじめに")
st.write("テキストマイニングとは！？")

st.write("テキストマイニングとは、簡単に言えば、大量のテキストデータの中に埋もれている「意味のある情報」を自然言語処理（Natural Language Processing）という技術を用いて取り出すことです。本当はデータの収集、前処理などちょっと面倒な作業があるのですが、ここではそんな過程をすっとばして、すぱっとさっくりテキスト分析を楽しんでみましょう。")

max_length_1 = 50  # 例として100文字を最大とする
max_length_2 = 50000

# サイドバーにテキストを表示
st.sidebar.title("準備")
st.sidebar.write("下記のボックスに文章を入力してみましょう！サンプルデータを用意していますので、必要に応じてコピー＆ペーストしてください。2社比較したい場合はもう1社入力してください。")

# サイドバーにユーザーに文章を入力してもらうテキストエリアを配置
user_input_text_A1 = st.sidebar.text_area("会社名（例：トヨタ自動車）", key='user_input_text_A1')
user_input_text_A2 = st.sidebar.text_area("文章を入力してください:", key='user_input_text_A2')
if len(user_input_text_A1) >max_length_1:
    st.warning(f'入力されたテキストが{max_length_1}文字を超えています！')
    user_input_text_A1 = user_input_text_A1[:max_length_1]  # 入力を最大文字数まで切り詰める
if len(user_input_text_A2) >max_length_2:
    st.warning(f'入力されたテキストが{max_length_2}文字を超えています！')
    user_input_text_A2 = user_input_text_A2[:max_length_2]  # 入力を最大文字数まで切り詰める

# テキストファイルを読み込む
with open("sample_A.txt", "r", encoding="utf-8") as f:
    text_content_A = f.read()

def get_text_download_link(text, filename):
    """Generates a link allowing the text to be downloaded"""
    b64 = base64.b64encode(text.encode()).decode()  # 文字をbase64エンコードする
    return f'<a href="data:file/txt;base64,{b64}" download="{filename}">Download text file</a>'

# サイドバーにテキストファイルのダウンロードリンクを作成
download_link_A = get_text_download_link(text_content_A, "sample_A.txt")
st.sidebar.markdown(f"サンプルデータA（トヨタ自動車　2023年3月期　決算短信　MD&A）　 {download_link_A}", unsafe_allow_html=True)

user_input_text_B1 = st.sidebar.text_area("会社名（例：日産自動車）", key='user_input_text_B1')
user_input_text_B2 = st.sidebar.text_area("文章を入力してください:", key='user_input_text_B2')
if len(user_input_text_B1) >max_length_1:
    st.warning(f'入力されたテキストが{max_length_1}文字を超えています！')
    user_input_text_B1 = user_input_text_B1[:max_length_1]  # 入力を最大文字数まで切り詰める
if len(user_input_text_B2) >max_length_2:
    st.warning(f'入力されたテキストが{max_length_2}文字を超えています！')
    user_input_text_B2 = user_input_text_B2[:max_length_2]  # 入力を最大文字数まで切り詰める

# テキストファイルを読み込む
with open("sample_B.txt", "r", encoding="utf-8") as f:
    text_content_B = f.read()

# サイドバーにテキストファイルのダウンロードリンクを作成
download_link_B = get_text_download_link(text_content_B, "sample_B.txt")
st.sidebar.markdown(f"サンプルデータB（日産自動車　2023年3月期　決算短信　MD&A）　 {download_link_B}", unsafe_allow_html=True)

#%%
#最初の分析
st.title("ステップ１　文字数、単語数、文章数")
st.write("まずは入力した文章の文字数、単語数、文章数が出力されますので確認してください。文章は長いですか、それとも短いですか？文字数と単語数の割合などどうなっていますか？")

# JanomeのTokenizerを初期化
#tokenizer = Tokenizer()
tokenizer = Tokenizer(udic="user_dic.csv", udic_enc='utf8')

import pandas as pd
import streamlit as st

# DataFrameを初期化します。
df = pd.DataFrame(columns=['社名', '文字数', '単語数', '文章数'])

def process_text(user_input_text, company_name):
    global df
    if user_input_text:
        user_input_text = user_input_text.translate(str.maketrans({chr(0xFF01 + i): chr(0x21 + i) for i in range(94)}))
        user_input_text = user_input_text.replace("キャッシュ・フロー", "キャッシュフロー")
        
        # 文字数をカウント
        char_count = len(user_input_text)

        # 単語数をカウント
        tokens = [token.surface for token in tokenizer.tokenize(user_input_text)]
        word_count = len(tokens)

        # 文章数をカウント
        sentence_count = len(re.split('[。.!?]', user_input_text)) - 1
        
        # 新しい行を作成し、既存のDataFrameに連結
        new_row = pd.DataFrame({'社名': [company_name], '文字数': [char_count], '単語数': [word_count], '文章数': [sentence_count]})
        df = pd.concat([df, new_row], ignore_index=True)

# ユーザー入力テキストを処理します。
process_text(user_input_text_A2, user_input_text_A1)

# user_input_text_B2が空でない場合、処理を実行します。
if user_input_text_B2:
    process_text(user_input_text_B2, user_input_text_B1)

# DataFrameを転置します。
df_transposed = df.set_index('社名').T

# DataFrameをHTMLテーブルの文字列に変換し、行のインデックスを表示しないようにします。
html_table = df_transposed.to_html(index_names=False)

# HTMLテーブルをStreamlitで表示します。
st.markdown(html_table, unsafe_allow_html=True)

#%%
# Streamlitのインターフェイス
st.title("ステップ２　単語の出現頻度")
st.write("続いて単語の出現頻度を分析してみましょう。下のボックスからカウントしたい品詞を選択してください。どのような単語が多く使われているでしょうか？")

def display_pos_frequency(user_input_text, company_name, selected_pos):
    if user_input_text:
        pos_freq = count_pos_frequency(user_input_text, selected_pos)
        st.write(f"{company_name} - {selected_pos}の出現頻度")
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
            ax.set_title(f"{company_name} - {selected_pos}の出現頻度", fontproperties=fontprop)
            ax.set_yticklabels(words[::-1], fontproperties=fontprop)
            plt.tight_layout()

            # Streamlitで表示
            st.pyplot(fig)

selected_pos = st.selectbox("カウントする品詞を選んでください:", ("名詞", "動詞", "形容詞"), key='my_unique_selectbox_key')
fontprop = FontProperties(fname="MEIRYO.TTC")

if user_input_text_A2:
    display_pos_frequency(user_input_text_A2, user_input_text_A1, selected_pos)

if user_input_text_B2:
    display_pos_frequency(user_input_text_B2, user_input_text_B1, selected_pos)

#%%
st.title("ステップ３　ワードクラウド")
st.write("ワードクラウドは単語の出現頻度をイラストにしたもので、どのような単語が多く使われているかを視覚的にわかりやすく表現できます。")
st.write("下の図は、名詞のみ抽出してワードクラウドを作成しています。")

def display_wordcloud(user_input_text, company_name, additional_stop_words):
    if not user_input_text.strip():  # テキストが空または空白のみの場合
        st.write(f"{company_name}: テキストが入力されていません。")
        return
    
    wordcloud = make_wordcloud(user_input_text, additional_stop_words)
    
    if not wordcloud:  # 単語がない場合
        st.write(f"{company_name}: 単語が見つかりませんでした。")
        return
    
    st.subheader(f"{company_name}-ワードクラウド")
    fig, ax = plt.subplots(figsize=(15, 12))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    st.pyplot(fig)

# A社のストップワード
additional_stop_words_A = st.text_area("A社の追加ストップワードを入力してください（スペースで区切って複数入力可能）",key='additional_stop_words_A').split()

if user_input_text_A2:
    display_wordcloud(user_input_text_A2, user_input_text_A1, additional_stop_words_A)

# B社のストップワード
additional_stop_words_B = st.text_area("B社の追加ストップワードを入力してください（スペースで区切って複数入力可能）",key='additional_stop_words_B').split()

if user_input_text_B2:
    display_wordcloud(user_input_text_B2, user_input_text_B1, additional_stop_words_B)

#%%
st.title("ステップ４　共起ネットワーク")
st.write("共起ネットワークは、単語同士のつながりをイラストにしたもので、どのような単語の組み合わせがみられるかを視覚的にわかりやすく表現できます。")

def display_network(user_input_text, company_name, slider_value, additional_stop_words):
    if not user_input_text.strip():  # テキストが空または空白のみの場合
        st.write(f"{company_name}: テキストが入力されていません。")
        return
    
    network = make_network(user_input_text, slider_value, additional_stop_words)
    
    if not network:  # ネットワークが生成できなかった場合
        st.write(f"{company_name}: 単語の共起ネットワークが見つかりませんでした。")
        return
    
    st.subheader(f"{company_name}-共起ネットワーク")
    st.pyplot(network)

# A社のストップワード
additional_stop_words_A1 = st.text_area("A社の追加ストップワードを入力してください（スペースで区切って複数入力可能）",key='additional_stop_words_A1').split()
slider_value_A = st.slider('A社の共起の閾値:', min_value=1, max_value=10, value=2, key='slider_value_A')

if user_input_text_A2:
    display_network(user_input_text_A2, user_input_text_A1, slider_value_A, additional_stop_words_A1)

# B社のストップワード
additional_stop_words_B1 = st.text_area("B社の追加ストップワードを入力してください（スペースで区切って複数入力可能）",key='additional_stop_words_B1').split()
slider_value_B = st.slider('B社の共起の閾値:', min_value=1, max_value=10, value=2, key='slider_value_B1')

if user_input_text_B2:
    display_network(user_input_text_B2, user_input_text_B1, slider_value_B, additional_stop_words_B)

#%%
st.write("続いて、文章がどの程度わかりやすいか（＝可読性）、そしてどのようなニュアンス（＝トーン）で書かれているかを判断する指標が算出されます。")

st.title("ステップ５　可読性")
st.write("タダイマ開発中デス　m(_ _)m。")

st.title("ステップ６　トーン")
st.write("トーンは－1（超ネガティブ）から1（超ポジティブ）で計算されます。0は中立（ニュートラル）となります。 文章のトーンはどのくらいポジティブ（ネガティブ）でしょうか？")

# A社のみの入力がある場合
if user_input_text_A2 and not user_input_text_B2:
    score_A, nofpword_A, nofnword_A, top_pwords_A, top_nwords_A = tone_score(user_input_text_A2)
    evaluation_A = tone_eval(score_A)
    
    df_tone_A = pd.DataFrame({
        user_input_text_A1: [score_A, evaluation_A]
    }, index=['トーンスコア', 'トーンレベル'])
    
    st.table(df_tone_A)
    
    df_pwords_A = pd.DataFrame(top_pwords_A, columns=["Positive Word", "Frequency"])
    df_nwords_A = pd.DataFrame(top_nwords_A, columns=["Negative Word", "Frequency"])
    
    st.write(f"{user_input_text_A1} 最も頻繁なポジティブワード:")
    st.table(df_pwords_A)
    
    st.write(f"{user_input_text_A1} 最も頻繁なネガティブワード:")
    st.table(df_nwords_A)

# B社のみの入力がある場合
elif user_input_text_B2 and not user_input_text_A2:
    score_B, nofpword_B, nofnword_B, top_pwords_B, top_nwords_B = tone_score(user_input_text_B2)
    evaluation_B = tone_eval(score_B)
    
    df_tone_B = pd.DataFrame({
        user_input_text_B1: [score_B, evaluation_B]
    }, index=['トーンスコア', 'トーンレベル'])
    
    st.table(df_tone_B)
    
    df_pwords_B = pd.DataFrame(top_pwords_B, columns=["Positive Word", "Frequency"])
    df_nwords_B = pd.DataFrame(top_nwords_B, columns=["Negative Word", "Frequency"])
    
    st.write(f"{user_input_text_B1} 最も頻繁なポジティブワード:")
    st.table(df_pwords_B)
    
    st.write(f"{user_input_text_B1} 最も頻繁なネガティブワード:")
    st.table(df_nwords_B)

# A社とB社の両方の入力がある場合
elif user_input_text_A2 and user_input_text_B2:
    # 上記の両方が入力された場合のコードをここに記述
    score_A, nofpword_A, nofnword_A, top_pwords_A, top_nwords_A = tone_score(user_input_text_A2)
    evaluation_A = tone_eval(score_A)
    
    score_B, nofpword_B, nofnword_B, top_pwords_B, top_nwords_B = tone_score(user_input_text_B2)
    evaluation_B = tone_eval(score_B)
    
    # トーンスコアとトーンレベルのDataFrameを作成
    df_tone = pd.DataFrame({
        user_input_text_A1: [score_A, evaluation_A],
        user_input_text_B1: [score_B, evaluation_B]
    }, index=['トーンスコア', 'トーンレベル'])
    
    st.table(df_tone)  # トーンスコアとトーンレベルの表を表示
    
    df_pwords_A = pd.DataFrame(top_pwords_A, columns=["Positive Word", "Frequency"])
    df_nwords_A = pd.DataFrame(top_nwords_A, columns=["Negative Word", "Frequency"])
    
    df_pwords_B = pd.DataFrame(top_pwords_B, columns=["Positive Word", "Frequency"])
    df_nwords_B = pd.DataFrame(top_nwords_B, columns=["Negative Word", "Frequency"])
    
    # A社とB社のデータフレームを横に結合
    df_pwords = pd.concat([df_pwords_A.add_prefix(f"{user_input_text_A1} "), df_pwords_B.add_prefix(f"{user_input_text_B1} ")], axis=1)
    df_nwords = pd.concat([df_nwords_A.add_prefix(f"{user_input_text_A1} "), df_nwords_B.add_prefix(f"{user_input_text_B1} ")], axis=1)
    
    st.write("最も頻繁なポジティブワード:")
    st.table(df_pwords)
    
    st.write("最も頻繁なネガティブワード:")
    st.table(df_nwords)

# 入力がない場合
else:
    st.write("A社またはB社のテキストを入力してください。")

#%%
# タイトルを設定
st.title('ステップ７　文章の類似度')
st.write("２つの文章はどのくらい似ていますか？なぜ似ている（似ていない）のか自分でも考えてみましょう。")

# ユーザーに文章Aを入力してもらう
text_A = user_input_text_A2

# ユーザーに文章Bを入力してもらう
text_B = user_input_text_B2

# テキストファイルを読み込む
with open("sample_B.txt", "r", encoding="utf-8") as B:
    text_content_b = B.read()

# テキストファイルをダウンロードするための関数
def get_text_download_link_b(text, filename):
    """Generates a link allowing the text to be downloaded"""
    b64 = base64.b64encode(text.encode()).decode()  # 文字をbase64エンコードする
    return f'<a href="data:file/txt;base64,{b64}" download="{filename}">Download text file</a>'

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
    
    # 類似度とそっくりレベルをDataFrameに保存
    evaluation = evaluate_sim(cos_sim)
    data = {
        '類似度': [cos_sim],
        'そっくりレベル': [evaluation]
    }
    df = pd.DataFrame(data)
    
    # DataFrameをMarkdown形式で左寄せに表示
    st.markdown(df.to_markdown(index=False), unsafe_allow_html=True)

#%%
st.title('おわりに')
st.write("テキスト分析はいかがでしたでしょうか？次はぜひ自分の興味のある文章を入れて結果を確かめてみましょう！")

st.write("【サイト運営者】")
st.write("青山学院大学　経営学部　矢澤憲一研究室")

st.write("【免責事項】")
st.write("このウェブサイトおよびそのコンテンツは、一般的な情報提供を目的としています。このウェブサイトの情報を使用または適用することによって生じるいかなる利益、損失、損害について、当ウェブサイトおよびその運営者は一切の責任を負いません。情報の正確性、完全性、時宜性、適切性についても、一切保証するものではありません。")
