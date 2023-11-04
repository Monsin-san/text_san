#%%
#バージョン3:2社比較
#自宅
#conda activate env_py39
#streamlit run D:\GoogleDrive\python\python_code\streamlit_app_3\app.py 
#path C:\Users\yazaw\anaconda3\envs\env_py39

#大学
#streamlit run F:\マイドライブ\python\python_code\streamlit_app_3\app.py 

import streamlit as st
import pandas as pd
from PIL import Image
from janome.tokenizer import Tokenizer
import base64
import re
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from janome.tokenizer import Tokenizer
from janome.charfilter import *
from janome.tokenfilter import *
import os
from collections import Counter
from count_pos_frequency_1 import count_pos_frequency
from wordcloud_1 import make_wordcloud
from networkx_1 import make_network_with_jaccard_enhanced
from readability_1 import readability,calculate_gosyu_ratios,identify_gosyu
from tone import tone_score,tone_eval

#os.chdir(r"D:\GoogleDrive\python\python_code\streamlit_app_3") #home
#os.chdir(r"F:\マイドライブ\python\python_code\streamlit_app_3") #office

st.title("ネコでも使える！会計テキストマイニング") # タイトル

image = Image.open("title.png") 
st.image(image,use_column_width=True)

st.write("青山学院大学矢澤研究室では「会計・財務データを用いたテキスト分析」に取り組んでいます。研究活動の一環として、テキスト分析の魅力を体感できるウェブサイトを作成しました。肩の力を抜いてお楽しみください！")
st.write("最終更新日：2023年11月4日")

# %%
st.title("はじめに")
st.write("テキストマイニングとは！？")
st.write("テキストマイニングとは、簡単に言えば、大量のテキストデータの中に埋もれている「意味のある情報」を自然言語処理（Natural Language Processing）という技術を用いて取り出すことです。本当はプログラミング言語でコードを書いたりなどちょっと面倒な作業があるのですが、ここではそんな過程をすっとばして、すぱっとさっくりテキスト分析を楽しんでみましょう。")
st.write("ステップ１と２はサイドバーに表示されています。まずは説明に従ってデータを入力してください。")

#%%
# サイドバーにテキストを表示
st.sidebar.title("ステップ１　データ収集")
st.sidebar.write("下記のボックスに文章を入力してみましょう！サンプルデータを用意していますので、必要に応じてコピー＆ペーストしてください。2社比較したい場合はもう1社入力してください。")
st.sidebar.write("会社名は最大10文字、文章は最大30,000字まで。")

max_length_1 = 30  
max_length_2 = 30000

# サイドバーにユーザーに文章を入力してもらうテキストエリアを配置
user_input_text_A1 = st.sidebar.text_area("会社名（例：ソニー）", key='user_input_text_A1')
if len(user_input_text_A1) >max_length_1:
    st.sidebar.error(f'入力されたテキストが{max_length_1}文字を超えています！')
    user_input_text_A1 = user_input_text_A1[:max_length_1]  # 入力を最大文字数まで切り詰める
user_input_text_A2 = st.sidebar.text_area("文章を入力してください:", key='user_input_text_A2')
if len(user_input_text_A2) >max_length_2:
    st.sidebar.error(f'入力されたテキストが{max_length_2}文字を超えています！')
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
st.sidebar.markdown(f"サンプルデータA（ソニー有価証券報告書　2023年3月期　経営方針等） {download_link_A}", unsafe_allow_html=True)

user_input_text_B1 = st.sidebar.text_area("会社名（例：パナソニック）", key='user_input_text_B1')
if len(user_input_text_B1) >max_length_1:
    st.sidebar.error(f'入力されたテキストが{max_length_1}文字を超えています！')
    user_input_text_B1 = user_input_text_B1[:max_length_1]  # 入力を最大文字数まで切り詰める
user_input_text_B2 = st.sidebar.text_area("文章を入力してください:", key='user_input_text_B2')
if len(user_input_text_B2) >max_length_2:
    st.sidebar.error(f'入力されたテキストが{max_length_2}文字を超えています！')
    user_input_text_B2 = user_input_text_B2[:max_length_2]  # 入力を最大文字数まで切り詰める

# テキストファイルを読み込む
with open("sample_B.txt", "r", encoding="utf-8") as f:
    text_content_B = f.read()

# サイドバーにテキストファイルのダウンロードリンクを作成
download_link_B = get_text_download_link(text_content_B, "sample_B.txt")
st.sidebar.markdown(f"サンプルデータB（パナソニック有価証券報告書　2023年3月期　経営方針等）　 {download_link_B}", unsafe_allow_html=True)

st.sidebar.title("ステップ２　前処理")
st.sidebar.write("以下の前処理が自動で実行されます。")
st.sidebar.write("・データクリーニング（空白、改行の削除）")
st.sidebar.write("・大文字→小文字")
st.sidebar.write("・全角→半角")
st.sidebar.write("・トークン化（形態素解析器にはJanomeを使用")

#%%

def maesyori(text):
    # 全角記号を半角に変換
    table = str.maketrans({chr(0xFF01 + i): chr(0x21 + i) for i in range(94)})
    text = text.translate(table)
    
    # テキストを小文字に変換
    text = text.lower()
    
    # 特定の文字列の置換
    text = text.replace("キャッシュ・フロー", "キャッシュフロー")
    
    # 空白（全角、半角）の削除
    text = text.replace(" ", "")  # 半角空白の削除
    text = text.replace("　", "")  # 全角空白の削除
    
    # 行ごとに処理
    lines = text.splitlines()
    processed_lines = []
    for line in lines:
        # 空行でなければ処理
        if line:
            # 句点で終わっていない行に句点を追加
            if not line.endswith("。"):
                line += "。"
            processed_lines.append(line)
    
    # 改行文字を用いて行を結合
    text = "\n".join(processed_lines)
    
    return text

user_input_text_A2 = maesyori(user_input_text_A2)
user_input_text_B2 = maesyori(user_input_text_B2)

#最初の分析
st.title("ステップ３　分析")
st.subheader("①　文字数、単語数、文章数")
st.write("入力した文章の前処理が終了すると、文字数、単語数、文章数が出力されますので確認してください。文章は長いですか、それとも短いですか？文字数と単語数の割合などどうなっていますか？")

# JanomeのTokenizerを初期化
tokenizer = Tokenizer(udic="user_dic.csv", udic_enc='utf8')

# DataFrameを初期化します。
df = pd.DataFrame(columns=['社名', '文字数', '単語数', '文章数', '平均文章長'])

def process_text(user_input_text, company_name):
    global df
    if user_input_text:
        # 文字数をカウント
        char_count = len(user_input_text)

        # 単語数をカウント
        tokens = [token.surface for token in tokenizer.tokenize(user_input_text)]
        word_count = len(tokens)

        # 文章数をカウント
        sentence_count = len(re.split('。', user_input_text)) - 1

        # 平均的な文章の長さを計算（文章数が0でない場合）
        avg_sentence_length = int(round(char_count / sentence_count)) if sentence_count else 0
        
        # 社名が空の場合にデフォルト名を設定
        if not company_name:
            if df.empty or df['社名'].isnull().all():
                company_name = 'A'
            else:
                company_name = 'B'
        
        # 新しい行を既存のDataFrameに追加
        df.loc[len(df)] = [company_name, char_count, word_count, sentence_count, avg_sentence_length]

# ユーザー入力テキストを処理します。
process_text(user_input_text_A2, user_input_text_A1 or 'A')  # 'A' as default if name is empty

# user_input_text_B2が空でない場合、処理を実行します。
if user_input_text_B2:
    process_text(user_input_text_B2, user_input_text_B1 or 'B')  # 'B' as default if name is empty

# DataFrameが空でない場合にのみ、DataFrameを転置して表示します。
if not df.empty:
    df_transposed = df.set_index('社名').T
    st.write(df_transposed)

#%%
st.subheader("②　単語の出現頻度")
st.write("続いて単語の出現頻度を分析してみましょう。下のボックスからカウントしたい品詞を選択してください。どのような単語が多く使われているでしょうか？")

def display_pos_frequency(user_input_text, company_name, selected_pos):
    if user_input_text:
        top_words = count_pos_frequency(user_input_text, selected_pos)
        
        # 表示用のデータフレームを作成
        df = pd.DataFrame(top_words, columns=["単語", "出現頻度"])
        
        # 社名が空の場合デフォルトの名前を使用
        if not company_name:
            company_name = 'A' if 'B' not in company_name else 'B'
            
        st.write(f"{company_name} - {selected_pos}の単語と出現頻度")
        
        # データフレームを表示
        st.write(df)

selected_pos = st.selectbox("カウントする品詞を選んでください:", ("名詞", "動詞", "形容詞"), key='my_unique_selectbox_key')

# 両方のデータがある場合、A社とB社のデータを横に連結して表示
if user_input_text_A2 and user_input_text_B2:
    df_a_table = pd.DataFrame(count_pos_frequency(user_input_text_A2, selected_pos), columns=[f"{user_input_text_A1 or 'A'}-単語", f"{user_input_text_A1 or 'A'}-出現回数"])
    df_b_table = pd.DataFrame(count_pos_frequency(user_input_text_B2, selected_pos), columns=[f"{user_input_text_B1 or 'B'}-単語", f"{user_input_text_B1 or 'B'}-出現回数"])

    # reset index before concatenating
    df_a_table.reset_index(drop=True, inplace=True)
    df_b_table.reset_index(drop=True, inplace=True)

    df_merged = pd.concat([df_a_table, df_b_table], axis=1)
    
    st.write(f"{selected_pos}の単語と出現頻度")
    st.write(df_merged)

# A社のデータがある場合、A社の表を表示
elif user_input_text_A2:
    display_pos_frequency(user_input_text_A2, user_input_text_A1 or 'A', selected_pos)

# B社のデータがある場合、B社の表を表示
elif user_input_text_B2:
    display_pos_frequency(user_input_text_B2, user_input_text_B1 or 'B', selected_pos)

#%%
st.subheader("ワードクラウドによる可視化")
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
    
    fig, ax = plt.subplots(figsize=(15, 12))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    st.pyplot(fig)

# A社のストップワード
if user_input_text_A2:
    st.subheader(f"{user_input_text_A1 or 'A'}-ワードクラウド")
    additional_stop_words_A = st.text_area(f"追加ストップワードを入力してください（スペースで区切って複数入力可能）",key='additional_stop_words_A').split()
    display_wordcloud(user_input_text_A2, user_input_text_A1 or 'A', additional_stop_words_A)

# B社のストップワード
if user_input_text_B2:
    st.subheader(f"{user_input_text_B1 or 'B'}-ワードクラウド")
    additional_stop_words_B = st.text_area(f"追加ストップワードを入力してください（スペースで区切って複数入力可能）",key='additional_stop_words_B').split()
    display_wordcloud(user_input_text_B2, user_input_text_B1 or 'B', additional_stop_words_B)


#%%
st.subheader("③　共起ネットワーク")
st.write("共起ネットワークは、単語同士のつながりをイラストにしたもので、どのような単語の組み合わせがみられるかを視覚的にわかりやすく表現できます。")

def display_network_final(user_input_text, company_name, additional_stop_words, slider_key_prefix):
    if not user_input_text.strip():
        st.write(f"{company_name}: テキストが入力されていません。")
        return
    
    # Get user input for jaccard_threshold, min_word_freq, min_cooccurrence, and top_n_edges
    jaccard_threshold = st.slider(f"Jaccard係数の閾値", min_value=0.0, max_value=1.0, value=0.2, step=0.1, key=f"{slider_key_prefix}_jaccard_threshold")
    min_word_freq = st.slider(f"単語の最小出現数", min_value=1, max_value=50, value=5, key=f"{slider_key_prefix}_min_word_freq")
    min_cooccurrence = st.slider(f"最小共起数", min_value=1, max_value=10, value=2, key=f"{slider_key_prefix}_min_cooccurrence")
    top_n_edges = st.slider(f"上位共起関係の数", min_value=1, max_value=60, value=30, key=f"{slider_key_prefix}_top_n_edges")
    
    network = make_network_with_jaccard_enhanced(user_input_text, jaccard_threshold, additional_stop_words, min_word_freq, min_cooccurrence, top_n_edges)
    
    if not network:
        st.write(f"{company_name}: 単語の共起ネットワークが見つかりませんでした。")
        return

    st.pyplot(network)

if user_input_text_A2:
    st.subheader(f"{user_input_text_A1 or 'A'}-共起ネットワーク")
    additional_stop_words_A1 = st.text_area(f"追加ストップワードを入力してください（スペースで区切って複数入力可能）", key='additional_stop_words_A1').split()
    display_network_final(user_input_text_A2, user_input_text_A1 or 'A', additional_stop_words_A1, 'A')

if user_input_text_B2:
    st.subheader(f"{user_input_text_B1 or 'B'}-共起ネットワーク")
    additional_stop_words_B1 = st.text_area(f"追加ストップワードを入力してください（スペースで区切って複数入力可能）", key='additional_stop_words_B1').split()
    display_network_final(user_input_text_B2, user_input_text_B1 or 'B', additional_stop_words_B1, 'B')

#%%
st.subheader("④　可読性")
st.write("可読性は、上級後半（超難しい）から初級前半（超易しい）まで文章の読みやすさを6段階で評価します。文章はどれくらい読みやすいでしょうか？")

def get_readability_data(user_input, company_name):
    if user_input:
        # 可読性スコアとその他の指標を計算
        score, buncho, kango, wago, dousi, jyosi = readability(user_input)
        
        # 各指標を四捨五入
        score = round(score, 2)
        buncho = round(buncho, 2)
        #kango = round(kango, 2)
        #wago = round(wago, 2)
        #dousi = round(dousi, 2)
        #jyosi = round(jyosi, 2)
        
        # スコアに基づくレベルの判定
        if 0.5 <= score < 1.5:
            level = "超難しい"
        elif 1.5 <= score < 2.5:
            level = "難しい"
        elif 2.5 <= score < 3.5:
            level = "やや難しい）"
        elif 3.5 <= score < 4.5:
            level = "ふつう"
        elif 4.5 <= score < 5.5:
            level = "易しい"
        elif 5.5 <= score < 6.5:
            level = "超易しい"
        else:
            level = "判定不能"
        
        return [score, level, buncho]

def display_readability(user_input_text_A1, user_input_text_A2, user_input_text_B1, user_input_text_B2):
    data = []
    if user_input_text_A2:
        data_A = get_readability_data(user_input_text_A2, user_input_text_A1)
        data.append(pd.Series(data_A, name=user_input_text_A1 or 'A'))

    if user_input_text_B2:
        data_B = get_readability_data(user_input_text_B2, user_input_text_B1)
        data.append(pd.Series(data_B, name=user_input_text_B1 or 'B'))

    if data:
        df = pd.DataFrame(data).T
        df.index = ['可読性スコア', '可読性レベル', '一文の語数']
        st.write("可読性スコアとレベル",df)

# A社の入力があり、B社の入力がない場合
if user_input_text_A2 or user_input_text_B2:
    display_readability(user_input_text_A1, user_input_text_A2, user_input_text_B1, user_input_text_B2)

if user_input_text_A1:
    # テキストがある場合は、そのテキストに"＋語種"を追加してサブヘッダーに設定
    subheader_text_A1 = f"{user_input_text_A1}-語種"
else:
    # テキストがない場合は、"A＋語種"としてサブヘッダーに設定
    subheader_text_A1 = "A-語種"

if user_input_text_B1:
    # テキストがある場合は、そのテキストに"＋語種"を追加してサブヘッダーに設定
    subheader_text_B1 = f"{user_input_text_B1}-語種"
else:
    # テキストがない場合は、"A＋語種"としてサブヘッダーに設定
    subheader_text_B1 = "B-語種"

if user_input_text_A2:
    # 語種の割合を計算
    gosyu_ratios = calculate_gosyu_ratios(user_input_text_A2)
    gosyu_counts = identify_gosyu(user_input_text_A2)
    gosyu_counts.pop('記号・数字', None)  # symbols_numbersを除外

    # 表とグラフを並べて表示
    col1, col2 = st.columns(2)

    with col1:
        st.subheader(subheader_text_A1)
        # インデックスをリセットして削除
        df_gosyu = pd.DataFrame(list(gosyu_counts.items()), columns=['語種', '例'])
        df_gosyu = df_gosyu.set_index('語種')
        st.dataframe(df_gosyu)  # 語種の数を表示

        with col2:
            st.subheader("円グラフ")
            # 円グラフのカラーパレットをアースカラーに設定
            earth_colors = ['#8FBC8F', '#F5DEB3', '#EEE8AA', '#C2B280', '#BDB76B']  # アースカラーパレットに5色を設定
            # 日本語フォントの設定
            font_path = "MEIRYO.TTC"  # フォントのパスを指定
            font_prop = FontProperties(fname=font_path, size=20)  # フォントサイズを設定
            # 円グラフの描画設定
            labels = list(gosyu_ratios.keys())
            sizes = list(gosyu_ratios.values())
            # グラフのサイズを大きくする
            fig, ax = plt.subplots(figsize=(10, 8))  # figsizeを調整してグラフのサイズを大きくする
            plt.rcParams['font.size'] = 16  # パーセンテージのフォントサイズを調整
            wedges, texts, autotexts = ax.pie(sizes, labels=['']*len(labels), autopct='%1.1f%%', startangle=90, colors=earth_colors, pctdistance=0.85)  # pctdistanceを調整
            # 凡例をグラフの外側に配置する
            plt.legend(wedges, labels, loc='center left', bbox_to_anchor=(1, 0, 0.5, 1), prop=font_prop)
            for text in autotexts:
                text.set_fontproperties(font_prop)
                text.set_color('Black')
            ax.axis('equal')  # 円形を保つ
            st.pyplot(fig)  # Streamlitにグラフを表示
            
if user_input_text_B2:
    # 語種の割合を計算
    gosyu_ratios = calculate_gosyu_ratios(user_input_text_B2)
    gosyu_counts = identify_gosyu(user_input_text_B2)
    gosyu_counts.pop('記号・数字', None)  # symbols_numbersを除外

    # 表とグラフを並べて表示
    col1, col2 = st.columns(2)

    with col1:
        st.subheader(subheader_text_B1)
        df_gosyu = pd.DataFrame(list(gosyu_counts.items()), columns=['語種', '例'])
        df_gosyu = df_gosyu.set_index('語種')
        st.dataframe(df_gosyu)  # 語種の数を表示
            
        with col2:
            st.subheader("円グラフ")
            # 円グラフのカラーパレットをアースカラーに設定
            earth_colors = ['#8FBC8F', '#F5DEB3', '#EEE8AA', '#C2B280', '#BDB76B']  # アースカラーパレットに5色を設定
            # 日本語フォントの設定
            font_path = "MEIRYO.TTC"  # フォントのパスを指定
            font_prop = FontProperties(fname=font_path, size=20)  # フォントサイズを設定
            # 円グラフの描画設定
            labels = list(gosyu_ratios.keys())
            sizes = list(gosyu_ratios.values())
            # グラフのサイズを大きくする
            fig, ax = plt.subplots(figsize=(10, 8))  # figsizeを調整してグラフのサイズを大きくする
            plt.rcParams['font.size'] = 16  # パーセンテージのフォントサイズを調整
            wedges, texts, autotexts = ax.pie(sizes, labels=['']*len(labels), autopct='%1.1f%%', startangle=90, colors=earth_colors, pctdistance=0.85)  # pctdistanceを調整
            # 凡例をグラフの外側に配置する
            plt.legend(wedges, labels, loc='center left', bbox_to_anchor=(1, 0, 0.5, 1), prop=font_prop)
            for text in autotexts:
                text.set_fontproperties(font_prop)
                text.set_color('Black')
            ax.axis('equal')  # 円形を保つ
            st.pyplot(fig)  # Streamlitにグラフを表示
        
#%%
st.subheader("⑤　トーン")
st.write("トーンは－1（超ネガティブ）から1（超ポジティブ）で計算されます。0は中立（ニュートラル）となります。 文章のトーンはどのくらいポジティブ（ネガティブ）でしょうか？")

def display_tone_data(user_input_text_A1, user_input_text_A2, user_input_text_B1, user_input_text_B2):
    # デフォルトの社名を設定
    company_A = user_input_text_A1 if user_input_text_A1 else "A"
    company_B = user_input_text_B1 if user_input_text_B1 else "B"
    
    # 結果を格納するデータフレームを初期化
    df_tone = pd.DataFrame(index=['トーンスコア', 'トーンレベル'])
    df_positive_words_A = pd.DataFrame()
    df_negative_words_A = pd.DataFrame()
    df_positive_words_B = pd.DataFrame()
    df_negative_words_B = pd.DataFrame()

    # A社のデータを処理
    if user_input_text_A2:
        score_A, _, _, top_pwords_A, top_nwords_A = tone_score(user_input_text_A2)
        evaluation_A = tone_eval(score_A)
        df_tone[company_A] = [score_A, evaluation_A]
        df_positive_words_A = pd.DataFrame({f"{company_A} ポジティブ単語": [word for word, _ in top_pwords_A],
                                            f"{company_A} ポジティブ頻度": [freq for _, freq in top_pwords_A]})
        df_negative_words_A = pd.DataFrame({f"{company_A} ネガティブ単語": [word for word, _ in top_nwords_A],
                                            f"{company_A} ネガティブ頻度": [freq for _, freq in top_nwords_A]})
        
    # B社のデータを処理
    if user_input_text_B2:
        score_B, _, _, top_pwords_B, top_nwords_B = tone_score(user_input_text_B2)
        evaluation_B = tone_eval(score_B)
        df_tone[company_B] = [score_B, evaluation_B]
        df_positive_words_B = pd.DataFrame({f"{company_B} ポジティブ単語": [word for word, _ in top_pwords_B],
                                            f"{company_B} ポジティブ頻度": [freq for _, freq in top_pwords_B]})
        df_negative_words_B = pd.DataFrame({f"{company_B} ネガティブ単語": [word for word, _ in top_nwords_B],
                                            f"{company_B} ネガティブ頻度": [freq for _, freq in top_nwords_B]})

    # Streamlitでトーンスコアとレベルを表示
    if not df_tone.empty:
        st.write("トーンスコアとレベル", df_tone) 
    
    # 両方の企業のポジティブ単語データフレームがある場合、結合して表示
    if not df_positive_words_A.empty or not df_positive_words_B.empty:
        df_positive_words = pd.concat([df_positive_words_A, df_positive_words_B], axis=1)
        st.write("ポジティブ単語と頻度", df_positive_words)

    # 両方の企業のネガティブ単語データフレームがある場合、結合して表示
    if not df_negative_words_A.empty or not df_negative_words_B.empty:
        df_negative_words = pd.concat([df_negative_words_A, df_negative_words_B], axis=1)
        st.write("ネガティブ単語と頻度", df_negative_words)

# ユーザー入力に応じて関数を実行
if user_input_text_A2 or user_input_text_B2:
    display_tone_data(user_input_text_A1, user_input_text_A2, user_input_text_B1, user_input_text_B2)
#%%
# タイトルを設定
st.subheader("⑥　類似度")
st.write("２つの文章はどのくらい似ていますか？なぜ似ている（似ていない）のか自分でも考えてみましょう。")

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
    
# デフォルトの社名を設定
company_A = user_input_text_A1 if user_input_text_A1 else "A"
company_B = user_input_text_B1 if user_input_text_B1 else "B"

# 両方のテキストが入力された場合、類似度を計算
if user_input_text_A2 and user_input_text_B2:
    # TF-IDFベクトルを計算
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([user_input_text_A2, user_input_text_B2])

    # コサイン類似度を計算
    cos_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    cos_sim = round(cos_sim, 3)
    
    # 類似度とそっくりレベルをDataFrameに保存
    evaluation = evaluate_sim(cos_sim)
    data = {
        '類似度': [cos_sim],
        'そっくりレベル': [evaluation]
    }
    df = pd.DataFrame(data)
    
    # DataFrameを転置
    df_transposed = df.transpose()
    df_transposed.columns = [f"{company_A}と{company_B}"]
    
    # 転置したDataFrameをst.writeで表示
    st.write(df_transposed)


#%%
st.title('おわりに')
st.write("テキスト分析はいかがでしたでしょうか？次はぜひ自分の興味のある文章を入れて結果を確かめてみましょう！")

st.write("【サイト運営者】")
st.write("青山学院大学　経営学部　矢澤憲一研究室")

st.write("【免責事項】")
st.write("このウェブサイトおよびそのコンテンツは、一般的な情報提供を目的としています。このウェブサイトの情報を使用または適用することによって生じるいかなる利益、損失、損害について、当ウェブサイトおよびその運営者は一切の責任を負いません。情報の正確性、完全性、時宜性、適切性についても、一切保証するものではありません。")
