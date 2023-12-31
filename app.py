import plotly.express as px
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


st.title("ネコでも使える！会計テキストマイニング") # タイトル

image = Image.open("title.png") 
st.image(image,use_column_width=True)

st.write("青山学院大学矢澤研究室では「会計・財務データを用いたテキスト分析」に取り組んでいます。研究活動の一環として、テキスト分析の魅力を体感できるウェブアプリケーションを作成しました。肩の力を抜いてお楽しみください！")
st.write("ご利用にあたっては、ページ最下段の【諸注意】と【免責事項】もお読みください。")
st.write("最終更新日：2023年12月1日")

# %%
st.title("はじめに")
st.write("テキストマイニングとは！？")
st.write("テキストマイニングとは、簡単に言えば、大量のテキストデータの中に埋もれている「意味のある情報」を自然言語処理という技術を用いて取り出すことです。本当はプログラミング言語でコードを書いたりなどちょっと面倒な作業があるのですが、ここではそんな過程をすっとばして、すぱっとさっくりテキスト分析を楽しんでみましょう。")

#%%
# サイドバーにテキストを表示
st.sidebar.title("ステップ１　データ収集")
st.sidebar.write("下記のボックスに文章を入力してみましょう！サンプルデータを用意していますので、必要に応じてコピー＆ペーストしてください。2社比較したい場合はもう1社入力してください。")
st.sidebar.write("会社名は最大30文字、文章は最大30,000字まで。")

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

st.write("ステップ１はサイドバーに表示されています。説明に従ってデータを入力してください。")

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

st.title("ステップ２　前処理")
st.write("前処理は自動で実行されます。")

#最初の分析
st.title("ステップ３　分析")
st.subheader("①　文字数、単語数、文章数")
st.write("まずは入力した文章の文字数、単語数、文章数が出力されますので確認してください。文章は長いですか、それとも短いですか？文字数と単語数の割合などどうなっていますか？")

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
if user_input_text_A2:
    user_input_text_A2 = maesyori(user_input_text_A2)
    process_text(user_input_text_A2, user_input_text_A1 or 'A')  # 'A' as default if name is empty

# user_input_text_B2が空でない場合、処理を実行します。
if user_input_text_B2:
    user_input_text_B2 = maesyori(user_input_text_B2)
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
        
        if not top_words:  # 抽出された単語がない場合
            st.error(f"{company_name}: テキストから{selected_pos}を抽出できませんでした。")
            return

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
    try:
        wordcloud = make_wordcloud(user_input_text, additional_stop_words)
    except ValueError as e:
        # Streamlit UIで日本語のエラーメッセージを表示
        st.error(f"{company_name}: {e}")
        return

    fig, ax = plt.subplots(figsize=(15, 12))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    st.pyplot(fig)
    
# A社のストップワード
if user_input_text_A2:
    st.subheader(f"{user_input_text_A1 or 'A'}-ワードクラウド")
    additional_stop_words_A = st.text_area("追加ストップワードを入力してください（スペースで区切って複数入力可能）", key='additional_stop_words_A').split()
    display_wordcloud(user_input_text_A2, user_input_text_A1 or 'A', additional_stop_words_A)

# B社のストップワード
if user_input_text_B2:
    st.subheader(f"{user_input_text_B1 or 'B'}-ワードクラウド")
    additional_stop_words_B = st.text_area("追加ストップワードを入力してください（スペースで区切って複数入力可能）", key='additional_stop_words_B').split()
    display_wordcloud(user_input_text_B2, user_input_text_B1 or 'B', additional_stop_words_B)
    
#%%
st.subheader("③　共起ネットワーク")
st.write("共起ネットワークは、単語同士のつながりをイラストにしたもので、どのような単語の組み合わせがみられるかを視覚的にわかりやすく表現できます。")

def display_network_final(user_input_text, company_name, additional_stop_words, slider_key_prefix):
    if not user_input_text.strip():
        st.write(f"{company_name}: テキストが入力されていません。")
        return
    
    # Jaccard係数、単語の最小出現数、最小共起数、上位共起関係の数の設定
    jaccard_threshold = st.slider(f"Jaccard係数の閾値", min_value=0.0, max_value=1.0, value=0.2, step=0.1, key=f"{slider_key_prefix}_jaccard_threshold")
    min_word_freq = st.slider(f"単語の最小出現数", min_value=1, max_value=50, value=5, key=f"{slider_key_prefix}_min_word_freq")
    min_cooccurrence = st.slider(f"最小共起数", min_value=1, max_value=10, value=2, key=f"{slider_key_prefix}_min_cooccurrence")
    top_n_edges = st.slider(f"上位共起関係の数", min_value=1, max_value=60, value=30, key=f"{slider_key_prefix}_top_n_edges")
    
    network = make_network_with_jaccard_enhanced(user_input_text, jaccard_threshold, additional_stop_words, min_word_freq, min_cooccurrence, top_n_edges)
    
    if not network:
        st.error(f"{company_name}: 単語の共起ネットワークが見つかりませんでした。")
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
    if not user_input.strip():
        st.error(f"{company_name}: テキストが入力されていません。")
        return None

    readability_result = readability(user_input)
    
    if readability_result is None:
        st.error(f"{company_name}: 可読性のデータを計算できませんでした。")
        return None

    score, buncho, kango, wago, dousi, jyosi = readability_result

    if score is None:
        st.error(f"{company_name}: スコアを計算できませんでした。")
        return None

    score = round(score, 2)
    buncho = round(buncho, 2) if buncho is not None else "N/A"
    
    # スコアに基づくレベルの判定
    if 0.5 <= score < 1.5:
        level = "超難しい"
    elif 1.5 <= score < 2.5:
        level = "難しい"
    elif 2.5 <= score < 3.5:
        level = "やや難しい"
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
    company_A = user_input_text_A1 or 'A'  # A社のデフォルト名を設定
    company_B = user_input_text_B1 or 'B'  # B社のデフォルト名を設定
    
    if user_input_text_A2:
        data_A = get_readability_data(user_input_text_A2, company_A)
        if data_A:
            data.append(pd.Series(data_A, name=company_A))

    if user_input_text_B2:
        data_B = get_readability_data(user_input_text_B2, company_B)
        if data_B:
            data.append(pd.Series(data_B, name=company_B))

    if data:
        df = pd.DataFrame(data).T
        df.index = ['可読性スコア', '可読性レベル', '一文の語数']
        st.write("可読性スコアとレベル", df)

# A社の入力があり、B社の入力がない場合
if user_input_text_A2 or user_input_text_B2:
    display_readability(user_input_text_A1, user_input_text_A2, user_input_text_B1, user_input_text_B2)

if user_input_text_A1:
    # テキストがある場合は、そのテキストに"＋語種"を追加してサブヘッダーに設定
    subheader_text_A1 = f"{user_input_text_A1}-円グラフ"
else:
    # テキストがない場合は、"A＋語種"としてサブヘッダーに設定
    subheader_text_A1 = "A-円グラフ"

if user_input_text_B1:
    # テキストがある場合は、そのテキストに"＋語種"を追加してサブヘッダーに設定
    subheader_text_B1 = f"{user_input_text_B1}-円グラフ"
else:
    # テキストがない場合は、"A＋語種"としてサブヘッダーに設定
    subheader_text_B1 = "B-円グラフ"

if user_input_text_A2:
    # 語種の割合を計算
    gosyu_ratios = calculate_gosyu_ratios(user_input_text_A2)
    gosyu_counts = identify_gosyu(user_input_text_A2)
    gosyu_counts.pop('記号・数字', None)  # symbols_numbersを除外
    
    st.subheader(subheader_text_A1)
    # アースカラーパレットを設定
    earth_colors = ['#8FBC8F', '#F5DEB3', '#EEE8AA', '#C2B280', '#D3D3D3']

    # 最大値の要素を特定
    max_value = max(gosyu_ratios.values())
    pull_values = [0.1 if value == max_value else 0 for value in gosyu_ratios.values()]

    # Plotlyを使って円グラフを作成
    fig = px.pie(values=list(gosyu_ratios.values()), names=list(gosyu_ratios.keys()),
                color_discrete_sequence=earth_colors)

    # 強調表示の設定
    fig.update_traces(textinfo='percent+label', pull=pull_values)
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5)) # 凡例の位置調整
    fig.update_layout(showlegend=False)
    
    # Streamlitにグラフを表示
    st.plotly_chart(fig)

    df_gosyu = pd.DataFrame(list(gosyu_counts.items()), columns=['語種', '例'])
    df_gosyu = df_gosyu.set_index('語種')
    st.dataframe(df_gosyu)  # 語種の数を表示

if user_input_text_B2:
    # 語種の割合を計算
    gosyu_ratios = calculate_gosyu_ratios(user_input_text_B2)
    gosyu_counts = identify_gosyu(user_input_text_B2)
    gosyu_counts.pop('記号・数字', None)  # symbols_numbersを除外

    st.subheader(subheader_text_B1)
            
    # アースカラーパレットを設定
    earth_colors = ['#8FBC8F', '#F5DEB3', '#EEE8AA', '#C2B280', '#D3D3D3']

    # 最大値の要素を特定
    max_value = max(gosyu_ratios.values())
    pull_values = [0.1 if value == max_value else 0 for value in gosyu_ratios.values()]

    # Plotlyを使って円グラフを作成
    fig = px.pie(values=list(gosyu_ratios.values()), names=list(gosyu_ratios.keys()),
                color_discrete_sequence=earth_colors)

    # 強調表示の設定
    fig.update_traces(textinfo='percent+label', pull=pull_values)
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5)) # 凡例の位置調整

    # Streamlitにグラフを表示
    st.plotly_chart(fig)
        
    df_gosyu = pd.DataFrame(list(gosyu_counts.items()), columns=['語種', '例'])
    df_gosyu = df_gosyu.set_index('語種')
    st.dataframe(df_gosyu)  # 語種の数を表示
        
#%%
st.subheader("⑤　トーン")
st.write("トーンは－1（超ネガティブ）から1（超ポジティブ）で計算されます。0は中立（ニュートラル）となります。 文章のトーンはどのくらいポジティブ（ネガティブ）でしょうか？")

def display_tone_data(user_input_text_A1, user_input_text_A2, user_input_text_B1, user_input_text_B2):
    company_A = user_input_text_A1 if user_input_text_A1 else "A"
    company_B = user_input_text_B1 if user_input_text_B1 else "B"

    df_tone = pd.DataFrame(index=['トーンスコア', 'トーンレベル'])
    df_positive_words_A = pd.DataFrame()
    df_negative_words_A = pd.DataFrame()
    df_positive_words_B = pd.DataFrame()
    df_negative_words_B = pd.DataFrame()

    if user_input_text_A2:
        score_A, _, _, top_pwords_A, top_nwords_A = tone_score(user_input_text_A2)
        if score_A == "データなし":
            st.error(f"{company_A}: データが不足しているため、トーンスコアを計算できません。")
        else:
            evaluation_A = tone_eval(score_A)
            df_tone[company_A] = [score_A, evaluation_A]
            df_positive_words_A = pd.DataFrame({f"{company_A} ポジティブ単語": [word for word, _ in top_pwords_A],
                                                f"{company_A} ポジティブ頻度": [freq for _, freq in top_pwords_A]})
            df_negative_words_A = pd.DataFrame({f"{company_A} ネガティブ単語": [word for word, _ in top_nwords_A],
                                                f"{company_A} ネガティブ頻度": [freq for _, freq in top_nwords_A]})
        
    if user_input_text_B2:
        score_B, _, _, top_pwords_B, top_nwords_B = tone_score(user_input_text_B2)
        if score_B == "データなし":
            st.error(f"{company_B}: データが不足しているため、トーンスコアを計算できません。")
        else:
            evaluation_B = tone_eval(score_B)
            df_tone[company_B] = [score_B, evaluation_B]
            df_positive_words_B = pd.DataFrame({f"{company_B} ポジティブ単語": [word for word, _ in top_pwords_B],
                                                f"{company_B} ポジティブ頻度": [freq for _, freq in top_pwords_B]})
            df_negative_words_B = pd.DataFrame({f"{company_B} ネガティブ単語": [word for word, _ in top_nwords_B],
                                                f"{company_B} ネガティブ頻度": [freq for _, freq in top_nwords_B]})

    if not df_tone.empty:
        st.write("トーンスコアとレベル", df_tone) 

    if not df_positive_words_A.empty or not df_positive_words_B.empty:
        df_positive_words = pd.concat([df_positive_words_A, df_positive_words_B], axis=1)
        if df_positive_words.empty:
            st.write("ポジティブな単語が見つかりませんでした。")
        else:
            st.write("ポジティブ単語と頻度", df_positive_words)

    if not df_negative_words_A.empty or not df_negative_words_B.empty:
        df_negative_words = pd.concat([df_negative_words_A, df_negative_words_B], axis=1)
        if df_negative_words.empty:
            st.write("ネガティブな単語が見つかりませんでした。")
        else:
            st.write("ネガティブ単語と頻度", df_negative_words)

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

try:
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
        
    # 片方のテキストのみが入力された場合
    elif user_input_text_A2 or user_input_text_B2:
        st.error('両方のテキストデータを入力してください。')
        
except ValueError as e:
    if 'empty vocabulary' in str(e):
        st.error('入力されたテキストが短すぎるか、ストップワードのみ含まれています。もっと長いテキストを入力してください。')
    else:
        raise  # その他のValueErrorはそのまま再発生させる

st.title('おわりに')
st.write("テキスト分析はいかがでしたでしょうか？次はぜひ自分の興味のある文章を入れて結果を確かめてみましょう！")

st.write("【サイト運営者】")
st.write("青山学院大学　経営学部　矢澤憲一研究室")

st.write("【諸注意】")
st.write("１．私的目的での利用について：")
st.write("本ウェブアプリケーションは、個人的な用途で自由にご利用いただけます。しかしながら、公序良俗に反する行為は固く禁じられています。利用者の皆様には、社会的な規範を尊重し、責任ある行動をお願いいたします。")
st.write("２．ビジネス目的での利用について：")
st.write("本アプリケーションをビジネス目的で使用される場合は、事前に以下の連絡先までご一報ください。使用に関する詳細な情報を提供いたします。")
st.write("２．学術論文執筆目的での利用について：")
st.write("学術論文の執筆に当たり、本アプリケーションのデータや機能を利用される場合は、下記の文献を参考文献として明記し、同時に以下の連絡先までご一報いただくようお願いいたします。")
st.write("参考文献：矢澤憲一（2023）「ネコでも使える！会計テキストマイニング第1回～第3回」『週刊経営財務』3632～3634号。")
st.write("連絡先：yazawa(at)busi.aoyama.ac.jp")

st.write("【免責事項】")
st.write("本ウェブアプリケーションおよびそのコンテンツは、一般的な情報提供を目的としています。このウェブサイトの情報を使用または適用することによって生じるいかなる利益、損失、損害について、当ウェブサイトおよびその運営者は一切の責任を負いません。情報の正確性、完全性、時宜性、適切性についても、一切保証するものではありません。")

