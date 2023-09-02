'''
streamlit run D:\GoogleDrive\python\python_code\streamlit_app_2\app.py
'''

import streamlit as st
from PIL import Image
from sudachipy import Dictionary
import base64
import base64
import re

st.title("ネコでも使える！テキスト分析（β版）") # タイトル
st.write("少しずつ機能を追加していきたいと思います。")

image = Image.open("title.png")
st.image(image,use_column_width=True)

st.write("私の研究室では「会計・財務研究におけるテキスト分析」に取り組んでいます。研究活動の一環として、テキスト分析の魅力を体感できるウェブサイトを作成しました。肩の力を抜いてお楽しみください！")
st.write("最終更新日：2023年8月31日")

# %%
st.title("はじめに")
st.write("テキストマイニングとは！？")

st.write("テキストマイニングとは、簡単に言えば、大量のテキストデータの中に埋もれている「意味のある情報」を自然言語処理（Natural Language Processing）という技術を用いて取り出すことです。本当はプログラミング言語でコードを書いたりなどちょっと面倒な作業があるのですが、ここではそんな過程をすっとばして、すぱっとさっくりテキスト分析を楽しんでみましょう。")

st.title("準備")
st.write("下記のボックスに文章を入力してみましょう！サンプルデータを用意していますのでコピー＆ペーストしてください。")

# SudachiPyの辞書を初期化
dictionary = Dictionary().create()

# ユーザーに文章を入力してもらう
user_input_text = st.text_area("文章を入力してください:")

# テキストファイルを読み込む
with open("sample_A.txt", "r", encoding="utf-8") as f:
    text_content = f.read()

# テキストファイルをダウンロードするための関数
def get_text_download_link(text, filename):
    """Generates a link allowing the text to be downloaded"""
    b64 = base64.b64encode(text.encode()).decode()  # 文字をbase64エンコードする
    return f'<a href="data:file/txt;base64,{b64}" download="{filename}">Download text file</a>'

# テキストファイルのダウンロードリンクを作成
st.write("サンプルテキスト")
st.markdown(f"【トヨタ自動車　2023年3月期　決算短信】　 {get_text_download_link(text_content, 'sample_A.txt')} ", unsafe_allow_html=True)

st.title("ステップ１　文字数、単語数、文章数")
st.write("まずは入力した文章の文字数、単語数、文章数が出力されますので確認してください。文章は長いですか、それとも短いですか？文字数と単語数の割合などどうなっていますか？")

# テキストが入力された場合の処理
if user_input_text:
    user_input_text = user_input_text.translate(str.maketrans({chr(0xFF01 + i): chr(0x21 + i) for i in range(len(user_input_text))}))
    user_input_text = user_input_text.replace("キャッシュ・フロー","キャッシュフロー")

    # 文字数をカウント
    char_count = len(user_input_text)
    st.write(f'文字数： {char_count} 字')

    # 単語数をカウント
    tokens = dictionary.tokenize(user_input_text)
    word_count = len(tokens)
    st.write(f'単語数： {word_count} 語')

    # 文章数をカウント
    sentence_count = len(re.split('[。.!?]', user_input_text)) - 1
    st.write(f'文章数： {sentence_count} 文')

