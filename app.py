'''
streamlit run D:\GoogleDrive\python\python_code\streamlit_app_2\app.py
'''

import streamlit as st
from PIL import Image

st.title("ネコでも使える！テキスト分析（β版）") # タイトル
st.write("少しずつ機能を追加していきたいと思います。")

image = Image.open("title.png")
st.image(image,use_column_width=True)

st.write("私の研究室では「会計・財務研究におけるテキスト分析」に取り組んでいます。研究活動の一環として、テキスト分析の魅力を体感できるウェブサイトを作成しました。肩の力を抜いてお楽しみください！")
st.write("最終更新日：2023年8月31日")

# %%
