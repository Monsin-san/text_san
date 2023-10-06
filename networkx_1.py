from collections import Counter
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import networkx as nx
from networkx.algorithms import community
import re
from janome.tokenizer import Tokenizer
from janome.charfilter import *
from janome.tokenfilter import *

def make_network(text, frequency_threshold, additional_stop_words=[]):
    font = FontProperties(fname="MEIRYO.TTC")
    
    # Janomeの設定
    tokenizer = Tokenizer(udic='user_dic.csv', udic_enc='utf8')

    stop_words = set(["%)","これ", "それ", "あれ", "この", "その", "あの", "ここ", "そこ", "あそこ", "こちら", "どこ", "だれ", "なに", "なん", "何", "私", "こと", "もの"])
    stop_words.update(additional_stop_words)  # 追加のストップワードを含める
    
    # 強制的にカウントしたい単語リスト
    forced_words = []
    
    text = text.translate(str.maketrans({chr(0xFF01 + i): chr(0x21 + i) for i in range(94)}))
    text = text.replace("キャッシュ・フロー","キャッシュフロー")
    
    nodes = []
    for token in tokenizer.tokenize(text):
        if token.part_of_speech.split(',')[0] == "名詞":
            word = token.surface if token.base_form not in forced_words else token.base_form
            if len(word) == 1 or re.match(r'^\d+$', word) or word in stop_words:
                continue
            nodes.append(word)

    bigram_counts = Counter(zip(nodes, nodes[1:]))
    node_freq = Counter(nodes)

    G = nx.Graph()
    for (word1, word2), freq in bigram_counts.items():
        if freq >= frequency_threshold:
            G.add_edge(word1, word2, weight=freq)
    
    if not G.edges():
        print("No edges in graph.")
        return None

    communities = community.greedy_modularity_communities(G)
    color_map = {node: i for i, com in enumerate(communities) for node in com}
    node_colors = [color_map.get(node, 0) for node in G.nodes()]

    pos = nx.spring_layout(G, k=0.7, iterations=50, scale=2.0, seed=30)
    edge_width = [G[u][v]['weight'] * 0.2 for u, v in G.edges()]
    node_size = [node_freq[node] * 300 for node in G.nodes()]

    fig, ax = plt.subplots(figsize=(16, 16))
    nx.draw(G, pos, with_labels=False, node_color=node_colors, node_size=node_size, width=edge_width, cmap=plt.get_cmap("Pastel2"), ax=ax)

    for node, (x, y) in pos.items():
        plt.text(x, y, node, fontproperties=font, fontsize=16, ha='center', va='center')

    edge_labels = {e: G[e[0]][e[1]]["weight"] for e in G.edges}
    for (node1, node2), label in edge_labels.items():
        x1, y1 = pos[node1]
        x2, y2 = pos[node2]
        x, y = (x1 + x2) / 2, (y1 + y2) / 2
        plt.text(x, y, str(label), fontproperties=font, fontsize=12, ha='center', va='center')

    plt.axis('off')
    return fig
