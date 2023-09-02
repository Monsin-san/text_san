#%%
from collections import Counter
import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms import community  # networkxのバージョンが2.0以上の場合
from sudachipy import Dictionary, Tokenizer  # SudachiPyをインポート
import re

def make_network(text, frequency_threshold):
    plt.rcParams['font.family'] = 'Meiryo'

    tokenizer = Dictionary().create()  # SudachiPy

    # ストップワードの定義
    stop_words = set(["これ", "それ", "あれ", "この", "その", "あの", "ここ", "そこ", "あそこ", "こちら", "どこ", "だれ", "なに", "なん", "何", "私", "こと", "もの"])

    text = text.translate(str.maketrans({chr(0xFF01 + i): chr(0x21 + i) for i in range(len(text))}))
    text = text.replace("キャッシュ・フロー","キャッシュフロー")

    nodes = []
    for token in tokenizer.tokenize(text):
        part_of_speech = token.part_of_speech()[0]
        if part_of_speech == "名詞":
            word = token.surface()

            if len(word) == 1 or re.match(r'^\d+$', word) or word in stop_words:
                continue

            nodes.append(word)

    bigrams = [(nodes[i], nodes[i + 1]) for i in range(len(nodes) - 1)]
    bigram_counts = Counter(bigrams)
    node_freq = Counter(nodes)

    G = nx.Graph()

    for bigram, freq in bigram_counts.items():
        if freq >= frequency_threshold:
            word1, word2 = bigram
            G.add_edge(word1, word2, weight=freq)

    if len(G.edges()) == 0:
        print("No edges in graph.")
        return None

    communities = community.greedy_modularity_communities(G)

    color_map = {}
    for i, com in enumerate(communities):
        for node in com:
            color_map[node] = i

    node_colors = [color_map.get(node, 0) for node in G.nodes()]

    pos = nx.spring_layout(G, k=0.5, iterations=50, scale=2.0, seed=30)
    edge_width = [G[u][v]['weight'] * 0.2 for u, v in G.edges()]
    labels = {e: G[e[0]][e[1]]["weight"] for e in G.edges}
    node_size = [node_freq[node] * 300 for node in G.nodes()]

    fig, ax = plt.subplots(figsize=(16, 16))
    nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=node_size, font_size=16, font_family='Meiryo', width=edge_width, cmap=plt.get_cmap("Pastel2"), ax=ax)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_size=12, font_family='Meiryo')

    plt.axis('off')

    return fig
