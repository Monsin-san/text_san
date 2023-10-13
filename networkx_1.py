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


#%%
def jaccard_similarity(set1, set2):
    """
    Calculate the Jaccard similarity between two sets.
    """
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0

def make_network_with_jaccard_enhanced(text, jaccard_threshold, additional_stop_words=[], min_word_freq=1, min_cooccurrence=1, max_nodes=None):
    font = FontProperties(fname="MEIRYO.TTC")
    
    # Janomeの設定
    tokenizer = Tokenizer(udic='user_dic.csv', udic_enc='utf8')

    stop_words = set(["%)","これ", "それ", "あれ", "この", "その", "あの", "ここ", "そこ", "あそこ", "こちら", "どこ", "だれ", "なに", "なん", "何", "私", "こと", "もの"])
    stop_words.update(additional_stop_words)
    
    sentences = re.split(r'。', text)
    
    word_to_sentences = {}
    for sentence in sentences:
        for token in tokenizer.tokenize(sentence):
            if token.part_of_speech.split(',')[0] == "名詞":
                word = token.surface
                if len(word) == 1 or re.match(r'^\d+$', word) or word in stop_words:
                    continue
                if word not in word_to_sentences:
                    word_to_sentences[word] = set()
                word_to_sentences[word].add(sentence)
                
    # Filter words with frequency less than min_word_freq
    word_to_sentences = {word: sent_set for word, sent_set in word_to_sentences.items() if len(sent_set) >= min_word_freq}

    G = nx.Graph()
    words = list(word_to_sentences.keys())
    edge_data = []

    for i in range(len(words)):
        for j in range(i+1, len(words)):
            word1 = words[i]
            word2 = words[j]
            
            jaccard_coeff = jaccard_similarity(word_to_sentences[word1], word_to_sentences[word2])
            
            # Only add an edge if jaccard_coeff is above the threshold and the co-occurrence is above min_cooccurrence
            if jaccard_coeff >= jaccard_threshold and len(word_to_sentences[word1].intersection(word_to_sentences[word2])) >= min_cooccurrence:
                edge_data.append((word1, word2, jaccard_coeff))
    
    # If max_nodes is defined, sort the edges by weight and keep only the top ones
    if max_nodes:
        edge_data.sort(key=lambda x: x[2], reverse=True)
        edge_data = edge_data[:max_nodes]

    for word1, word2, weight in edge_data:
        G.add_edge(word1, word2, weight=weight)

    if not G.edges():
        return None

    communities = community.greedy_modularity_communities(G)
    color_map = {node: i for i, com in enumerate(communities) for node in com}
    node_colors = [color_map.get(node, 0) for node in G.nodes()]

    pos = nx.spring_layout(G, k=0.7, iterations=50, scale=2.0, seed=30)
    edge_width = [G[u][v]['weight'] * 3 for u, v in G.edges()]
    node_sizes = [len(word_to_sentences[node]) * 500 for node in G.nodes()]  # Node size based on word frequency

    # 次数中心性を計算
    degree_centrality = nx.degree_centrality(G)

    # 次数中心性に基づいてノードの大きさを調整
    scaling_factor = 50000  # この値を調整して、ノードの大きさのスケールを変更します
    #node_sizes = [degree_centrality[node] * scaling_factor for node in G.nodes()]

    fig, ax = plt.subplots(figsize=(16, 16))
    nx.draw(G, pos, with_labels=False, node_color=node_colors, node_size=node_sizes, width=edge_width, cmap=plt.get_cmap("Pastel2"), ax=ax)

    for node, (x, y) in pos.items():
        plt.text(x, y, node, fontproperties=font, fontsize=16, ha='center', va='center')

    edge_labels = {e: "{:.2f}".format(G[e[0]][e[1]]["weight"]) for e in G.edges}
    for (node1, node2), label in edge_labels.items():
        x1, y1 = pos[node1]
        x2, y2 = pos[node2]
        x, y = (x1 + x2) / 2, (y1 + y2) / 2
        plt.text(x, y, str(label), fontproperties=font, fontsize=12, ha='center', va='center')

    plt.axis('off')
    return fig
