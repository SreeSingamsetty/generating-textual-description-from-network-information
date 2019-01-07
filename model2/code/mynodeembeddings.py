# -*- coding: utf-8 -*-
from node2vec import Node2Vec
import networkx as nx
import gensim

# Generate walks
graph = nx.read_edgelist("33.csv",delimiter="\t", encoding="utf-8")
node2vec = Node2Vec(graph, dimensions=200, walk_length=20, num_walks=10)
model = node2vec.fit(window=10)

model.wv.save_word2vec_format("33_200d_20wl10nm.emb")
