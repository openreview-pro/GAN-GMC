<h1 align = "center">GAN-GMC: Multi-Graph Representation Learning with GANs for Geometric Matrix Completion</h1>

<h1 align = "center">Abstract</h1> 

With careful consideration of the pairwise graph relationships between users/items, Geometric Matrix Completion (GMC) has been widely explored for recommender systems. Existing methods for GMC with Graph Neural Networks (GNN) usually follow a generative manner, i.e. building learning modules to reconstruct a new rating matrix. This methodology however relies heavily on a considerable amount of observed ratings and favorable GNN models. In this paper, by virtue of the striking success of Generative Adversarial Nets (GAN), we propose an innovative multi-graph representation learning framework with GAN for GMC, which unifies a generator $G$ and a discriminator $D$. With the competition between $G$ and $D$, both of them can alternately and iteratively boost their performance. Further, to build a more powerful generator, we develop a deep recursive inference module for both user and item graph representation learning, which combines graph convolutional networks and bidirectional LSTMs. We evaluate our proposed model on several standard datasets, showing that it outperforms the state-of-the-art approaches. Robustness analysis is also given to show the advantage of our proposed framework in handling GMC tasks with few training samples.

<h1 align = "center">Motivation</h1>

Recently, graph convolutional networks (GCN) have attracted much attention in light of their favorable performance on many graph data modeling tasks. Existing methods for GMC problems with GNN/GCN usually build learning modules to extract the latent feature representations of users and items and then reconstruct a new rating matrix based on the corresponding embeddings. However, this methodology relies heavily on an amount of observed ratings and existing favorable GNN models. Firstly, the observed rating matrix may be extremely sparse (training samples are very few) in real applications, which degrades the performance of completion. Meanwhile, existing favorable GNNs are kind of shallow model (no deeper than 3 or 4 layers) because of the over-fitting and over-smoothing problems~\cite{xu2018representation} caused by the increase of GCN layers, which inherently limits the expressive power of the GCNs.

To address the above issues, we propose a novel multi-graph representation learning framework with GANs to solve GMC problem termed GAN-GMC, which unifies a generator $G$ and a discriminator $D$. $G$ generating the missing entries conditioned on what is actually observed and a discriminator $D$ estimating the probability that a rating score came from the training rating matrix rather than $G$. Specifically, new rating scores generated by $G$ are viewed as ``fake'' samples to fool the discriminator $D$, while $D$ tries to detect whether the rating scores are from ground truth rating matrix or generated by $G$. With the competition between the two modules, both of them can alternately and iteratively boost their performance. Further, we develop a deep recursive inference module for both user and item graph representation learning in generator, which can avoid the over-fitting and over-smoothing problems caused by the increase of GCN layers.

<h1 align = "center">Overview</h1>

<div align="center">
    <img src="image/MAGCN_structure.jpg" width="100%" height ="100%" alt="MAGCN.jpg" />
</div>
<p align = 'center'>
<small> Figure 1. The overall structure of our MAGCN. </small>
</p>

To better understand the proposed model, we provide a summary of the algorithm flow:

- **Multi-GCN (unfold)**: The multi-view graph <img src="images/maths/G1.jpg" align="center" border="0" weight="24" height="16" alt="G*" /> with 5 nodes, n topologies and a feature matrix <img src="images/maths/X.jpg" align="center" border="0" weight="24" height="16" alt="X" />, is first expressed by the multi-GCN (unfold) block to obtain a multiview representation <img src="images/maths/tensorX.jpg" align="center" border="0" weight="24" height="16" alt="TensorX" />.

- **Multiview Attention**: Then a multiview attention is utilized to fuse the <img src="images/maths/X2.jpg" align="center" border="0" weight="24" height="16" alt="tensorX2" /> to a complete representation <img src="images/maths/tensorX2.jpg" align="center" border="0" weight="24" height="16" alt="X2" />.

- **Multi-GCN (merge)**: Finally, a multi-GCN (merge) block with softmax is introduced to obtain the final classification expressing matrix <img src="images/maths/X3.jpg" align="center" border="0" weight="24" height="16" alt="X3" />.

<h1 align = "center">Experiments</h1>

## Semi-Supervised Classification.

<p align = 'center'>
<small> Table 1. Semi-supervised Classification Accuracy (%). </small>
</p>

<div align="center">
    <img src="images/semi-results.jpg" width="70%" height ="70%" alt="semi-results.jpg" />
</div>

## Robustness Analysis.

To further demonstrate the advantage of our proposed method, we test the performance of MAGCN, GCN and GAT when dealing with some uncertainty issues in the node classification tasks. Here we only use Cora dataset, and consider two types of uncertainty issues: random topology attack (RTA) and low label rates (LLR), that can lead to potential perturbations and affect the classification performance.

### Random Topology Attack (RTA)

<div align="center">
    <img src="images/RTA.jpg" width="70%" height ="70%" alt="RTA.jpg" />
</div>
<p align = 'center'>
<small> Figure 2. Test performance comparison for GCN, GAT, and MAGCN on Cora with different levels of random topology attack. </small>
</p>

### Low Label Rates (LLR)

<div align="center">
    <img src="images/LLR.jpg" width="70%" height ="70%" alt="LLR.jpg" />
</div>
<p align = 'center'>
<small> Figure 3. Test performance comparison for GCN, GAT, and MAGCN on Cora with different low label rates. </small>
</p>

<h1 align = "center">Visualization and Complexity</h1>

## Visualization

To illustrate the effectiveness of the representations of different methods, a recognized visualization tool t-SNE is utilized. Compared with GCN, the distribution of the nodes representations in a same cluster is more concentrated. Meanwhile, different clusters are more separated.

<div align="center">
    <img src="images/visualization.jpg" width="100%" height ="100%" alt="visualization.jpg" />
</div>
<p align = 'center'>
<small> Figure 4. t-SNE visualization for the computed feature representations of a pre-trained model's first hidden layer on the Cora dataset: GCN (left) and our MAGCN (right). Node colors denote classes. </small>
</p>

## Complexity

- **GCN** [(Kipf & Welling, 2017)](https://arxiv.org/abs/1609.02907): <img src="images/maths/GCN-complexity.jpg" align="center" border="0" weight="24" height="16" alt="\mathcal{O}(|E|FC)" />
- **GAT** [(Veličković et al., 2018)](https://arxiv.org/abs/1710.10903): <img src="images/maths/GAT-complexity.jpg" align="center" border="0" weight="24" height="16" alt="\mathcal{O}(|V|FC + |E|C)" />
- **MAGCN**: <img src="images/maths/MAGCN-complexity.jpg" align="center" border="0" weight="24" height="16" alt="\mathcal{O}(n|E|FC + KC)" />

where <img src="images/maths/V.jpg" align="center" border="0" weight="24" height="16" alt="V" /> and <img src="images/maths/e.jpg" align="center" border="0" weight="24" height="16" alt="E" /> are the number of nodes and edges in the graph, respectively. F and C denote the dimensions of the input feature and output feature of a single layer. n denotes the number of the views, <img src="images/maths/Attention-complexity.jpg" align="center" border="0" weight="24" height="16" alt="\mathcal{O}(KC)" /> is the cost of computing multi-view attention and K denotes the neuron number of multilayer perceptron (MLP) in multi-view attention block. Although the introduction of multiple views multiplies the storage and parameter requirements by a factor of n compared with GCN, while the individual views’ computations are fully independent and can be parallelized. Overall, the computational complexity is on par with the baseline methods GCN and GAT.

<h1 align = "center">Applications</h1>

In the real-world graph-structured data, nodes have various roles or characteristics, and they have different types of correlations. Multiview graph learning/representation is of great importance in various domains. Here, we will provide a gentle introduction of possible applications of our proposed MAGCN (or its potential variants in both architecture/algorithm level). 


### Urban computing

The deployment of urban sensor networks is one of the most important progresses in urban digitization process. Recent advances
in sensor technology enables the collection of a large variety of datasets. Region-level prediction is a fundamental task in data-driven urban management. There are rich number of topics, including citizen flow prediction, traffic demand prediction, arrival time estimation and meteorology forecasting. Non-Euclidean structures exist in station-based prediction tasks, including bike-flow prediction, traffic volume prediction and point-based taxi demand prediction. 

The core issue for multi-modal machine learning is to build models that can process or relate information from multiple modalities, while multi-modal fusion is one of the most challenging problems in urban computing. Most existing works incorporate multi-modality
auxiliary data as handcrafted features in a straightforward manner, which is impossible to make full use of the multi-modal features. MAGCN and its future variants like Relational MAGCN, Multi-modal MAGCN, or Spatiotemporal MAGCN, etc., can provide an effective and favorable way to encode non-Euclidean pair-wise correlations among regions into multiple graphs (as shown in Figure 6) and then explicitly model these correlations using multi-graph convolution, which is technically significant for some region-level demand forecasting in unban computing.

<div align="center">
    <img src="images/urban.jpg" width="100%" height ="100%" alt="urban.jpg" />
</div>
<p align = 'center'>
<small> Figure 6. Different aspects of relationships among regions and the corresponding multi-view graph. </small>
</p>

<h1 align = "center">Conclusion</h1>

We propose in this paper a novel graph convolutional network model called MAGCN, allowing us to aggregate node features from different hops of neighbors using multi-view topology of the graph and attention mechanism. Theoretical analysis on the expressive power and flexibility is provided with rigorous mathematical proofs, showing a good potential of MAGCN over vanilla GCN model in producing a better node-level learning representation. Experimental results demonstrate that it yields results superior to the state of the art on the node classification task. Our work paves a way towards exploiting different adjacency matrices representing distinguished graph structure to build graph convolution.
