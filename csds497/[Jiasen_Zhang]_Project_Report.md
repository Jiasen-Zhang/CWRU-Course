Learning-Free Integer Programming Summarizer (LFIP-SUM)    
----------------------------------------------------------------------------------------------------------------------     
Jiasen Zhang
----------------------------------------------------------------------------------------------------------------------   


----------------------------------------------------------------------------------------------------------------------  
1 Survey
----------------------------------------------------------------------------------------------------------------------  
1.1 Overview

The goal of document summarization is to describe the text with a much shorter one while preserving important information. There are three kinds of summerization: Extractive summerization select some important sentences of the raw text. abstractive summerization generate a new short summary with novel words like human. Hybrid summerization combines the two method by generating abstractive summary for some salient sentences in the text. 

Extractive summarization methods are mostly unsupervised. Most of them are graph-based and rank the sentences by assigning scores according to the graph. Each vertex of the graph is a sentence and the edge weights are represented by the similarity between two sentences. Traditional methods compute the similarity with word frequency like famous TextRank. Recently, many new extractive summarization methods make use of pre-trained word embeddings like BERT to represent the sentences with a set of vectors. Abstractive and hybrid summarization methods are usually based on neural network such as recurrent neural network (RNN) and transformer (They are also available for extractive summarization). For extractive summarization, they work as classifiers to select salient sentences. For abstractive summarization, their encoding-decoding architecture can learn to generate numerical representations of the words. RNN has the disadvantage that it cannot capture lone range dependency across the long text. Recently, transformer methods start to draw much attention for their better performance than RNN not only in summarization but also in other NLP topics. They usually use pre-trained BERT model to improve efficiency.

1.2 Selected papers

In my project I mainly implement Learning Free Integer Programming Summarizer (LFIP-SUM)[1] proposed in 2021, which is an unsupervised extractive method using pre-trained word embeddings. It first represent the sentences with continuous vectors with the embedding models. Then, to capture the latent meaning of the sentences, the deep representation of the sentences are obtained using self-attention method leveraging positional encoding. It use principal component analysis (PCA) to reduce the dimension (number of sentence) of deep representation to obtain condensed sentences containing more information. The importance score of each sentence is computed as the similarity of the sentence and the condensed sentences. Finally, the final score of each sentence is computed with its importance score the cosine similarities of the sentences. The details of LFIP-SUM will be introduced in the next section.

During searching papers, I focused on unsupervised extractive methods using pre-trained word embeddings and found some other interesting methods. Hierarchical and Positional Ranking model (HipoRank)[2] is a graph-based model that also consider the sentences in different sections in the long document. So the graph is hierarchical and the edges can represent inter-sectional connections (sentence-sentence edges) and intra-sectional connections (section-sentence edges). The sentence-sentence weight is just cosine similarity. The section-sentence weight is a little different in which the representation vector of section is the average of representations of all sentences in the section. The centrality score of each sentence is a weighted sum of sentence-sentence centrality and section-sentence centrality. This method is not suitable for all documents because it requires us to divide the document into sections. But some of the details are effective and can be applied to other methods. For example, it defines a section boundary function assigning larger weights to the sentences near a document’s boundaries. I will test its effect in my extension section.

Another of my extension is from an affinity graph based approach[3]. The original cosine matrix only consider the direct link between two sentences and ignore similarities more than two steps. In the affinity graph based approach, it applies the diffusion process on the graph to acquire the implicit semantic relationship between sentences. Although LFIP-SUM is not graph-based, it also makes use of cosine similarity to represent similarities between two sentences so it's also reasonable to apply diffusion process. In fact, diffusion process can actually improve performance in my extension. For multi-document summarization, it computes both intra-document and inter-document matrices and the new affinity matrix is the weighted sum of them. The information richness of sentences are computed with the new affinity matrix and like other graph-based methods, they are used to compute rank scores iteratively. Although both the affinity graph based approach[3] and LFIP-SUM use pre-trained word embedding and compute cosine similarity, LFIP-SUM considers similarity with deep representation of sentences so I think it make use of information of word embedding more thoroughly.


----------------------------------------------------------------------------------------------------------------------  
2 LFIP-SUM Method
----------------------------------------------------------------------------------------------------------------------  
This section describes the details of LFIP-SUM. Suppose the documentation is $D=(s_1, \cdots, s_n)$ where $s_i$ is the i-th sentence and n is number of sentence. With the pre-trained word embedding model we can get the basic representation of document $D_{basic}=(sv_1, \cdots, sv_n)$. It's a $d\times n$ matrix where $d$ is the dimension of representation. $sv_i$ is the sentence representation vector of $s_i$. In this paper they test different word embedding models and BERT model is the best. So I use BERT model in this project.

The matrix $D_{basic}$ contains the intrinsic meaning of the sentences but lacks positional information. Therefore we apply positional encoding, which is used in transformer. The positional encoding matrix PE is:

$$ PE(pos, 2i) = \sin(pos/10000^{2i/d}) $$

$$ PE(pos, 2i+1) = \cos(pos/10000^{2i/d}) $$

where $pos$ is the position of sentence, $i$ is representation dimension. Then the final input embedding matrix $D_{emb}$ is calculated as $D_{emb}=D_{basic}+PE$. To obtain deeper representation, we apply scaled dot product self-attention to the embedding matrix which doesn't need training parameters. The self-attention is:

$$ Attention(Q,K,V) =  softmax(QK^T / \sqrt{d_k})V$$

where $d_k$ is the dimension of Q and K. For self-attention, the matrices Q, K and V are all $D_{emb}$ so we can get the deep representation $D_{deep} = Attention(D_{emb}, D_{emb}, D_{emb})$.

The deep representation has the same size as $D_{basic}$ and it contains the deep representation of each sentence:

$$D_{basic}=(svdeep_1, \cdots, svdeep_n)$$

Then we use principal component analysis (PCA) to reduce the dimension of $D_{deep}$. The generated principal components (PCs) are linear combination of  the original variables. As a result, the projection of the data on the first axis will have the largest variance. To implement PCA, first we need to decide the dimension k of output of PCA. For efficiency, I define k as half of the number of sentence. We perform SVD: $D_{deep} = USV^T$, then the principal components are the matrix multiplication of $D_{deep}$ and the first k columns of $V$. We define the principal component as principal sentences (PS) because each of them contains information of more than sentences. The output of PCA can be written as $(PS_1, \cdots, PS_k)$.

Because PS condenses more intrinsic information, if a sentence in the text has larger cosine similarity with a PS, the sentence should also contains more information and should be more important. Therefore we can define the importance score of each sentence as:

$$ imp(s_i) = \sum_k \cos(svdeep_i, PS_k)$$

Besides, we also need to compute the cosine similarities of the deep sentence representation vectors to evaluate the importance of a sentence:

$$ sim(s_i, s_j) = \cos(svdeep_i, svdeep_j)$$

LFIP-SUM is based on Integer Linear Programming (ILP) method[4]. Assuming the sentences to be extracted should have high importance score and low redundancy (similarities), it generates the summary by maximizing the following formula:

$$ \max_{x,y} \sum_i imp(s_i)x_i - \sum_j \sum^n_{j=i+1} sim(s_i,s_j) y_{i,j} \quad s.t. \sum_i l_i x_i \leq Lmax  \quad  i=1, \cdots, n$$

$l_i$ is the length of sentence $s_i$. $Lmax$ is the maximum length of summary. $x_i$ is a binary variable indicating whether $s_i$ is included in summary. $y_{i,j}$ is binary variable indicating whether both $s_i$ and $s_j$ are included in summary. The method in ILP is really computationally expensive. Instead, LFIP-SUM performs sentence pruning to remove sentences that are not important. In practice, we define the pruning score of sentence $s_i$ as:

$$ pr-score(s_i) = \frac{imp(s_i)}{\frac{1}{n-1} \sum_{j \neq i}sim(s_i,s_j)} $$

Given a maximum sentence number L, the sentences with top L pruning scores will be included in summary. In my project, I set the summary size as the number of sentences of the text divided by 2.5 (rounded value). 


----------------------------------------------------------------------------------------------------------------------  
3 Research
----------------------------------------------------------------------------------------------------------------------  
3.1 Diffusion process on graph

As we mentioned, cosine matrix represents the similarities between to sentences, in other words it only consider the direct links between sentences. However, some sentences may have indirect links with each other, some paths with more than two steps can also reflect semantic relationships. So the affinity graph based approach[3] apply diffusion process on the graph. Obviously, such diffusion process can be applied into all methods using cosine similarity other than graph-based models, such as LFIP-SUM. Although considering all possible paths between any two nodes of a graph can be really computationally expensive, it has been proved that we can compute the affinity between any two nodes without checking all possible paths.

This idea comes from diffusion kernels on graphs[5], which was generalized from simulating heat diffusion. As shown in [5], a continuous diffusion process represented by a matrix M satisfies heat equation $ dK/dt = MK$ whose solution is $K=e^M$. Thus, the diffusion process of graph can be computed with exponentiation operation on cosine similarity matrix. Generally, we can define a decay factor $\gamma$ representing the speed of diffusion. Then the diffusion process can be approximated by power series:

$$M' = \sum^\infty_{t=1} \gamma^{t-1} M^t $$

where $M$ is cosine similarity matrix. In practice, $M$ is first normalized so that the sum of each row is 1. And the value of $t$ is finite. In [3] $t$ is limited to 5 and in my project $t$ is limited to 10.

3.2 Sentence boundary function

In HipoRank[2], it's assumed that in a document the sentences near the boundary (start or end) are more important. Based on this idea, the authors proposed a sentence boundary function $db(i) = \min(x_i, a(n-x_i))$ where n is the number of sentences, $x_i$ is i-th sentence's position and $a$ is a parameter representing importance of the start or end of a document. The cosine matrix M is then modified as:

$$Mij = \lambda_1 * sim(s_i, s_j) \quad if \quad db(i) \ge db(j) $$

$$Mij = \lambda_2 * sim(s_i, s_j) \quad if \quad db(i) < db(j) $$

where $\lambda_1<\lambda_2$ so that the weight of edge between i and j is larger if i is closer to the boundary. in my project I set $a=1$, $\lambda_1=0.3$ and $\lambda_2=0.7$.

In terms of graph, this operation assigns asymmetric edge weights for the graph. For cosine similarity the edge weights are symmetric but one sentence can be more important than the other one when considering their discourse structures. Incorporating directionality in the graph can help prevent redundancy because similar sentences usually have different positions so their asymmetric weight are different. 


3.3 Multiple self-attention

Another extension is a simple but natural idea. LFIP-SUM make use of self-attention method. In transformer, there are often more than one self-attentions to get deeper representations and this idea is obviously reasonable for LFIP-SUM too. In my opinion, the deeper representations can contain more implicit semantic relationships and the corresponding principal components can contain more informations. In my experiment, using multiple self-attention can largely improve the performance.

----------------------------------------------------------------------------------------------------------------------  
4 Results, Analysis and Discussion
---------------------------------------------------------------------------------------------------------------------- 
4.1 Results      

The dataset of our group is a part of BBC News Summary dataset. It consists of 445 short articles and corresponding summary. Because my algorithm is time consuming, I just test one value for each factor and observe the effect. In my experiment, I implement original LFIP-SUM, LFIP-SUM with diffusion process with factor 0.9, LFIP-SUM with sentence boundary function with factor 1, LFIP-SUM with 3 self-attentions and LFIP-SUM with all the 3 extensions. The evaluation metrics are Rouge 1 F1, Rouge 2 F1 and Rouge L F1 and the results are shown below.       
![11](https://raw.githubusercontent.com/cwru-courses/csds497-f22-3/main/Project/jxz867%20Jiasen-Zhang/result.png?token=GHSAT0AAAAAAB2D5UVLPXOQ4DVUOYXJZIYKY4UGHRQ)

4.2 Discussion

According to the results, all the three extensions can improve the performance of original LFIP-SUM. Specifically, multiple self-attention has the strongest effect of improvement for all the three Rouge scores. I just add three self-attentions and if I use more the results can be even better. It reflects the greatness of invention of attention mechanism. Because there are many other unsupervised methods using pre-trained word embedding, self-attention can be used in all of them. For supervised methods using pre-trained word embedding, we can apply both self-attention and cross attention, whose key and value matrices are from ground truth. It shows that attention mechanism is not limited to deep learning methods

The boundary position function can also largely improve the performance and it also has the strongest effect of improving Rouge L F1 for this dataset. This extension makes an important assumption that the sentences near the boundary are more important. That's reasonable for most articles like scientific papers and news including our dataset. However, it may not be suitable for some datasets. For example, for some conversations people discuss with each other and then get the important idea. In this case the most important sentence can be at the center. 

The diffusion process can also improve all the performance, but has the weakest effect. But this extension prove that this diffusion process can be used in all extractive summarization methods with similarity matrix. The matrix needs not to be cosine matrix computed with word embedding. For example, in LexRank[6] the similarity matrix is computed with idf-modified-cosine similarity where idf means inverse document frequency, so the similarity matrix is computed from word frequency. I think diffusion process can also be applied to such kind of similarity matrix because it also represents the weights of edges of graph like cosine matrix. 

Finally I apply all the 3 extensions and get the best performance. According to my discussion, all of these three extensions can be widely used in many extractive methods and even abstractive methods. And they can be used to improve the details of many proposed summarization methods. It still needs further investigation.


----------------------------------------------------------------------------------------------------------------------  
Bibliography
---------------------------------------------------------------------------------------------------------------------- 
[1] Jang, M. and Kang, P., 2021. Learning-free unsupervised extractive summarization model. IEEE Access, 9, pp.14358-14368.      
[2] Dong, Y., Mircea, A. and Cheung, J.C., 2020. Discourse-aware unsupervised summarization of long scientific documents. arXiv preprint arXiv:2005.00513.      
[3] Wan, X. and Yang, J., 2006, June. Improved affinity graph based multi-document summarization. In Proceedings of the human language technology conference of the NAACL, Companion volume: Short papers (pp. 181-184).      
[4] McDonald, R., 2007, April. A study of global inference algorithms in multi-document summarization. In European Conference on Information Retrieval (pp. 557-564). Springer, Berlin, Heidelberg.      
[5] Kondor, R.I. and Lafferty, J., 2002, July. Diffusion kernels on graphs and other discrete structures. In Proceedings of the 19th international conference on machine learning (Vol. 2002, pp. 315-322).     
[6] Erkan, G. and Radev, D.R., 2004. Lexrank: Graph-based lexical centrality as salience in text summarization. Journal of artificial intelligence research, 22, pp.457-479.      
