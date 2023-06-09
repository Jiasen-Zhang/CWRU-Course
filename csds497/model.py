
import preprocessing
import numpy as np
import torch
import copy


''' ------------------- LFIP-SUM Functions -------------------   '''
''' Position encoding matrix'''
def positional_encoding_matrix(num_sentence, embedding_dim):
    PE = np.zeros((embedding_dim, num_sentence))

    for i in range(embedding_dim):
        for pos in range(num_sentence):
            if i%2==1:
                PE[i, pos] = np.cos(pos / pow(10000, 2 * i / embedding_dim))
            else:
                PE[i, pos] = np.sin(pos / pow(10000, 2 * i / embedding_dim))

    return PE


''' Reduce the rank of matrix M to k'''
def PCA(M, k):
    (U, S, V) = torch.pca_lowrank(torch.Tensor(M), q=k)

    PC = np.matmul(M, V.numpy())

    return PC


''' Cosine similarity of two sentences'''
def cos_sim(v1, v2):
    sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    return sim


''' Compute sentence importance score'''
def sentence_importance_score(text_deep, PC):
    imp = []
    num_sentence = text_deep.shape[1]
    num_PC = PC.shape[1]

    for i in range(num_sentence):
        imp_i = 0
        for k in range(num_PC):
            if k!= i:
                imp_i += cos_sim(text_deep[:,i], PC[:,k])

        imp.append(imp_i)

    return imp


''' Cosine similarity matrix'''
def sim_matrix(text_deep, boundary = 0):
    num_sentence = text_deep.shape[1]
    cos_matrix = np.zeros((num_sentence,num_sentence))
    db = boundary_positional_function(num_sentence)

    if boundary != 0:
        lambda1 = 0.3
        lambda2 = 1 - lambda1
        for i in range(num_sentence):
            for j in range(i+1,num_sentence):
                if db[i] < db[j]:
                    cos_matrix[i, j] = lambda1 * cos_sim(text_deep[:,i], text_deep[:,j])
                else:
                    cos_matrix[i, j] = lambda2 * cos_sim(text_deep[:, i], text_deep[:, j])
            for j in range(i):
                cos_matrix[i, j] = cos_matrix[j,i]
    else:
        for i in range(num_sentence):
            for j in range(i+1,num_sentence):
                cos_matrix[i,j] = cos_sim(text_deep[:,i], text_deep[:,j])
            for j in range(i):
                cos_matrix[i, j] = cos_matrix[j,i]

    return cos_matrix
    # return np.array(diffusion_graph(cos_matrix))


''' Compute the score for each sentence'''
def prune_score(imp, cos_matrix):
    num_sentence = len(imp)
    pr_score = []

    for i in range(num_sentence):
        denom = np.sum(cos_matrix[i,:])/(num_sentence-1)
        pr_score.append(imp[i]/denom)

    return pr_score


''' ------------------- Extension Functions -------------------   '''
'''Compute textrank similarities of sentences'''
def compute_similarity(text):
    # text filtered or not filtered

    num_sentence = len(text)
    similarity = np.zeros((num_sentence, num_sentence))

    for i in range(num_sentence):
        for j in range(num_sentence):
            if j != i:
                w = len(set(text[i]) & set(text[j]))
                similarity[i,j]=w / (np.log(len(text[i])) + np.log(len(text[j])))
    return similarity


''' Sentence boundary function'''
def boundary_positional_function(num_sentence, a=1.0):
    db = np.zeros(num_sentence)
    for i in range(num_sentence):
        db[i] = np.min([i, a*(num_sentence-i)])
    # print(db)
    return db


'''Compute inverse document frequency'''
def compute_idf(text, word):
    num_sentence = len(text)
    n = 0

    for sentence in text:
        if word in sentence:
            n += 1

    return np.log(num_sentence/n)


'''Compute idf-modified-cosine of two sentences'''
def idf_modified_cosine(text, s1, s2):
    sum_1 = 0
    sum_2 = 0
    sum_12 = 0
    for word in s1:
        sum_1 += (s1.count(word) * compute_idf(text, word))**2
        if word in s2:
            sum_12 += s1.count(word) * s2.count(word) * compute_idf(text, word) * compute_idf(text, word)
    for word in s2:
        sum_2 += (s2.count(word) * compute_idf(text, word))**2

    return sum_12 / np.sqrt(sum_1*sum_2)


'''Compute eigenvector of M using power method, the corresponding eigenvalue is 1'''
def eigenvector(M, e=1e-5):
    N = M.shape[0]
    p = np.ones(N)/N
    max_iter = 100

    for i in range(max_iter):
        p_old = copy.copy(p)

        p = np.matmul( np.transpose(M), p)

        delt = np.linalg.norm(p-p_old)

        if delt<e:
            break
    return p


def normalize_matrix(M):
    n_row = M.shape[0]
    for i in range(n_row):
        M[i,:] /= np.sum(M[i,:])
    return M


''' Diffusion process '''
def diffusion_graph(M, gamma=0.9, T=5):
    M1 = np.mat(M)
    M1 = normalize_matrix(M1)
    M0 = 0
    for t in range(T):
        t0 = t+1
        M0 += (gamma**t) * (M1**t0)

    M0 = normalize_matrix(M0)
    return M0


''' Sort the sentences according to the scores and generate summary'''
def ranking(scores, text, summary_size):
    rank = np.argsort(scores)

    summary = ''
    for i in range(summary_size):
        idx = rank[-1-i]
        # idx = rank[i]
        # summary.append(text[idx])
        summary += text[idx] + '. '

    return summary


''' Main function of LFIP-SUM'''
def LFIP_SUM(filename, boundary=0, diffusion=0, n_att=1):

    text = preprocessing.load_text(filename)
    # text_filtered = preprocessing.synatic_filter(text_tokenized_stem)
    # print(text)

    text_embedding = np.transpose(preprocessing.bert_embedding(text))
    embedding_dim = text_embedding.shape[0]
    num_sentence = text_embedding.shape[1]

    summary_size = 2*num_sentence//5
    # summary_size = 3

    # positional_encoding
    PE = positional_encoding_matrix(num_sentence, embedding_dim)
    text_emb = text_embedding + PE

    # apply self-attention to get deep representation
    for i in range(n_att):
        text_deep = np.matmul(text_emb, np.transpose(text_emb))/np.sqrt(embedding_dim)
        text_deep = torch.softmax(torch.Tensor(text_deep), dim=0).numpy()
        text_deep = np.matmul(text_deep, text_emb)
        text_emb = copy.copy(text_deep)

    PC = PCA(text_deep, k=num_sentence//2)

    imp = sentence_importance_score(text_deep, PC)

    cos_matrix = sim_matrix(text_deep, boundary = boundary)
    if diffusion != 0:
        cos_matrix = diffusion_graph(cos_matrix, gamma=diffusion)

    pr_score = prune_score(imp, cos_matrix)

    summary = ranking(pr_score, text, summary_size = summary_size)

    return summary


