# CSDS 497: AI: Statistical NLP  Written Assignment 2

Names and github IDs (if your github ID is not your name or Case ID):
 
7.	Assume we have a vocabulary of size $V$. Let $N$ be the number of times we observe the context string $C=w_1,…,w_{n-1}$. Let $r$ be the number of times we observe the token $w_n$ after $C$ and let $N_0$ be the number of tokens we never observe after this context. Absolute discounting defines $Pr(w_n | C)=(r−\delta)/N$ if $r > 0$, for some $\delta > 0$. Assuming the remaining probability is distributed uniformly over unseen tokens, determine the probability needed to be assigned to each unseen token so that the result is a valid probability distribution. (10 points)

Answer:

8.	Assume the same setting as above. Linear discounting defines $Pr(w_n | C)=(1−\alpha)r/N$ if $r > 0$, for some $\alpha > 0$. Assuming the remaining probability is distributed uniformly over unseen tokens, determine the probability needed to be assigned to each unseen token so that the result is a valid probability distribution. (10 points)

Answer:

9.	Assume we have a vocabulary of size $V$, and observe a corpus of $N$ tokens with a vocabulary $V^\prime$. Let $N_r$ be the number of tokens in the corpus that appear $r$ times. The Good-Turing adjusted count of a token occurring $r > 0$ times is $GT(r)=(r+1)N_{r+1}/N_r$ . Prove that if we assume all unseen tokens have equal probability, the probability assigned to any unseen token must be $N_1/NN_0$. (20 points)

Answer:

10.	Consider the optimization problem $max_{(\lambda, x)} \sum_i \lambda_if_i(x),  \sum_i \lambda_i=1, \lambda_i \geq 0, \(i=1,…,n\)$. Show that $\lambda^\* = (0,…,0,1,0,…0)$ is always a feasible optimal solution regardless of the optimal $x^*$. (20 points)

Answer:

The given optimization problem is the dual problem of the primal problem: $min_{\lambda} \sum_i \lambda_i$ ,  $\sum_k \lambda_k=f_i(x), \lambda_i \geq 0, \(i=1,…,n\)$. Obviously, the solution of the primal problem is $\min_k = f_k(x)$ and $\lambda^\* = (0,…,0,1,0,…0)$ where the k-th enrty of $\lambda^\* $ is one. According to the theory of primal-dual method, the objective function satisfies KKT conditions so the solutions of the primal and dual problem are equal. So for the given problem we also have $\lambda^\* = (0,…,0,1,0,…0)$ and it's independent of $x^\*$.

11.	Write down the dynamic programming recursion for the backward algorithm in HMMs. (10 points)

Answer: 
