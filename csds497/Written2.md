# CSDS 497: AI: Statistical NLP  Written Assignment 2

Names and github IDs (if your github ID is not your name or Case ID):
    
Zhengyu Fang, fangzy96;      
Tianyi Li, ShPhoebus;       
Jiasen Zhang, Jiasen-Zhang;      
Yao Fu, ClarkFu007     

 
7.	Assume we have a vocabulary of size $V$. Let $N$ be the number of times we observe the context string $C=w_1,…,w_{n-1}$. Let $r$ be the number of times we observe the token $w_n$ after $C$ and let $N_0$ be the number of tokens we never observe after this context. Absolute discounting defines $Pr(w_n | C)=(r−\delta)/N$ if $r > 0$, for some $\delta > 0$. Assuming the remaining probability is distributed uniformly over unseen tokens, determine the probability needed to be assigned to each unseen token so that the result is a valid probability distribution. (10 points)

Answer: 
      
Known $V$ is the size of vocabulary, and $N_0$ is the number of unseen tokens after this context. And the number of seen tokens equals $V - N_{0}$. Known absolute discounting $Pr(w_n | C)=\frac{(r−\delta)}{N}$ if $r > 0$. If w/o smoothing, the $Pr(w_n | C)=\frac{r}{N}$, and if w/ smoothing we'd like to consider the unseen tokens part, which means we will assign probability for unseen tokens. Assuming the remaining probability is distributed uniformly over unseen tokens, and $\frac{\delta}{N}$ is the probability we assigned for each unseen token. Thus the remaining probability equals $\frac{\delta/N \times (V-N_{0})}{N_0}=\frac{\delta \times (V-N_{0})}{NN_0}$.     
Finally, we get:     
$Pr(w_n | C)=\frac{(r−\delta)}{N}$, if $r > 0$    
$Pr(w_n | C)=\frac{\delta \times (V-N_{0})}{NN_0}$, otherwise    

8.	Assume the same setting as above. Linear discounting defines $Pr(w_n | C)=(1−\alpha)r/N$ if $r > 0$, for some $\alpha > 0$. Assuming the remaining probability is distributed uniformly over unseen tokens, determine the probability needed to be assigned to each unseen token so that the result is a valid probability distribution. (10 points)

Answer:

Known Linear discounting defines $Pr(w_n | C)=(1−\alpha)r/N$, if $r > 0$, for some $\alpha > 0$. So the remaining probability of $w_n$ is $\frac{\alpha r}{N}$, if $r>0$, so the overall remaining probability is $\displaystyle \sum_{r = 1}{\frac{\alpha r}{N}}$
, if $r>0$. Since remaining probability is distributed uniformly over unseen tokens, the probability needed to be assigned to each unseen token: $Pr(w_u | C)=\displaystyle \frac{\sum_{r = 1}{\frac{\alpha r}{N}}}{N_0}$, since $\displaystyle \sum_{r = 1}{r}=N$, so $Pr(w_u | C)=\displaystyle \frac{\sum_{r = 1}{\frac{\alpha r}{N}}}{N_0}=\frac{\alpha}{N_0}$ ,if $r>0$, for some $\alpha > 0$.


9.	Assume we have a vocabulary of size $V$, and observe a corpus of $N$ tokens with a vocabulary $V^\prime$. Let $N_r$ be the number of tokens in the corpus that appear $r$ times. The Good-Turing adjusted count of a token occurring $r > 0$ times is $GT(r)=(r+1)N_{r+1}/N_r$ . Prove that if we assume all unseen tokens have equal probability, the probability assigned to any unseen token must be $N_1/NN_0$. (20 points)

Answer:

The probability of seen tokens is $\sum\limits_{r=1}^{+\infty}\frac{GT(r).N_r}{N}$,  
which is equal to $\sum\limits_{r=1}^{+\infty}\frac{(r+1).N_{r+1}}{N}$
due to $GT(r)=\frac{(r+1).N_{r+1}}{N_r}$.

Since we know that $N_1+2\times N_2+3\times N_3+...+(+\infty)\times N_{+\infty}=N$,    
the probability of seen tokens can be further reduced to $\frac{N-N_1}{N} = 1 - \frac{N_1}{N}$.   
Therefore, P(unseen tokens) = 1 - P(seen tokens) = $\frac{N_1}{N}$.        
Moreover, due to that all unseen tokens have equal probability,  
the probability assigned to any unseen token should be $\frac{N_1}{N.N_0}$.


10.	Consider the optimization problem $max_{(\lambda, x)} \sum_i \lambda_if_i(x),  \sum_i \lambda_i=1, \lambda_i \geq 0, \(i=1,…,n\)$. Show that $\lambda^\* = (0,…,0,1,0,…0)$ is always a feasible optimal solution regardless of the optimal $x^*$. (20 points)

Answer:

The given optimization problem is the dual problem of the primal problem. To find the primal problem we compute the generalized Larangian:  
$L(\lambda, x, u, v) = \sum_i \lambda_if_i(x) + \sum_i \sum_k u_k (1-\lambda_i) - \sum_i v_i \lambda_i \quad , \quad u_i \ge 0, v_i=0$  
$\frac{\partial L}{\partial\lambda_i} = f_i - \sum_k u_k - v_i = 0  \quad\rightarrow\quad f_i - \sum_k u_k = v_i = 0$  
$\rightarrow\quad  \max_{\lambda_i} L =  \sum_i u_i$  
So the primal problem is: $\min_u \sum_i u_i \quad , \quad \sum_i u_i = f_k \quad , \quad u_i \ge 0 \quad , \quad k = 1,2,3,...$  
    
Obviously the solution of the primal problem is $\min_k f_k \quad , \quad u^* = (0,...,1,...,0)$ where k-th enrty of $u^\*$ is one. Because the solutions of the primal and dual problem are equal. For the given problem we also have $\lambda^\* = (0,…,0,1,0,…0)$ and it's independent of $x^*$.
     

11.	Write down the dynamic programming recursion for the backward algorithm in HMMs. (10 points)

Answer: 
    
Initialization: $r_{k} (n)=1$, $k$ is any observation, $n$ is the number of state.     
In backward algorithm recursion we need to compute: $r_{k} (i)=Pr(o_{i+1:n}|s_{i}=k)$.    
Known $Pr(s_{i}^{\*}=k|s_{i+1}=p)$ is the transition, and $Pr(o_{i+1}|s_{i+1}^{\*}=p)$ is the emitting observation $i+1$.    
The recursion for backward is: $r_{k} (i)=max_{p} r_{p} (i+1) Pr(s_{i}^{\*}=k|s_{i+1}=p) Pr(o_{i+1}|s_{i+1}^{\*}=p)$.




