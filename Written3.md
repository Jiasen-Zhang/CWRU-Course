# CSDS 497: AI: Statistical NLP  Written Assignment 3

Names and github IDs (if your github ID is not your name or Case ID):
 
12. Prove that the state sequence $s^\*$ returned by the Viterbi algorithm for some observation $o$ from an HMM indeed satisfies $s^\*=\arg\max_s Pr(s|o)$. (10 points)

Answer:   
Suppose the most likely state sequence is $(s_1^\*, ..., s_n^\*) = (k_1, ..., k_n)$. We know that the Viterbi algorithm is:
$$\gamma_{k_i} (i) = \max_p \gamma_p (i-1) Pr(s_i^\*=k_i |s_{i-1}^\* = p) Pr(o_i|s_i^\*=k_i) $$
$$\rightarrow\quad \gamma_{k_i} (i) = \gamma_{k_{i-1}} (i-1) Pr(s_i^\*= k_i |s_{i-1}^\* = k_{i-1}) Pr(o_i|s_i^\*=k_i) $$
Similarly,
$$\gamma_{k_{i-1}} (i-1) = \gamma_{k_{i-2}} (i-2) Pr(s_{i-1}^\*= k_{i-1} |s_{i-2}^\* = k_{i-2}) Pr(o_{i-1}|s_{i-1}^\*=k_{i-1}) $$
$$\gamma_{k_1} (1) = \gamma_{START} (0) Pr(s_{1}^\*= k_{1}) Pr(o_1|s_1^\*=k_1) = Pr(s_{1}^\*= k_{1}) Pr(o_1|s_1^\*=k_1) \quad \gamma_{START} (0)=1$$
So we have:
$$\gamma_{END}(n) = Pr(s_{1}^\*= k_{1}) Pr(o_1|s_1^\*=k_1) \prod^n_{r=2} Pr(s_r^\*= k_r |s_{r-1}^\* = k_{r-1}) Pr(o_r|s_r^\*=k_r) $$
$$\rightarrow\quad \gamma_{END}(n) = Pr( (o_1,...o_n), (s_1^\*,...s_n^\*)) = Pr(o,s^\*) = \max_s Pr(o,s) = \max_s Pr(s|o)P(o) $$
Therefore, the state sequence returned by Viterbi algorithm satisfies $s^\*=\arg\max_s Pr(s|o)$.


13.	Prof. Methodical wants their experiments to be replicable, so they suggest that instead of initializing their HMM parameters randomly as the Baum-Welch algorithm wants, they should start by initializing all parameters to zero. Explain to Prof. Methodical why this might not be a good idea. (10 points)

Answer: 

14.	Suppose we have observations over a set $X=$ \{ $x_0, x_1,\ldots, x_n$ \}, such that the number of observations is $N$ and the $i^{th}$ element is observed $n_i$ times. Find a probability distribution $p^\*$ over $X$ that solves $p^\*=\arg\max_p H(p)$, where $H$ is the entropy function. Explain the significance of $p^\*$ in the context of conditional random fields (CRFs). (20 points)

Answer:     
Suppose the probability of $x_i$ is $p_i$, then the entropy is $H(p) = -\sum_i p_i \log p_i$. There are two constraints: $\sum_i p_i=1$ and $\sum_i n_i p_i = N/(n+1)$. We can use the method of Lagrange Multipliers to find the maximum:
$$F =  -\sum_i p_i \log p_i + \lambda_1 (\sum_i p_i-1) + \lambda_2 (\sum_i n_i p_i - N/(n+1))$$
$$\frac{\partial F}{\partial p_i} =-(1+\log p_i) + \lambda_1 + \lambda_2 n_i$$
$$p_i = e^{\lambda_1 -1} e^{\lambda_2 n_i} $$
We use the first constraint: 
$$\sum_i p_i = \sum_i e^{\lambda_1 -1} e^{\lambda_2 n_i} = 1  \quad\rightarrow\quad e^{\lambda_1 -1}= \frac{1}{\sum_k e^{\lambda_2 n_k}} \quad\rightarrow\quad p_i = \frac{e^{\lambda_2 n_i}}{\sum_k e^{\lambda_2 n_k}} $$
We use the second constraint to compute $\lambda_2$:
$$\sum_i n_i e^{\lambda_2 n_i} = N/(n+1) \sum_k e^{\lambda_2 n_k} \quad\rightarrow\quad \lambda_2$$
So the optimal probability distribution is:
$$p_i^\* = \frac{e^{\lambda_2 n_i}}{\sum_k e^{\lambda_2 n_k}} $$
where $\lambda_2$ is computed by the equation above.


16.	Suppose we see a set of $n$ observations $(S,O)=$\{ $(s_1,o_1),\ldots ,(s_n, o_n)$ \}. We learn a CRF $P(s|o)$ with features $f_i(s, o)$ and parameters $\lambda_i$ by maximizing log conditional likelihood. Show that the optimal solution $\lambda^*$ satisfies $\sum_{(s,o)} f_i(s, o)= \sum_{(s,o)} \sum_{s'} P(s'|o) f_i(s', o)$.  (20 points)

Answer:     
For CRF, we have
$$P(s|o)= \frac{e^{\lambda_i f_i(s,o)}}{\sum_{(s,o)} e^{\lambda_i f_i(s,o)}} \qquad \lambda^\* = \arg\max_\lambda \sum_i \log P(s|o) = \arg\max_\lambda F(\lambda)$$
$$F(\lambda) = \sum_i ( \lambda_i f_i(s,o) - \log \sum_{(s,o)} e^{\lambda_i f_i(s,o)} ) $$
$$\frac{\partial F}{\partial \lambda_i} = $$

17.	Give an example of an English sentence which could be difficult to parse with an HMM. Explain in your own words why it could be difficult. (10 points)

Answer:    
He tries to fire the fire to light the light so that he can walk the walk.     
I think that that that that that student wrote on blackboard is wrong.     
I know that she know that I know that he know that we know you.      


18.	Prove that the union of two regular languages is regular. (10 points)

Answer:


19.	Prove that the intersection of two regular languages is regular. (10 points)

Answer:

20.	Prove that the language $L=$\{ $A^nB^{n-1}$ likes milk \} is not regular, where $A$ and $B$ are sets of terminals. (10 points)

Answer:      
$L=$\{ $A A^{n-1}B^{n-1}$ likes milk \} = \{ $A A^{n-1}B^{n-1}$ likes milk \}

21.	Explain intuitively why a language $L=$\{ $a^nb^nc^n$ \}  cannot be context free. (Hint: A pushdown automaton to recognize a context free language uses an (infinite) stack as a memory.) (10 points)

Answer: 
