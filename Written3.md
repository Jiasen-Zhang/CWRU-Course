# CSDS 497: AI: Statistical NLP  Written Assignment 3

Names and github IDs (if your github ID is not your name or Case ID):
 
12. Prove that the state sequence $s^\*$ returned by the Viterbi algorithm for some observation $o$ from an HMM indeed satisfies $s^\*=\arg\max_s Pr(s|o)$. (10 points)

Answer:   
Suppose the most likely state sequence is $(s_1^\*, ..., s_n^\*) = (k_1, ..., k_n)$,  
We know that $\gamma_{k_i} (i) = \max_p \gamma_p (i-1) Pr(s_i^\*=k_i |S_{i-1}^\* = p) Pr(o_i|s_i^\*=k_i) = \gamma_{k} (i-1) Pr(s_i^\*= k_i |S_{i-1}^\* = p) Pr(o_i|s_i^\*=k_i) $,




13.	Prof. Methodical wants their experiments to be replicable, so they suggest that instead of initializing their HMM parameters randomly as the Baum-Welch algorithm wants, they should start by initializing all parameters to zero. Explain to Prof. Methodical why this might not be a good idea. (10 points)

Answer: 

14.	Suppose we have observations over a set $X=$ \{ $x_0, x_1,\ldots, x_n$ \}, such that the number of observations is $N$ and the $i^{th}$ element is observed $n_i$ times. Find a probability distribution $p^\*$ over $X$ that solves $p^\*=\arg\max_p H(p)$, where $H$ is the entropy function. Explain the significance of $p^\*$ in the context of conditional random fields (CRFs). (20 points)

Answer: 

16.	Suppose we see a set of $n$ observations $(S,O)=$\{ $(s_1,o_1),\ldots ,(s_n, o_n)$ \}. We learn a CRF $P(s|o)$ with features $f_i(s, o)$ and parameters $\lambda_i$ by maximizing log conditional likelihood. Show that the optimal solution $\lambda^*$ satisfies $\sum_{(s,o)} f_i(s, o)= \sum_{(s,o)} \sum_{s'} P(s'|o) f_i(s', o)$.  (20 points)

Answer: 

17.	Give an example of an English sentence which could be difficult to parse with an HMM. Explain in your own words why it could be difficult. (10 points)

Answer: 

18.	Prove that the union of two regular languages is regular. (10 points)

Answer:


19.	Prove that the intersection of two regular languages is regular. (10 points)

Answer:

20.	Prove that the language $L=$\{ $A^nB^{n-1}$ likes milk \} is not regular, where $A$ and $B$ are sets of terminals. (10 points)

Answer: 

21.	Explain intuitively why a language $L=$\{ $a^nb^nc^n$ \}  cannot be context free. (Hint: A pushdown automaton to recognize a context free language uses an (infinite) stack as a memory.) (10 points)

Answer: 
