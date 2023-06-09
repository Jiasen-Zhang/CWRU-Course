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
In transition matrix, the sum of frequency for transitting to next states should be 1, and the sum of frequency of emitting different words should be 1. So if we initialize all parameters to zero, it will not satisify such requirements. And in M-step of Baum-Welch Algorithm, we cannot get the transition distribution and emission distribution, because the numerator and denominator will be zero.

14.	Suppose we have observations over a set $X=$ \{ $x_0, x_1,\ldots, x_n$ \}, such that the number of observations is $N$ and the $i^{th}$ element is observed $n_i$ times. Find a probability distribution $p^\*$ over $X$ that solves $p^\*=\arg\max_p H(p)$, where $H$ is the entropy function. Explain the significance of $p^\*$ in the context of conditional random fields (CRFs). (20 points)

Answer:       
Suppose the probability of $x_i$ is $p_i$, then the entropy is $H(p) = -\sum_i p_i \log p_i$. There are two constraints: $\sum_i p_i=1$ and $\sum_i n_i p_i = N/(n+1)$. We can use the method of Lagrange Multipliers to find the maximum:
$$F =  -\sum_i p_i \log p_i + \lambda_1 (\sum_i p_i-1) + \lambda_2 (\sum_i n_i p_i - N/(n+1))$$
$$\frac{\partial F}{\partial p_i} =-(1+\log p_i) + \lambda_1 + \lambda_2 n_i \quad\rightarrow\quad p_i = e^{\lambda_1 -1} e^{\lambda_2 n_i} $$
We use the first constraint: 
$$\sum_i p_i = \sum_i e^{\lambda_1 -1} e^{\lambda_2 n_i} = 1  \quad\rightarrow\quad e^{\lambda_1 -1}= \frac{1}{\sum_k e^{\lambda_2 n_k}} \quad\rightarrow\quad p_i = \frac{e^{\lambda_2 n_i}}{\sum_k e^{\lambda_2 n_k}} $$
We use the second constraint to compute $\lambda_2$:
$$\sum_i n_i e^{\lambda_2 n_i} = N/(n+1) \sum_k e^{\lambda_2 n_k} \quad\rightarrow\quad \lambda_2$$
So the optimal probability distribution is:
$$p_i^\* = \frac{e^{\lambda_2 n_i}}{\sum_k e^{\lambda_2 n_k}} $$
where $\lambda_2$ is computed by the equation above. The distribution $p^\*$ is similar to that of CRF. Both of them are exponential distribution and that imples that the distribution of CRF also maximes the entropy. 



16.	Suppose we see a set of $n$ observations $(S,O)=$\{ $(s_1,o_1),\ldots ,(s_n, o_n)$ \}. We learn a CRF $P(s|o)$ with features $f_i(s, o)$ and parameters $\lambda_i$ by maximizing log conditional likelihood. Show that the optimal solution $\lambda^*$ satisfies $\sum_{(s,o)} f_i(s, o)= \sum_{(s,o)} \sum_{s'} P(s'|o) f_i(s', o)$.  (20 points)

Answer:        
For CRF, we have
$$P(s|o)= \frac{e^{\lambda_i f_i(s,o)}}{\sum_{s} e^{\lambda_i f_i(s,o)}} \qquad \lambda^\* = \arg\max_\lambda \sum_i \log P(s|o) = \arg\max_\lambda F(\lambda)$$
$$F(\lambda) = \sum_i ( \lambda_i f_i(s,o) - \log \sum_{s} e^{\lambda_i f_i(s,o)} ) $$
To find the optimal solution $\lambda^\*$ we need
$$\frac{\partial F}{\partial \lambda_i} = f_i(s,o) - \frac{\sum_{s} f_i(s,o) e^{\lambda_i f_i(s,o)}}{\sum_{s} e^{\lambda_i f_i(s,o)}}=0$$
$$f_i(s,o) = \sum_{s} f_i(s,o) ( \frac{e^{\lambda_i f_i(s,o)}}{\sum_{s} e^{\lambda_i f_i(s,o)}} ) = \sum_{s} f_i(s,o) P(s|o) $$
So we have
$$\sum_{(s,o)} f_i(s,o) = \sum_{(s,o)} \sum_{s'} f_i(s',o) P(s'|o) $$

17.	Give an example of an English sentence which could be difficult to parse with an HMM. Explain in your own words why it could be difficult. (10 points)

Answer:     
The HMM can handle on modeling short-term dependencies, but cannot capture long-range dependencies between distant elements. Like the first order HMM, we only consider previous one state. If increasing the order, the computation will be so huge, so it's difficult for HMM to deal with long dependencies sentences. For example: The dog  found on the wall wearing a red hat lost by a man who was living down the street next to my house was given a bread. In this example, "The dog" was related to "was given a bread", but it is too distant for HMM to capture.

18.	Prove that the union of two regular languages is regular. (10 points)

Answer:     
    Suppose that $L_1$ and $L_2$ are two regular languages, both of which can be recognized by a deterministic finite automaton (DFA).    
    We could represent DFA with a 5-element tuple: $M_1=(Q_1, \Sigma_1, \delta_1, q_1, F_1)$, $M_2=(Q_2, \Sigma_2, \delta_2, q_2, F_2)$, where $Q$ is a finite set of states, $\Sigma$ is a finite set of symbols, $\delta$ is a transition function, $q \in Q$ is a start state, and $F \subseteq Q$ is a set of accepting states.  
    We have $L(M_1)=L_1$ and $L(M_2)=L_2$.      
    Let's define a DFA: $M_3=(Q_1\times Q_2, \Sigma_3, \delta_3, (q_1, q_2), (F_1\times Q_2)\bigcup (F_2\times Q_1))$, where $\delta_3$ is defined as $\delta_3((q_1,q_2),\delta)=(\delta_1(q_1,\Sigma), \delta_2(q_2,\Sigma)) \forall \Sigma \in \Sigma_3$ and $\Sigma_3 = \Sigma_1 \bigcap \Sigma_2$.   
$M_3$ accepts if and only $M_1$ or $M_2$ accepts. Since $L(M_3)=L_1 \bigcup L_2$, we could know that $L_1 \bigcup L_2$ is regular.


19.	Prove that the intersection of two regular languages is regular. (10 points)

Answer:
    Suppose that $L_1$ and $L_2$ are two regular languages, both of which can be recognized by a deterministic finite automaton (DFA).    
    We could represent DFA with a 5-element tuple: $M_1=(Q_1, \Sigma_1, \delta_1, q_1, F_1)$, $M_2=(Q_2, \Sigma_2, \delta_2, q_2, F_2)$, where $Q$ is a finite set of states, $\Sigma$ is a finite set of symbols, $\delta$ is a transition function, $q \in Q$ is a start state, and $F \subseteq Q$ is a set of accepting states.  
    We have $L(M_1)=L_1$ and $L(M_2)=L_2$.      
    Let's define a DFA: $M_3=(Q_1\times Q_2, \Sigma_3, \delta_3, (q_1, q_2), F_1\times F_2)$, where $\delta_3$ is defined as $\delta_3((q_1,q_2),\delta)=(\delta_1(q_1,\Sigma), \delta_2(q_2,\Sigma)) \forall \Sigma \in \Sigma_3$ and $\Sigma_3 = \Sigma_1 \bigcap \Sigma_2$.      
$M_3$ accepts if and only both $M_1$ and $M_2$ accept. Since $L(M_3)=L_1 \bigcap L_2$, we could know that $L_1 \bigcap L_2$ is regular.


20.	Prove that the language $L=$\{ $A^nB^{n-1}$ likes milk \} is not regular, where $A$ and $B$ are sets of terminals. (10 points)

Answer:  
According to the pumping lemma, if $L$ is regular, then there are strings $x$, $y$ and $z$ such that $xy^n z\in L$, $\forall n\ge 0$. Suppose the number of A and B is n(A) and n(B), then for $xy^n z\in L$ we have n(A)=n(B)+1.       
If $y$ consists of $A$, then for $xy^{n+k} z$ with any $k>0$ we have n(A)>n(B)+1. So $xy^{n+k} z \notin L$.      
Similarly, if $y$ consists of $B$, then for $xy^{n+k} z$ with any $k>0$ we have n(A)<n(B)+1. So $xy^{n+k} z \notin L$.      
If $y$ consists of $A$ and $B$, then for $xy^{n+k} z$ with any $k>0$, there are some $B$ on the left of $A$. So $xy^{n+k} z \notin L$.     
Therefore, we cannot find strings that can be pumped in $L$. So $L$ is not regular.

21.	Explain intuitively why a language $L=$\{ $a^nb^nc^n$ \}  cannot be context free. (Hint: A pushdown automaton to recognize a context free language uses an (infinite) stack as a memory.) (10 points)

Answer:  
According to the pumping lemma, if $L$ is regular, then there are strings $u$, $v$, $w$, $x$, and $y$ such that $uv^nwx^ny\in L$, $\forall n\ge 0$. Suppose the number of A, B and C is n(A),(B) and n(C), then for $u v^n w x^n y\in L$ we have $n(A)=n(B)=n(C)$.    
If $vwx$ consists of $A$, then for $uv^{n+k} w x^{n+k}y$ with any $k>0$ we have $n(B)\textless n(A)$ and $n(C)\textless n(A)$. So $uv^{n+k} w x^{n+k}y \notin L$.      
If $vwx$ consists of $B$, then for $uv^{n+k} w x^{n+k}y$ with any $k>0$ we have $n(A)\textless n(B)$ and $n(C)\textless n(B)$. So $uv^{n+k} w x^{n+k}y \notin L$.     
If $vwx$ consists of $C$, then for $uv^{n+k} w x^{n+k}u$ with any $k>0$ we have $n(A)\textless n(C)$ and $n(B)\textless n(C)$. So $uv^{n+k} w x^{n+k}y \notin L$.  
If $vwx$ consists of $A$ and $B$, then for $uv^{n+k} w x^{n+k}y$ with any $k>0$, we have $n(C)\textless n(A)$ and $n(C)\textless n(B)$. So $uv^{n+k} w x^{n+k}y \notin L$.     
If $vwx$ consists of $B$ and $C$, then for $uv^{n+k} w x^{n+k}y$ with any $k>0$, we have $n(A)\textless n(B)$ and $n(A)\textless n(C)$. So $uv^{n+k} w x^{n+k}y \notin L$.       
If $vwx$ consists of $A$, $B$ and $C$, then for $uv^{n+k} w x^{n+k}y$ with any $k>0$,  we have $n(A)\textless n(B)$ and $n(C)\textless n(B)$. So $uv^{n+k} w x^{n+k}y \notin L$.  
Therefore, we cannot find strings that can be pumped in $L$. So, a language $L=\{a^{n} b^{n} c^{n}\}$ cannot be context free
