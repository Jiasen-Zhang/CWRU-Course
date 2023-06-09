# CSDS 497: AI: Statistical NLP  Written Assignment 1

Names and github IDs (if your github ID is not your name or Case ID): 
    
Zhengyu Fang, fangzy96;      
Tianyi Li, ShPhoebus;       
Jiasen Zhang, Jiasen-Zhang;      
Yao Fu, ClarkFu007     

# 1. Is it possible for an automated system to "understand natural language"? Explain your answer in your own words. (10 points)

ANSWER:
---
We do not think an automated system can truly “understand” human’s “natural language. Understanding and generating language is an “AI complete” problem which would not be solved by a simple specific algorithm. Actually “Understanding” is a very complicated behavior for humans, it is hard to define how an automated system can “understand”. In fact, when people feed information into an automated system, it only needs to respond correctly and logically. For example in class, it is not necessary for a submarine to swim, because the submarine has its own way to move underwater.   
    
Some people designed a lot of tasks to test whether machines understood. For instance, they designed question answering systems, to make the machines trained and then answer some non-direct questions. Moreover, the machines are required to impute the missing blanks in a sentence. But even if the designed machines could show a good performance for those tasks, we still cannot say those machines truly “understand” because it is possible that they just show the correct response to people’s inputs.
    
But on the other hand, we also think of another possibility. First we ask how to define 'Understanding'? Just like the intentionality we discuss in the third question later. Here we can have another bold assumption, that is, 'understanding' comes from ‘thought’, and ‘thought’ is the content itself, or the content can be the information of the content. Philosopher Jiddu Krishnamurti made a similar claim, that is, "consciousness is the content itself". For natural language, there is a possibility that the existing language data can be used as content information, so as to develop the possibility of thinking and understanding. For example, AI drawing has become popular recently. The information of natural language does allow the machine to generate a certain content, and the vivid image drawn by the input text shows its expression of the content.


# 2. Read Turing's paper, "Computing Machinery and Intelligence," (https://academic.oup.com/mind/article/LIX/236/433/986238) and write a summary (~350 words) of his arguments for the possibility and existence of artificial intelligence. Don't just restate what he said, explain in your own words. (20 points)

ANSWER: 
---
According to Turing’s paper “Computing machinery and intelligence”, starting from the question “Can machines think?” Turing decomposed the question and explained his ideas about the possibility and existence of artificial intelligence. He also responded to various objections reasonably.  
    
He first claimed it is dangerous to define “machines” and “think” as commonly used and he refined the question by coming up with the imitation game, which made the question more specific and it is possible to think whether it can be solved. In order to relate with the machines designed by himself, Turing restricted the machine as an “electronic computer” or “digital computer”. Then he introduced three parts of the digital computer: store, executive unit and control which are already similar to the prototype of the modern computer. Based on the rules of the digital computer, it is possible to create a machine to mimic human thinking. Turing even identified a point humans and machines have in common: both digital computers and humans’ neural activities are triggered by electronic signals.  
    
To further refine the question “Can machines think?” Turing introduced the concept of “discrete state machines” that their final situation is decided by the input signals. It is possible to predict the future situation for such machines. And the digital computer can be classified to “discrete state machines”, then the central question can be further refined to “Can a particular digital computer, with adequate storage, suitably increasing its speed of action, and appropriate programming, perform well in imitation games?”  
    
In fact, there were a lot of contrary views on the main question. For example, the Mathematical Objection contended that there are limitations to the powers of discrete state machines, but although there are limitations, it has not been demonstrated that the limited capabilities that machines can achieve are limited to the realization of artificial intelligence. Similarly, Turing refuted all other objections with logical reasons.  
     
Turing first ridiculed that he did not have too many positive arguments, otherwise he would not rely on refuting others as his arguments. But those arguments also strongly prove the viability of Turing's views. For a learning machine, Turing pointed out that instead of simulating a complex adult brain, it is better to stimulate a new-born baby's brain. So our problem becomes a children's program with an educational process. Then he pointed out the important factor of a learning machine: Educators do not understand the internal structure of its implementation, but can predict the behavior of the machine to some extent. We can see something related to genetic algorithms and machine learning from here.  
     
In short, Turing starts from the main question “Can machines think?” He gradually refined the question and introduced his designed machines. Moreover, he refuted all opposing views one by one, which also support his points. Finally, he provided the famous motto: We can only see a short distance ahead, but we can see plenty there that needs to be done.
      

# 3. Research Searle's Chinese Room argument. Write a short summary of the argument in your own words and discuss potential objections to the argument in the literature (~350 words). Cite all references appropriately. (20 points)

ANSWER:
---
Short summary:
    
Searle believes that computers or AI don’t have real intelligence and understanding of words. Just like in the Chinese Room, the speaker does not really understand Chinese, but with the help of Chinese translation reference books, he can reply to fluent Chinese without understanding Chinese while outsiders have no way of identifying whether he really understands Chinese or not. The same is true when placed on a computer. Computers and other artificial intelligence do not need to be able to really understand and think about ideas or words, but they can analyze and process information through set rules and procedures, and output reasonable feedback to give an intelligent impression. Therefore, Searle argues that the "Turing Test" is insufficient, a thought experiment that highlights the fact that computers use only syntactic rules to manipulate strings of symbols, but know nothing about meaning or semantics.
    
Objections to the argument:
    
Searle responded to at least six different rebuttals to the Chinese Room experiment, here we focus on two important objections.
In System Reply, a very intuitive idea is that although the people in the room do not understand Chinese, and the books and cards in the room do not understand Chinese, this room, when being considered as a whole, seems to understand Chinese. Just like the human brain cells themselves are not conscious, but the brain as a collection of those cells is conscious. So, System Reply is that the room as a system has a certain state of mind in general. Searle responded by another thought experiment. Imagine the person in the room having been in the room long enough to have memorized both the rule book and the shape of Chinese characters. Then we opened the door and let the person out of the room. Now, when she walks on the street, she herself has become a walking Chinese room. The rest of the room, the rule book, had been internalized by her. Searle calls this scenario the Internalized Room. At this point, she can still respond fluently if someone passes her Chinese texts, but she doesn't know the real content of any of these texts. Searle therefore pointed out that the Chinese room and the program behind it that passed the Turing test are only a simulation of the mental state of "understanding Chinese", rather than a real understanding of Chinese. True understanding of language, and even the existence of any state of mind, requires the thinking subject to know exactly what is in the mind. This is philosophically called Intentionality.
In Robot Reply, some people believe that intentionality can be achieved entirely by computers and machines. Since this room as a system cannot connect Chinese characters with their corresponding objects, we can find a way to establish this connection. Connect this room with a series of sensors, such as cameras, microphones, haptics, taste sensors, etc., and a series of motion processors, in short, turn this room into a robot brain. Searle said that the signal received from the robot's sensor in the house may not be a direct image, sound, etc., but a string of digitized data. For example, when the robot's "eyes", that is, the camera, sees a flower, it does not mean that there is a screen in the room that reflects the small flower, but the camera collects a series of data and transmits it to the room by our robot. At this time, the robot still didn't know that his operation just now was about a flower. Intentionality still seems to be lacking, so true understanding still doesn't seem to exist. From my point of view, I am not very satisfied with Searle's response here. It seems that Searle is just repeating that some specific intentionality can only be given by the physiology of the human brain, but he does not provide a deeper reason for this assertion itself. I think the point is what intentionality and understanding are and how to define it precisely. When we have a precise definition, we can judge whether there is a real understanding.
    
References:                          
[1] Searle, John. R. (1980) Minds, brains, and programs. Behavioral and Brain Sciences 3 (3): 417-457.    
[2] Cole, David, "The Chinese Room Argument", The Stanford Encyclopedia of Philosophy (Winter 2020 Edition), Edward N. Zalta (ed.), URL = <https://plato.stanford.edu/archives/win2020/entries/chinese-room/>.
    


# 4. Research one AI system that can generate natural language, such as ELIZA, SHRDLU, WATSON, etc and write a summary (~350-700 words) describing how the system works and its strengths and weaknesses. Cite all references appropriately. (20 points)

ANSWER:
---
We choose ELIZA[1] and here is the summary. It's an early NLP program which makes conversation between human and computer possible. With some certain transformation rules, it can analyze input sentences and yield some reasonable outputs.  
   
At first the input text is scanned and analyzed to identify the keywords according to a dictionary of keywords. The keyword dictionary is constructed from a given ELIZA script, which consists of a set of keywords and their rule list structures. Besides, there are some conversational protocols requiring certain transformations on certain words. For example, some words can be unconditionally substituted by other words. These transformations should be performed during the text scan.   
   
For each keyword, there are some corresponding decomposition rules. For example, if the input text is 'It seems that you hate me.', the decomposition can be '0 YOU 0 ME', where '0' represents any number of words. And then the reassembly rule can be 'WHAT MAKES YOU THINK I 3 YOU', where '3' means the third component of the decomposition.  
   
A set of decomposition rules which are more potentially applicable will be chosen according to the 'keyword mechanism'. In short words, the keywords have ranks in the dictionary, and the pointer to the rule associated with the keyword of the highest rank is placed on top of a list called the keystack. If a decomposition rule fails to match the input sentence, then the next rule in the list is tried. Once the input text is decomposed, the corresponding reassembly rule is retrieved to generate an output message. For some keywords requiring identical sets of transformation rules, there are special types of rules characterized by '=' so that the same rule is applied to the keywords and the storage consumption is reduced.   
   
In many cases, the keystack can be empty or the input text contains no keywords and therefore no transformation rules are triggered. To solve this problem, there is a reserved keyword 'NONE' in the script. It’s associated with universally matching decomposition rules and followed by many content-free remarks in the form of transformation rules. Another reserved keyword is ‘MEMORY’ which is associated with another ordinary keyword. When the ordinary keyword is the highest ranking, one transformation on the MEMORY list is randomly selected and a copy of text is transformed with it. This chosen transformation is stored on a first-in-first-out stack. If we encounter a text without keywords later, we can use the transformation in the stack and print out the transformed text as a reply.   
   
Strengths and weakness: As discussed in the paper, ELIZA can simulate psychiatrist or psychotherapist and performs best when the speaker knows almost nothing of the real world. In this sense, the speakers contribute more to the conversation (background knowledge and insights)  and the task of ELIZA is hearing and understanding. So one advantage is that the script doesn’t need to store too much information of the real word. Besides, it performs well in terms of maintaining the illusion of understanding. On the other hand, although ELIZA is good at continuing a conversation, it cannot draw valid conclusions from what it’s told because it hardly stores the input information.   
   
References:    
[1] Weizenbaum, Joseph. "ELIZA—a computer program for the study of natural language communication between man and machine." Communications of the ACM 9.1 (1966): 36-45.
   


# 5. Research one non-English language. Write a summary of the major differences in this language from English (~350-700 words). Differences can be syntactic or semantic, but must be nontrivial. If your group does not have any familiarity with any non-English language, find someone who does (outside of class) and write an answer with their help. Wikipedia/Duolingo or similar can also be useful. (20 points)

ANSWER:
---
Difference 1:
The most basic unit of English is 26 letters, which utilizes combinations of those letters to make up words. Those intact words then construct sentences to demonstrate various semantics. But in Chinese, the most basic unit is a character, where each character represents one syllable with its own meaning. The Chinese language has a colossal number of characters creating words and phrases with the purpose of forming sentences to represent something without space in a characteristic way. 
    
Difference 2:
As mentioned in the 1st difference, Chinese has no spaces between words even in ancient China there is no punctuation mark such as comma, indicating that unless there is full understanding of the context, the sequence of characters or words in a Chinese sentence may cause huge ambiguity. Sometimes people need to directly consult with the author writing ambiguous sentences to figure out the real meaning. By contrast, English texts have a strict placement of space and punctuation.
    
Difference 3:
When we refer to Chinese, we’re usually referring to Mandarin Chinese, which is the official language of mainland China, while Cantonese is more often spoken in Hong Kong. There exist some different versions of the Chinese language based on region: Mandarin, Wu, Gan, Xiang, Min, Cantonese, Hakka, Jin, Hui, and Pinghua. In fact, some of those varieties of Chinese are so undistinguishable that even native speakers won’t understand one another. As for English, I heard that people in different countries such as the USA and the UK or in different states such as California and Ohio do have different accents and minor grammar differences. But generally speaking, it's not a hard time for them to have normal communication.
    
Difference 4:
There are currently two main styles of writing Chinese characters: Traditional and Simplified Chinese. The simplified version is more modern, coming to the force in the 1950s and 1960s. The main differences between the two versions are the number of characters and a simpler style. By comparison, English always has 26 single letters.   
   
Difference 5:
Chinese Names of places and organizations are usually written from largest to smallest. For example, in XX County, XX City, XX Province, China, it is customary to put high-level ones in the front and low-level ones in the back, while English is just the opposite. What's more, for Chinese last names ought to be placed first and first names ought to be placed behind. English is the opposite.   
   
Difference 6:
Chinese is a tonal language, which means pronunciation plays a significant role in spoken Chinese. There are four major tones that will completely change the meaning of the word based on how it’s spoken. For instance, first mā can mean mother, second má can mean sesame, third mǎ can mean horse, and fourth mà can mean scolding. It's not hard for us to notice that tones dramatically change the meaning of Chinese words. To the best of my knowledge, English has no such rules to follow.   
   
Difference 7:
Chinese uses simpler grammatical structures than English, which, to some extent, is one of the most challenging languages to learn due to grammatical rules. Chinese grammar is much more straightforward without tenses or plurality. More specifically, Chinese nouns don't have plurals, and verbs have no changes in tenses, voices, the subjective, and so on. In other words, Chinese words have only one unique form without any changes due to grammatical requirements. Therefore, people must attach more significance to the whole content of the sentence because it will reflect various tenses, voices, subject-object, singular, and plural grammatical meanings.

Difference 8:
In terms of fundamental word order, Chinese is generally classified as a subject-verb-object (SVO) language, which is like English. However, there are considerable exceptions. For instance, with noun phrases, the head noun always occurs at the end of the phrase, while relative clauses always come before the noun, which are indicated by a special particle 的 (de). For example, "de" is similar to "'s" in "Bob's car" but there is no expression in Chinese for "car of Bob".

Difference 9:
Chinese has its own novel expressions and odd metaphors to describe things, which is greatly different from English since it's largely influenced by Chinese culture and history of at least 2000 years, for which in 221 BC China was first unified under control of the Qin dynasty. Hence, sometimes we will fail to catch the core meaning of a given Chinese sentence by crudely translating it into English.

# 6. Using your answer to question 5, and thinking about the NLP/NLG pipelines and various tasks we have discussed in class, describe what differences might arise in building NLP systems for the language you researched, as compared to building an NLP/NLG system for English (~350-700 words). (20 points)

ANSWER:
---
For the first difference, NLP/NLG systems for Chinese should try to store Chinese characters as much as possible in their database for future vocabulary reference.

For the second difference, for NLU systems, before using grammar and vocabulary to parse Chinese texts we are bound to do text segmentation to make a sequence of Chinese words delimited with spaces like English. However, complications arise because there are frequently more than one way in which a given sequence of characters can be segmented, with each way holding a different meaning from others. In that situation, paying more attention to the surrounding context will help. For NLG systems, we should eliminate spaces of texts before doing speech synthesis.

For the third difference, there should be a specific machine translation system that is capable of translating different versions of Chinese texts into each other if we want to study one specific Chinese language in NLU/NLG systems.

For the fourth difference, we need to have a tool that could realize the invertible one-to-one mapping between a simplified Chinese word and a traditional Chinese word.

For the fifth difference, I think we don't have to add a new part for the English NLU/NLG systems. However, we need to modify the parsing rule when growing a Parse Tree in the NLU system and establishing a slightly different Parse Tree Builder in the NLG system, which is fit to Chinese order when mentioning names of Chinese people, places and organizations.

For the sixth difference, for NLU people are supposed to attach a great amount of importance to the Automatic Speech Recognition part since pronunciation is crucial to understand a given Chinese speech. Similarly, for NLG, Speech Synthesis should be regarded as equally important.

For the seventh difference, I have mentioned that Chinese grammar is simpler than English due to not considering tenses or plurality. Therefore, people must pay adequate attention to the whole content of the given sentence when doing the Semantic Parsing job for NLU, which is the same when handling the Semantically Annotation part for NLG.

For the eighth difference, I think Chinese's rules are simpler than English. All we have to do is to carefully consider the "de" rule in Chinese when finishing tasks relevant to semantics both for NLU and NLG.

For the ninth difference, I hold the strong belief that NLU/NLG developers should try to comprehensively contain common and popular Chinese novel expressions and odd metaphors when building the vocabulary database. When given a Chinese paragraph, we could quickly discover the complete idioms or phrases based on the database, thus avoiding adding unnecessary spaces when doing text segmentation. This is also conducive to semantics analysis by restraining awkwardly understanding the superficial meaning.


References for 5 and 6                               
[1]https://toppandigital.com/us/blog-usa/challenges-translating-chinese-natural-language-processing/
[2]https://www.smartling.com/resources/101/5-common-challenges-for-english-chinese-translations/
[3]https://multilingual.com/articles/machine-translation-and-the-challenge-of-chinese/

