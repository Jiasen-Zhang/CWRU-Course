Jiasen Zhang     
----------------------------------------------------------------------------------------------------------------------     
Learning-Free Integer Programming Summarizer (LFIP-SUM)      
----------------------------------------------------------------------------------------------------------------------      

Python files:     
1. main.py: This is the main file to summarize all the articles in the dataset and evaluate performance.
2. model.py: Include the functions performing the algorithms in LFIP-SUM and its extensions.     
3. preprocessing.py: Load the data files and get word embedding representations with pre-trained BERT model.

The test dateset is in the 'data' folder.     
       
----------------------------------------------------------------------------------------------------------------------      
Requirements of libraries:         
numpy        
torch          
transformers (may take about one minute to download pre-trained BERT model (420MB))             
rouge_score (available in pip)          

----------------------------------------------------------------------------------------------------------------------      
How to run:     
There are 3 parameters:          
Parameter 1: the factor of sentence boundary function. It should be nonnegative. Equal to zero means no effect.         
Parameter 2: the decay factor of diffusion process. It should be between 0 and 1. Equal to zero means no effect.          
Parameter 3: the number of self-attention. It should be positive integer and cannot be zero.         

Examples:     
Original LFIP-SUM: python main.py 0 0 1       
Apply diffusion process with factor 0.9: python main.py 0 0.9 1       
Apply sentence boundary function with factor 1: python main.py 1 0 1       
Apply 3 self-attentions: python main.py 0 0 3      
