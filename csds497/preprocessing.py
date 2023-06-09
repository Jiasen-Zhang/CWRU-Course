
from transformers import BertTokenizer, BertModel
import torch
import numpy as np
# import nltk
# from nltk.stem.porter import PorterStemmer
# nltk.download('averaged_perceptron_tagger')


'''Load the text and tokenization'''
def load_text(filename):
    with open(filename) as file:
        text = file.read()

    # # convert to lowercase
    text = text.lower()

    # remove the last full stop
    text = text[:-2]
    text = text.replace('. . .', '')
    text = text.replace('\n\n', '. ')
    text = text.replace('\n', '')
    text = text.replace('  ', ' ')

    text = text.replace('.m.', '<time>')
    text = text.replace('u.s.', '<us>')
    text = text.replace('. ', '\s')

    # # divide text into sentences
    text = text.replace('<time>', '.m.')
    text = text.replace('<us>', 'u.s.')
    # text = text.replace('<number>', '0 ')
    text = text.split('\s')

    while '' in text:
        text.remove('')

    return text


# '''remove less important words according to their POS'''
# def synatic_filter(text):
#     text_filtered = []
#     for sentence in text:
#         sentence_filtered = []
#         sentence_tag = nltk.pos_tag(sentence)
#         for item in sentence_tag:
#             if item[1] in ['NN','JJ','NNS']:
#                 sentence_filtered.append(item[0])
#         text_filtered.append(sentence_filtered)
#
#     return text_filtered


def bert_embedding(text):
    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model_name = "bert-base-uncased"
    bert_tokenizer = BertTokenizer.from_pretrained(model_name)
    bert_model = BertModel.from_pretrained(model_name, output_hidden_states = True)

    bert_model.eval()
    # bert_model.to(device)

    text_embedding = []

    for sentence in text:
        tokenized_sent = bert_tokenizer.tokenize(sentence)
        # print(tokenized_sent)
        indexed_tokens = bert_tokenizer.convert_tokens_to_ids(tokenized_sent)
        # print(indexed_tokens)

        tokens_tensor = torch.tensor([indexed_tokens])

        with torch.no_grad():
            output = bert_model(tokens_tensor.long())
            # print(output[0])

            hidden_states = output[2]
            token_embeddings = torch.stack(hidden_states, dim=0)
            token_embeddings = torch.squeeze(token_embeddings, dim=1)
            token_embeddings = token_embeddings.permute(1, 0, 2)

            token_vecs_sum = []

            for i in token_embeddings:
                sum_vec = torch.sum(i[-4:], dim=0)
                token_vecs_sum.append(sum_vec.cpu().numpy())

            # token_vecs_sum = np.mean(token_vecs_sum, axis=1)
            # print(token_vecs_sum)

        text_embedding.append(token_vecs_sum[0])

    result = np.array(text_embedding)
    return result

