from transformers import AutoModel, AutoTokenizer
from torch.nn import Module
import torch



class CLSExtractor:
    def __call__(self, out):
        return out.last_hidden_state[:, 0]

class PoolerExtractor:
    def __call__(self, out):
        return out['pooler_output']


class BaseTransformerEncoder(Module):
    def __init__(self, model_name_or_path, output_layer='cls', device='cpu'):
        super(BaseTransformerEncoder, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModel.from_pretrained(model_name_or_path).to(device)
        self.device = device

        if output_layer == 'cls':
            self.output_layer = CLSExtractor()
        elif output_layer == 'pooler':
            self.output_layer = PoolerExtractor()
        else:
            raise ValueError

    def forward(self, input):
        input = self.tokenizer(input, padding=True, truncation=True, return_tensors='pt').to(self.device)
        input = self.model(**input)
        return self.output_layer(input)








class RecurrentTransformerEncoder(BaseTransformerEncoder):
    def __init__(self, model_name_or_path, output_layer='cls',
                 device='cpu', recurrent_model = torch.nn.GRU, hidden_size=128,
                 num_layers=4, dropout=0):
        super(RecurrentTransformerEncoder, self).__init__(model_name_or_path, output_layer, device)
        # self.recurrent_model =
        # self.first_hidden =
        input_size = self.model.config.hidden_size
        self.hidden_size = hidden_size
        self.RNN = recurrent_model(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=False)

    def forward(self, input, ):
        input = super().forward(input)



        def forward(self, input, hidden):
            embedded = self.embedding(input).view(1, 1, -1)
            output = embedded
            output, hidden = self.gru(output, hidden)
            return output, hidden


    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=self.device)


class DenseRetriever:
    def __init__(self, vectorized_knowledge_file, score_function, projections=['K'], topk=5):

        tmp = torch.load(vectorized_knowledge_file)
        self.projections = projections
        self.topk = topk
        # self.matrices = [experiments[p]['matrices'] for p in self.projections]
        # self.indices = [experiments[p]['keys'] for p in self.projections]
        self.score_function = score_function
        self.projection_tree = {}

        indices = [set()]
        for p in projections:
            p_matrix = tmp[p]['matrix']
            p_indices = tmp[p]['keys']
            start = 0
            end = start
            for ind in indices:

                while end < len(p_indices) and len(set(ind).difference(p_indices[end])) == 0:
                    end += 1

                p_sliced_matrix = p_matrix[start:end, :]
                p_sliced_indices = [tuple(k) for k in p_indices[start:end]]

                self.projection_tree[tuple(ind)] = (p_sliced_matrix, p_sliced_indices)
                start = end
            indices = p_indices

    def __call__(self, vector):
        key = tuple()
        p_matrix, p_indices = self.projection_tree[key]

        for p in self.projections:
            scores = self.score_function(vector, p_matrix)

            if 'K' not in p:
                max_ind = scores.argmax().item()
                key = p_indices[max_ind]
                p_matrix, p_indices = self.projection_tree[key]
            else: # final prediction
                topk_ids = scores.topk(min(self.topk, len(scores))).indices
                key = [p_indices[ii.item()] for ii in topk_ids]

        return key



# class Encoder():
#     def __init__(self, model, tokenizer, preprocessing_transform=None, postprocessing_transform=None):
#         self.model = model
#         self.tokenizer = tokenizer
#         self.preprocessing_transform = preprocessing_transform
#         self.postprocessing_transform = postprocessing_transform
#
#     #todo implement batches
#     def __call__(self, item):
#
#         try:
#             iter(item)
#             data = item
#         except:
#             data = [item]
#
#         if self.preprocessing_transform is not None:
#         #     x = [self.preprocessing_transform(val) for val in item]
#             data = self.preprocessing_transform(data)
#
#
#         tokenized_data = self.tokenizer(data, return_tensors='pt', padding=True, truncation=True)
#         tokenized_data = {key:value.to(self.model.device) for key, value in tokenized_data.items()}
#         out = self.model(**tokenized_data)
#         del tokenized_data, data
#
#         if self.postprocessing_transform is not None:
#             out = self.postprocessing_transform(out)
#
#         return out

class Cosine:
    def __call__(self, vector, matrix):
        vector = vector.view(1,-1)
        v_shape = vector.shape
        m_shape = vector.shape

        if v_shape[1] == m_shape[1]:
            r = torch.cosine_similarity(vector, matrix)
        else:
            r = torch.cosine_similarity(vector, matrix.T)

        return r.flatten()

class DotProd:
    def __call__(self, vector, matrix):
        vector = vector.view(1,-1)
        v_shape = vector.shape
        m_shape = vector.shape

        if v_shape[1] == m_shape[0]:
            r = vector @ matrix
        else:
            r = vector @ matrix.T

        return r.flatten()

class Euclidean:

    def __call__(self, vector, matrix):
        vector = vector.view(1, -1)
        v_shape = vector.shape
        m_shape = matrix.shape

        if v_shape[1] == m_shape[1]:
            r = torch.cdist(vector, matrix)
        else:
            r = torch.cdist(vector, matrix.T)

        return 1/(r.flatten())


if __name__ == '__main__':
    # e = BaseTransformerEncoder('distilroberta-base')
    # from dataset import KnowledgeBase
    # k_base = KnowledgeBase('./DSTC9/data/knowledge.json')
    # from transforms import ConcatenateKnowledge
    #
    # x = ConcatenateKnowledge()(k_base[0])
    # e([x])

    d = DenseRetriever('DSTC9/data_eval/vectorized_knowledge/all-mpnet-base-v2_cls.pt', torch.cosine_similarity, ['DE', 'K'])
    for i in range(10):
        x = d(torch.rand(1, 768), Euclidean())
        print(x)