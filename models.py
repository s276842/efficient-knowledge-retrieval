from transformers import AutoModel, AutoTokenizer
from torch.nn import Module

class CLSExtractor:
    def __call__(self, out):
        return out.last_hidden_state[:, 0]

class PoolerExtractor:
    def __call__(self, out):
        return out['pooler_output']


class BaseTransformerEncoder(Module):
    def __init__(self, model_name_or_path, outputlayer='cls', device='cpu'):
        super(BaseTransformerEncoder, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModel.from_pretrained(model_name_or_path).to(device)
        self.device = device

        if outputlayer == 'cls':
            self.outputlayer = CLSExtractor()
        elif outputlayer == 'pooler':
            self.outputlayer = PoolerExtractor()
        else:
            raise ValueError

    def forward(self, data):
        data = self.tokenizer(data, padding=True, truncation=True, return_tensors='pt')
        data = {key:val.to(self.device) for key, val in data.items()}
        data = self.model(**data)
        return self.outputlayer(data)

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

if __name__ == '__main__':
    e = BaseTransformerEncoder('distilroberta-base')
    from dataset import KnowledgeBase
    k_base = KnowledgeBase('./DSTC9/data/knowledge.json')
    from transforms import ConcatenateKnowledge

    x = ConcatenateKnowledge()(k_base[0])
    e([x])