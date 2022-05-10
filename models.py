from transformers import AutoModel, AutoTokenizer
from torch import nn
import torch

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

class SentenceEncoder(nn.Module):
    def __init__(self, model_name, device, truncation_side='right'):
        super(SentenceEncoder, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, truncation_side=truncation_side, device=device)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.hidden_size = self.model.config.hidden_size
        self.device = device

    def forward(self, input):
        encoded_input = self.tokenizer(input, padding=True, truncation=True, return_tensors='pt')
        encoded_input = {key:val.to(self.device) for key, val in encoded_input.items()}
        model_output = self.model(**encoded_input)
        return mean_pooling(model_output, encoded_input['attention_mask'])

class LinearHead(nn.Module):
  def __init__(self, in_dim, out_dim, device, dropout = 0.3):
    super().__init__()
    self.linear = nn.Linear(in_features=in_dim, out_features=out_dim)
    torch.nn.init.eye_(self.linear.weight)
    # torch.nn.init.xavier_uniform_(self.classification.weight, gain=1.0)
    # torch.nn.init.kaiming_normal_(self.classification.weight)
    # torch.nn.init.eye_(self.classification.weight)
    torch.nn.init.zeros_(self.linear.bias)
    self.device = device
    self.dropout = nn.Dropout(dropout)
    self.to(device)

  def forward(self, documents_embeddings):
    docs = self.dropout(documents_embeddings)
    scores = self.linear(docs)
    return scores

class HeadedEncoder(nn.Module):
    def __init__(self, base_encoder_name_or_path, device,
                 heads=['domain', 'name', 'doc'], truncation_side='right'):

        super(HeadedEncoder, self).__init__()
        self.sentence_encoder = SentenceEncoder(base_encoder_name_or_path, device=device, truncation_side=truncation_side)
        self.hidden_size = self.sentence_encoder.hidden_size

        if isinstance(heads, dict):
            self.heads = heads
        elif isinstance(heads, list):
            self.heads = {key:LinearHead(self.hidden_size, self.hidden_size, device=device) for key in heads}

    def forward(self, input, output=None):
        input_encoding = self.sentence_encoder(input)

        if output is not None:
            if not hasattr(self, 'heads'):
                raise AttributeError('The encoder has no heads ({})'.format(output))
            elif isinstance(output, list):
                return {key:self.heads[key](input_encoding) for key in output}
            else:
                return self.heads[output](input_encoding)

        return input_encoding
