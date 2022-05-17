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

    def forward(self, input, **kwargs):
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

    def forward(self, input, *args, output=None, **kwargs):
        input_encoding = self.sentence_encoder(input)

        if output is not None:
            if not hasattr(self, 'heads'):
                raise AttributeError('The encoder has no heads ({})'.format(output))
            elif isinstance(output, list):
                return {key:self.heads[key](input_encoding) for key in output}
            else:
                return self.heads[output](input_encoding, *args, **kwargs)

        return input_encoding


from torch import Tensor, sum, exp, mm, bmm, nn

class NContrastiveLoss(nn.Module):
    '''
    Compute generalised Contrastive Loss, where there is 1 positive and N negative labels
    for each instance. The embedding of the instance gets pulled close to the positive
    label embedding while being pushed away from each of the N negative labels embeddings.
    https://papers.nips.cc/paper/2016/hash/6b180037abbebea991d8b1232f8a8ca9-Abstract.html
    '''
    def __init__(self) -> None:
        super(NContrastiveLoss, self).__init__()

    def forward(self, anchor: Tensor, positive: Tensor, negatives: Tensor) -> torch.float:
        '''
        Pulls anchor and positive closer together and pushes anchor and negatives further
        apart.
        For each example in the batch, there is 1 anchor, 1 positive and N negatives.
        The loss formulated here optimizes the dot product.

        Parameters
        ----------
        anchor: 2D tensor
                batch of anchors embeddings
        positive: 2D tensor
                  batch of positive embedding
        negatives : 3D tensor
                    batch of N-negatives embeddings per

        Returns
        -------
        Float tensor
            Sum of N-contrastive-loss for each element of the batch.
        '''
        # Make anchor and positive tensors 3D, by expanding empty dimension 1.
        batch_size = len(anchor)
        anchor = anchor.unsqueeze(1)
        positive = positive.unsqueeze(1)
        # Compute loss.
        A = exp(bmm(anchor, positive.transpose(2, 1))).view(batch_size)
        B = sum(exp(bmm(anchor, negatives.transpose(2, 1)).squeeze(1)), dim=-1)
        return -sum(torch.log(A / (A + B)), dim=-1) / batch_size

    def _get_name(self):
        return 'NContrastiveLoss'



if __name__ == '__main__':
    model= HeadedEncoder('sentence-transformers/all-MiniLM-L6-v2', device='cpu')

    model('Hello, world!')

    from utils import *
    from dataset import *
    knowledge_base = KnowledgeBase('./DSTC9/data')
    entity_embeddings = {}
    for domain, df in knowledge_base.knowledge.groupby('domain'):
        entities = df[['entity_id', 'name']].drop_duplicates()
        entity_ids = entities.entity_id.tolist()
        entity_embeddings[domain] = model(entities.name.tolist(), output='name')
        print(entity_embeddings)
        break
