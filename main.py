from torch.utils.data import DataLoader

from models import Encoder
from transforms import ConcatenateKnowledge, CLSExtractor
from dataset import VectorizedKnowledgeBase, KnowledgeBase


if __name__ == '__main__':
    from transformers import AutoTokenizer, AutoModel
    import torch
    tokenizer = AutoTokenizer.from_pretrained('distilroberta-base')
    model = AutoModel.from_pretrained('distilroberta-base')

    #knowledge
    k_preprocessing = ConcatenateKnowledge(sep_token=tokenizer.sep_token)
    k_postprocessing = CLSExtractor()
    k_encoder = Encoder(model, tokenizer, k_preprocessing, k_postprocessing)
    k_base = KnowledgeBase("DSTC9/data", "knowledge.json")

    k_vectors = torch.cat([k_encoder(val) for val in k_base[:10]])


    #dialog_context
    from transforms import ConcatenateDialogContext, PoolerExtractor
    from dataset import DSTCDataset
    d_preprocessing = ConcatenateDialogContext(user_special_token='<U>', agent_special_token='<S>')
    d_postprocessing = PoolerExtractor()
    d_encoder = Encoder(model, tokenizer, d_preprocessing, d_postprocessing)
    d_train_dataset = DSTCDataset('train', 'DSTC9/data/', knowledge_base=k_base, encoder=d_encoder, labels=True)

    d_train_dataset[0]

    #parameters
    d_train_dataloader = DataLoader(d_train_dataset, batch_size=2)

    from torch.nn.functional import softmax
    from torch import argmax
    for batch in d_train_dataloader:
        res = batch[0] @ k_base.knowledge_vectors.T
        res = softmax(res, dim=-1).topk(k=5)
        res = argmax(res, ).flatten()
        res = [k_base.list_docs[x] for x in res]
        break
        # dialog_batch, knowledge_batch = batch

