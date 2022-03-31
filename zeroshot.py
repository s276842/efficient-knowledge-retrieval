import json
import time

from models import BaseTransformerEncoder
from dataset import DialogContext
from torch.utils.data import DataLoader



def main():

    k_vectors = torch.load(knowledge_vectors_path)

    d_data = DialogContext(log_file_path, label_file_path)
    d_dataloader = DataLoader(d_data, batch_size)

    d_encoder = BaseTransformerEncoder(model_path, outputlayer, device)

    s_model = ScoreModel()

    res = {}
    total_selection_time = .0
    k = 0
    with torch.no_grad():
        for data, label in d_dataloader:
            if label['target'] == False:
                res.append(label)
            else:
                k+=1
                start_selection_time = time.time()
                e_data = d_encoder(data)
                k_index = s_model(e_data, k_vectors)
                k_docs = [k_base[ind] for ind in k_index]

                total_selection_time += time.time() - start_selection_time
                res.append()

        return res, total_selection_time, total_selection_time/k




if __name__ == '__main__':
    #--device        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #--model_path = 'sentence-transformers/all-mpnet-base-v2'
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path).to(device)

    from models import Encoder
    from transforms import ConcatenateKnowledge, CLSExtractor, PoolerExtractor
    from dataset import VectorizedKnowledgeBase, KnowledgeBase
    import torch

    k_preprocessing = ConcatenateKnowledge(sep_token=tokenizer.sep_token, add_domain=True, add_entity_name=True)
    k_postprocessing = PoolerExtractor()
    k_encoder = Encoder(model, tokenizer, k_preprocessing, k_postprocessing)
    k_base = KnowledgeBase("DSTC9/data", "knowledge.json")

    from transforms import ConcatenateDialogContext, PoolerExtractor
    from dataset import DSTCDataset

    d_preprocessing = ConcatenateDialogContext()
    d_postprocessing = PoolerExtractor()
    d_encoder = Encoder(model, tokenizer, d_preprocessing, d_postprocessing)
    d_train_dataset = DSTCDataset('val', 'DSTC9/data/', knowledge_base=k_base, encoder=d_encoder, labels=True)

    knowledge_vectors_path = 'DSTC9/data/all-mpnet-base-v2-domain-entity-question-answer'
    k_vectors = torch.load(knowledge_vectors_path)

    res = []
    from tqdm import tqdm

    with torch.no_grad():
        for dialog_context, label in tqdm(d_train_dataset):
            if label['target'] == False:
                res.append(label)
            else:
                r = {'target': True}
                dialog_context = d_preprocessing(dialog_context)

                dialog_context = tokenizer(dialog_context, return_tensors='pt', truncation=True)['input_ids']
                dialog_context = model(dialog_context.to(device))
                dialog_context = d_postprocessing(dialog_context).to('cpu')
                # scores = d_encoder(dialog_context) @ k_vectors.T
                scores = dialog_context @ k_vectors.T
                best_knowledge = scores.topk(5).indices.flatten()
                best_knowledge = [k_base.doc_list[ind] for ind in best_knowledge]
                r['knowledge'] = best_knowledge
                r['response'] = label['response']

                res.append(r)
