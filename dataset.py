import os.path

from torch.utils.data import Dataset
from scripts.knowledge_reader import KnowledgeReader
from scripts.dataset_walker import DatasetWalker
import random
import torch
import json
from bisect import bisect
from collections.abc import Iterable
from transforms import ConcatenateDialogContext, ConcatenateKnowledge
EMPTY_DOC = {'domain':'', 'entity_id':'', 'doc_id':''}
EMPTY_RESPONSE = ''



class KnowledgeBase:

    def __init__(self, knowledge_file='knowledge.json'):

        # load knowledge database as dictionary
        with open(knowledge_file, 'r') as f:
            self.knowledge = json.load(f)

        #
        self.domains = list(self.knowledge)
        self.entity_ids = [list(self.knowledge[domain]) for domain in self.domains]
        self.doc_ids = [[list(self.knowledge[domain][entity_id]['docs']) for entity_id in self.knowledge[domain]] for domain in self.domains]

        # iterate through the database and compute index span range for each domain and entity
        n_domains = len(self.knowledge)
        self.range_domain_docs = [0] * (n_domains+1)
        self.range_entity_docs = []

        for i, domain in enumerate(self.knowledge):
            self.range_domain_docs[i+1] = self.range_domain_docs[i]
            n_entities = len(self.knowledge[domain])
            self.range_entity_docs.append([self.range_domain_docs[i]] * (n_entities+1))

            for j, entity_id in enumerate(self.knowledge[domain]):
                self.range_entity_docs[i][j+1] = self.range_entity_docs[i][j]
                n_docs = len(self.knowledge[domain][entity_id]['docs'])
                self.range_domain_docs[i+1] += n_docs
                self.range_entity_docs[i][j+1] += n_docs

        self.len = self.range_domain_docs[-1]

    def __len__(self):
        return self.len

    def int_to_data(self, index):
        domain_ind = bisect(self.range_domain_docs, index) - 1
        domain = self.domains[domain_ind]
        entity_id_ind = bisect(self.range_entity_docs[domain_ind], index) - 1
        entity_id = self.entity_ids[domain_ind][entity_id_ind]
        entity_id = int(entity_id) if entity_id != '*' else entity_id
        doc_ind = index - self.range_entity_docs[domain_ind][entity_id_ind]
        doc_id = int(self.doc_ids[domain_ind][entity_id_ind][doc_ind])

        return {'domain': domain, 'entity_id': entity_id, 'doc_id': doc_id}

    def __getitem__(self, item):
        if isinstance(item, int):
            if item < 0:
                item += self.len
            if item < 0 or item >= self.len:
                raise IndexError(f'Index out of bound: {item}')
            else:
                keys = self.int_to_data(item)

        elif isinstance(item, slice):
            return [self[ii] for ii in range(*item.indices(len(self)))]

        elif isinstance(item, list):
            if all(isinstance(x, str) for x in item):
                if len(item) == 1:
                    return self.knowledge[item[0]]

            return [self[ii] for ii in item]

        elif isinstance(item, str):
            return self.knowledge[item]
        else:
            raise ValueError

        return self.getdoc(**keys)

    def getdoc(self, domain, entity_id, doc_id):
        tmp = self.knowledge[domain][str(entity_id)]
        entity_name = tmp['name']
        question, answer = tmp['docs'][str(doc_id)].values()
        entity_id = int(entity_id) if entity_id != '*' else entity_id
        doc_id = int(doc_id)
        return {'domain': domain,
                'entity_id': entity_id,
                'entity_name': entity_name,
                'doc_id': doc_id,
                'question': question,
                'answer': answer}

    def __iter__(self):
        for i in range(self.len):
            yield self.__getitem__(i)
# class VectorizedKnowledgeBase(KnowledgeBase):
#     def __init__(self, encoder, *args):
#         super(VectorizedKnowledgeBase, self).__init__(*args)
#         self.encoder = encoder
#         self.vectorize()
#
#     def vectorize(self):
#         self.knowledge_vectors = torch.cat([self.encoder(self.__getitem__(doc_ids)) for doc_ids in self.doc_list[:10]])
# class SelectionDialogContextDataset(DialogContextDataset):
#     def __init__(self, log_file_path, label_file_path=None, data_transform=None):
#         super(SelectionDialogContextDataset, self).__init__(log_file_path, label_file_path, data_transform)

class DialogContext:

    def __init__(self, log_file_path, label_file_path=None, knowledge_seeking_turns_only=False, data_transform=None):

        with open(log_file_path, 'r') as f:
            self.logs = json.load(f)

        self.labels = None

        if label_file_path is not None:
            with open(label_file_path, 'r') as f:
                self.labels = json.load(f)

        if knowledge_seeking_turns_only:
            self.__filter_turns()

        self.data_transform = data_transform

    def __filter_turns(self):
        # filter selection turns only
        only_selection_dialogs = []
        only_selection_labels = []
        for dialog_context, label in zip(self.logs, self.labels):
            if label['target'] == True:
                only_selection_dialogs.append(dialog_context)

                label = label['knowledge'][0]
                label['entity_id'] = str(label['entity_id'])
                label['doc_id'] = str(label['doc_id'])
                only_selection_labels.append(label)

        self.logs = only_selection_dialogs
        self.labels = only_selection_labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        if isinstance(item, int):
            data = self.logs[item]
            label = None

            if self.labels is not None:
                label = self.labels[item]

        elif isinstance(item, slice):
            return [self[ii] for ii in range(*item.indices(len(self)))]

        elif isinstance(item, list):
            return [self[ii] for ii in item]


        return self.data_transform(data), label

    def __iter__(self):
        for i in range(self.__len__()):
            yield self[i]



# class SelectionDSTC9Dataset(Dataset):
#     def __init__(self, k_base, d_base: SelectionDialogContext, data_transform=ConcatenateDialogContext(), target_transform=ConcatenateKnowledge()):
#         self.k_base = k_base
#         self.d_data = d_base
#
#         self.data_transform = data_transform
#         self.target_transform = target_transform
#
#     def __len__(self):
#         len(self.d_data)
#
#     def __getitem__(self, item):
#         if isinstance(item, int):
#             data, label = self.d_data[item]
#             knowledge = self.k_base.getdoc(**label)
#
#             if self.data_transform is not None:
#                 data = self.data_transform(data)
#
#             if self.target_transform is not None:
#                 knowledge = self.target_transform(knowledge)
#
#         return data, knowledge, label


class TripletSelectionDSTC9Dataset(Dataset):

    def __getitem__(self, item):
        anchor, pos, label = super(TripletSelectionDSTC9Dataset, self).__getitem__(item)

        domain, entity_id, doc_id = label.values()

        # sample neg
        # if self.target_transform is not None:
        #   neg = self.target_transform(neg)
        # return anchor, pos, neg, label


if __name__ == '__main__':
    import json
    from torch.utils.data import DataLoader
    log_file_path = './DSTC9/data/train/logs.json'
    label_file_path = './DSTC9/data/train/labels.json'
    d = DialogContext(log_file_path, label_file_path)
    k = KnowledgeBase('DSTC9/data/knowledge.json')

    d_dataset = SelectionDSTC9Dataset(d_base=d,k_base=k)
    d_dataset[0]
    d_dataloader = DataLoader(d_dataset, batch_size=8, shuffle=True)

    for batch in d_dataloader:
        print([k.getdoc(d,e,k2) for d,e,k2 in zip(*batch[1].values())])
        break

    # k.getdoc(batch[2], batch[3], batch[4])
    # import time
    # start = time.time()
    # print([k[:10] for _ in range(100000)])
    # print(time.time() - start)
    # d_train_dataset = DSTCDataset('train', 'DSTC9/data/', labels=True)

    # from transformers import AutoTokenizer, AutoModel
    # tokenizer = AutoTokenizer.from_pretrained('distilroberta-base')
    # model = AutoModel.from_pretrained('distilroberta-base')
    #
    # from transforms import ConcatenateKnowledge, CLSExtractor
    # knowledge_preprocessing = ConcatenateKnowledge(sep_token=tokenizer.sep_token)
    # knowledge_postprocessing = CLSExtractor()
    #
    # from models import Encoder
    # encoder = Encoder(model, tokenizer, knowledge_preprocessing, knowledge_postprocessing)
    #
    #
    # kb = VectorizedKnowledgeBase(encoder, "DSTC9/data", "knowledge.json")
    # kb.vectorize()