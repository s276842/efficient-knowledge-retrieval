import json
import os.path
import random
from torch.utils.data import Dataset
from typing import Callable
import pandas as pd
import numpy as np
from utils import *


class KnowledgeBase:
    def __init__(self, dataroot: str, knowledge_file: str = 'knowledge.json'):

        knowledge_path = os.path.join(dataroot, knowledge_file)

        with open(knowledge_path, 'r') as f:
            self.knowledge = json.load(f)


        data = []
        for domain in self.knowledge:
            sample = {'domain':domain}
            d_dict = self.knowledge[domain]
            for entity_id in d_dict:
                e_dict = d_dict[entity_id]
                sample['entity_id'] = entity_id
                if e_dict['name'] is None:
                    e_dict['name'] = domain
                sample.update({key:val for key, val in e_dict.items() if key != 'docs'})

                for doc_id in e_dict['docs']:
                    sample['doc_id'] = doc_id
                    sample['question'] = e_dict['docs'][doc_id]['title']
                    sample['answer'] = e_dict['docs'][doc_id]['body']
                    data.append(sample.copy())

        self.knowledge = pd.DataFrame(data)
        self.domains = self.knowledge.domain.unique().tolist()
        self.entities = {domain: df.name.unique().tolist() for domain, df in self.knowledge.groupby('domain')}

    def __getitem__(self, item):
        if isinstance(item, str):
            return self.knowledge[self.knowledge.domain == item]

        elif isinstance(item, tuple):
            mask = (self.knowledge.domain == item[0]).values
            for val, field in zip(item[1:], [self.knowledge.entity_id, self.knowledge.doc_id]):
                mask &= (field == val).values

            return self.knowledge[mask]

        elif isinstance(item, int):
            return self.knowledge.iloc[item]
        else:
            raise IndexError



def sample_kb_triplets(knowledge_base, anchor='domain', pos=['question', 'answer'], neg=['question', 'answer'], n_samples=1000, n_negatives=1):
    # preparing corpus, pos and neg
    n_samples_per_domain = n_samples//len(knowledge_base.domains)
    samples = []

    anchor_preprocessing = ConcatenateKnowledge(keys=anchor if isinstance(anchor, list) else [anchor])
    pos_preprocessing = ConcatenateKnowledge(keys=pos if isinstance(pos, list) else [pos])
    neg_preprocessing = ConcatenateKnowledge(keys=neg if isinstance(neg, list) else [neg])

    dfs = {domain:df for domain, df in knowledge_base.knowledge.groupby('domain')}

    for domain in knowledge_base.domains:
        anchor_choices = np.unique([anchor_preprocessing(row) for ind, row in dfs[domain].iterrows()]).tolist()
        pos_choices = np.unique([pos_preprocessing(row) for ind, row in dfs[domain].iterrows()]).tolist()
        # pos_choices = np.unique([' '.join(sample) if isinstance(sample, np.ndarray) else sample for sample in dfs[domain][pos_field].values]).tolist()
        neg_choices = np.unique(
            [neg_preprocessing(row) for d in knowledge_base.domains for ind, row in dfs[d].iterrows() if d != domain]).tolist()

        n_negatives = min(n_negatives, len(neg_choices))

        for _ in range(n_samples_per_domain):

            samples.append(
                {
                    'anchor': random.choice(anchor_choices),
                    'pos': random.choice(pos_choices),
                    'neg': np.random.choice(neg_choices, n_negatives, replace=False).tolist()
                }
            )

    return samples


class DialogsDataset(Dataset):
    def __init__(self, dataroot: str, dataset: str, log_file: str = 'logs.json', label_file: str = 'labels.json',
                 selection_turns_only=False, preprocessing_fn=None):
        super(DialogsDataset, self).__init__()

        logs_path = os.path.join(dataroot, dataset, log_file)
        with open(logs_path, 'r') as f:
            self.dialogs = json.load(f)

        self.labels = None
        self.selection_turns_only = selection_turns_only
        labels_path = os.path.join(dataroot, dataset, label_file)
        if os.path.exists(labels_path):
            with open(labels_path, 'r') as f:
                self.labels = json.load(f)

            if selection_turns_only:
                self.dialogs, self.labels = self.filter_selection_turns()

        self.preprocessing_fn = preprocessing_fn

    def filter_selection_turns(self):

        assert self.labels is not None

        selection_dialogs = []
        selection_labels = []
        for dialog_context, label in zip(self.dialogs, self.labels):
            if label['target'] == True:
                selection_dialogs.append(dialog_context)

                # entity_id and doc_id are stored as integers in the labels, but are used as str in the knowledge base
                label = label['knowledge'][0]
                label['entity_id'] = str(label['entity_id'])
                label['doc_id'] = str(label['doc_id'])
                selection_labels.append(label)

        return selection_dialogs, selection_labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        if isinstance(item, int) or isinstance(item, np.int32):
            data = self.dialogs[item]
            label = None
            if self.labels is not None:
                label = self.labels[item]
        elif isinstance(item, slice):
            return [self[ii] for ii in range(*item.indices(len(self)))]
        elif isinstance(item, list):
            return [self[ii] for ii in item]
        else:
            raise IndexError

        if self.preprocessing_fn is not None:
            data = self.preprocessing_fn(data)

        return data, label

    def __iter__(self):
        for i in range(self.__len__()):
            yield self[i]



def sample_dialog_triplets(dialogs: DialogsDataset, knowledge_base: KnowledgeBase,
                           knowledge_preprocessing=None, n_samples=1000, n_negatives=1):
    # preparing corpus, pos and neg
    assert dialogs.selection_turns_only is True

    n_samples_per_domain = n_samples//len(knowledge_base.domains)
    samples = []
    dfs = {domain: df for domain, df in knowledge_base.knowledge.groupby('domain')}
    if knowledge_preprocessing is None:
        knowledge_preprocessing = ConcatenateKnowledge()
    for domain in knowledge_base.domains:

        mask_domain = [i for i, label in enumerate(dialogs.labels) if label['domain'] == domain]

        # implement random in-domain in-entity
        neg_domains = list(knowledge_base.domains)
        neg_domains.remove(domain)
        neg_choices = np.unique(
            [knowledge_preprocessing(row) for d in neg_domains for ind, row in dfs[d].iterrows()]).tolist()

        n_negatives = min(n_negatives, len(neg_choices))

        for _ in range(n_samples_per_domain):
            dialog, label = dialogs[random.choice(mask_domain)]
            true_domain = label['domain']
            entity_id = str(label['entity_id'])
            doc_id = str(label['doc_id'])
            pos = knowledge_preprocessing(knowledge_base[true_domain, entity_id, doc_id])
            neg = np.random.choice(neg_choices, n_negatives, replace=False).tolist()

            samples.append(
                {
                    'anchor': dialog,
                    'pos': pos,
                    'neg': neg
                }
            )

    return samples

if __name__ == '__main__':
    kb = KnowledgeBase('./DSTC9/data')
    from utils import *
    d_data = DialogsDataset('./DSTC9/data', 'train', selection_turns_only=True, preprocessing_fn=ConcatenateDialogContext())

    x = sample_dialog_triplets(d_data, kb)

    from utils import print_triplet

    for triplet in x[:10]:
        anchor = triplet['anchor']
        pos = triplet['pos']
        neg = triplet['neg'][0]
        print_triplet(anchor, pos, neg)