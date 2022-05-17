import json
import os.path
import random
from torch.utils.data import Dataset
from typing import Callable
import pandas as pd
import numpy as np
from utils import *
from tqdm import tqdm

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

        else:
            return self.knowledge.iloc[item]



def sample_kb_triplets(knowledge_base, anchor_fields='domain', pos_fields=['question', 'answer'],
                       neg_fields=['question', 'answer'], n_samples=1000, n_negatives=1, domains=None):
    # preparing corpus, pos_fields and neg_fields
    n_samples_per_domain = n_samples//len(knowledge_base.domains)
    samples = []

    anchor_preprocessing = ConcatenateKnowledge(keys=anchor_fields if isinstance(anchor_fields, list) else [anchor_fields])
    pos_preprocessing = ConcatenateKnowledge(keys=pos_fields if isinstance(pos_fields, list) else [pos_fields])
    neg_preprocessing = ConcatenateKnowledge(keys=neg_fields if isinstance(neg_fields, list) else [neg_fields])

    dfs = {domain:df for domain, df in knowledge_base.knowledge.groupby('domain')}

    if domains is None:
        domains = knowledge_base.domains

    for domain in domains:
        anchor_choices = np.unique([anchor_preprocessing(row) for ind, row in dfs[domain].iterrows()]).tolist()
        pos_choices = np.unique([pos_preprocessing(row) for ind, row in dfs[domain].iterrows()]).tolist()
        # pos_choices = np.unique([' '.join(sample) if isinstance(sample, np.ndarray) else sample for sample in dfs[domain][pos_field].values]).tolist()
        neg_choices = np.unique(
            [neg_preprocessing(row) for d in knowledge_base.domains for ind, row in dfs[d].iterrows() if d != domain]).tolist()

        n_negatives = min(n_negatives, len(neg_choices))

        for _ in range(n_samples_per_domain):

            samples.append((random.choice(anchor_choices),random.choice(pos_choices), np.random.choice(neg_choices, replace=False)))

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
        elif isinstance(item, list) or isinstance(item, np.ndarray):
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
                           fields=['domain', 'name', 'question', 'answer'], n_samples=1000, n_negatives=1, method='random'):
    # preparing corpus, pos and neg
    assert dialogs.selection_turns_only is True

    #todo correct n_samples as the min between n_samples and the len of the possible samples
    n_samples_per_domain = n_samples//len(knowledge_base.domains)
    knowledge_preprocessing = ConcatenateKnowledge(keys=fields)

    triplets = []
    for domain in tqdm(knowledge_base.domains):
        mask_domain = [i for i, label in enumerate(dialogs.labels) if label['domain'] == domain]
        dialog_samples = dialogs[np.random.choice(mask_domain, n_samples_per_domain)]


        if method == 'random':
            df_samples = knowledge_base.knowledge[knowledge_base.knowledge.domain != domain]
            negatives = df_samples.loc[np.random.choice(df_samples.index, n_samples_per_domain)]

            for (dialog, label), (_, neg) in zip(dialog_samples, negatives.iterrows()):
                true_domain = label['domain']
                entity_id = str(label['entity_id'])
                doc_id = str(label['doc_id'])
                pos = knowledge_preprocessing(knowledge_base[true_domain, entity_id, doc_id])
                neg = knowledge_preprocessing(neg)
                triplets.append((dialog, pos, neg))

        else:
            kb_domain = knowledge_base.knowledge[knowledge_base.knowledge.domain == domain]

            for dialog, label in dialog_samples:

                true_domain = label['domain']
                entity_id = str(label['entity_id'])
                doc_id = str(label['doc_id'])
                pos = knowledge_preprocessing(knowledge_base[true_domain, entity_id, doc_id])

                if method == 'in-domain':
                    if entity_id == '*':
                        continue
                    df_samples = kb_domain[kb_domain.entity_id != entity_id]
                elif method == 'in-entity':
                    df_samples = kb_domain[kb_domain.entity_id == entity_id]

                df_samples = df_samples[fields].drop_duplicates()
                neg = knowledge_preprocessing(df_samples.loc[np.random.choice(df_samples.index)])

                triplets.append((dialog, pos, neg))

    return triplets


def test_break_points(ds, break_points):
    for i in range(1, len(break_points), 2):
        start_ind = break_points[i - 1] + 1
        end_ind = break_points[i]
        first_user_turn = ds.dialogs[end_ind][0]['text']
        print(f'checking from {start_ind}-{end_ind}')
        for ind in range(start_ind, end_ind):
            d = ds.dialogs[ind][0]['text']
            assert d == first_user_turn

class CondensedDialogsDataset(DialogsDataset):
    def __init__(self, *args, **kwargs):

        super(CondensedDialogsDataset, self).__init__(selection_turns_only=False, *args, **kwargs)
        lengths = [len(sample) for sample in self.dialogs]

        if args[1] != 'test' and kwargs.get('dataset') != 'test':
            differences = np.diff(lengths)
            break_points = np.argwhere(differences <= 0).flatten()

            test_break_points(self, break_points)

            self.dialogs = [dialog_to_turns(self.dialogs[i]) for i in break_points]
            labels = []
            turn_labels = []
            for i, diff in enumerate(differences):
                if diff == 0:
                    continue
                elif diff < 0 or i == 0:
                    labels.append(turn_labels)
                    turn_labels = []

                turn_labels += [{'target': False}] * ((lengths[i]) // 2 - 1)
                turn_labels.append(self.labels[i + 1])
            labels.pop(0)
            self.labels = labels
        else:
            break_points = range(len(self.dialogs))
            test_break_points(self, break_points)

            self.dialogs = [dialog_to_turns(self.dialogs[i]) for i in break_points]
            labels = []
            for i, l in enumerate(lengths):
                turn_labels = []
                turn_labels += [{'target': False}] * ((lengths[i]) // 2)
                turn_labels.append(self.labels[i])
                labels.append(turn_labels)
            self.labels = labels




if __name__ == '__main__':
    ds = CondensedDialogsDataset('./DSTC9/data_eval', 'test')

    for dialog, label in ds:
        try:
            assert len(dialog) == len(label)
        except:
            pass


