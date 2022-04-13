import json
import os.path
import random
from torch.utils.data import Dataset
# extract information (keys) from a knowledge snippet and concatenate using the separator token
SPECIAL_TOKENS = {'domain':'<D>', 'name': '<E>', 'question': '<Q>', 'answer': '<A>', 'U': '<U>', 'S': '<S>'}



class ConcatenateKnowledge:
    def __init__(self, keys=['domain', 'name', 'question', 'answer'], sep_tokens=SPECIAL_TOKENS):
        self.keys = keys
        self.sep_tokens = sep_tokens

        # if both domain and entity_name are required, we need to do an additional check
        # for those domains that does not have entities, such as train and taxi
        if 'name' in self.keys:
            self.check_entity_name = True
            self.alt_keys = ['domain'] + [key for key in keys if key != 'domain' and key!= 'name']
        else:
            self.check_entity_name = False

    def __call__(self, knowledge_snippet):

        # if the current knowledge snippet belongs to a domain without entities
        if self.check_entity_name and knowledge_snippet['name'] == None:
            keys = self.alt_keys
        else:
            keys = self.keys

        pairs_sep_key = [f'{self.sep_tokens[key]} {knowledge_snippet[key]} ' for key in keys]
        return ''.join(pairs_sep_key)




class KnowledgeBase:
    def __init__(self, dataroot: str, knowledge_file: str = 'knowledge.json', preprocessing_fn=None, aggregate_additional_info=False):

        # load knowledge database as dictionary. The DSTC9 track1 / DSTC10 track2 kb is organized in a
        # hierarchical fashion: domain -> entity -> knowledge snippets
        knowledge_path = os.path.join(dataroot, knowledge_file)

        with open(knowledge_path, 'r') as f:
            self.knowledge = json.load(f)

        # iterate through the dictionary indexing in a list the tuple of keys (domain, entity_id, doc_id)
        self.ids = []
        self.entity_name_dict = {}
        self.ranges = {}
        i = 0
        for domain in self.knowledge:
            ind_start_domain = i
            d_dict = self.knowledge[domain]
            for entity_id in d_dict:
                ind_start_entity = i
                e_dict = d_dict[entity_id]
                entity_name = e_dict['name']
                if entity_name is not None:
                    self.entity_name_dict[entity_name] = {'domain': domain, 'entity_id':entity_id}

                for doc_id in e_dict['docs']:
                    self.ids.append((domain, entity_id, doc_id))
                    i+= 1

                self.ranges[(domain, entity_id)] = (ind_start_entity, i-1)

            self.ranges[domain] = (ind_start_domain, i-1)


        self.preprocessing_fn = preprocessing_fn
        self.aggregate_additional_info = aggregate_additional_info

    def get_label(self, index: int):
        if isinstance(index, int):
            domain, entity_id, doc_id = self.ids[index]
            return {'domain': domain, 'entity_id': entity_id, 'doc_id': doc_id}


    def keys(self):
        return self.knowledge.keys()


    def __getitem__(self, item):
        if isinstance(item, slice):
            return [self[ii] for ii in range(*item.indices(len(self)))]
        elif isinstance(item, list):
            return [self[ii] for ii in item]
        elif isinstance(item, int):
            id = self.ids[item]
            return self[id]
        elif isinstance(item, tuple):
            domain, entity_id, doc_id = item
            if not isinstance(entity_id, str):
                entity_id = str(entity_id)
            if not isinstance(doc_id, str):
                doc_id = str(doc_id)

            e_dict = self.knowledge[domain][entity_id]
            question, answer = e_dict['docs'][doc_id].values()
            knowledge_snippet = {'domain': domain, 'entity_id': entity_id, 'doc_id': doc_id,
                 'question': question, 'answer': answer}


            # todo check if additional info (city) can be useful for classification of the entities
            if self.aggregate_additional_info:
                s = list()
                for key in e_dict:
                    if key != 'docs':
                        s += ([val.strip().lower() for val in e_dict[key].split()])
                knowledge_snippet['name'] = ' '.join(set(s))

            else:
                for key in e_dict:
                    if key != 'docs':
                        knowledge_snippet[key] = e_dict[key]

        elif isinstance(item, str):
            return self.knowledge[item]
        else:
            raise IndexError

        if self.preprocessing_fn is not None:
            knowledge_snippet = self.preprocessing_fn(knowledge_snippet)

        return knowledge_snippet

    def __len__(self):
        return len(self.ids)

    def __iter__(self):
        for id in self.ids:
            yield self[id]







class DialogContext:

    def __init__(self,
                 dataroot: str,
                 dataset: str,
                 log_file: str = 'logs.json',
                 label_file: str = 'labels.json',
                 selection_turns_only=False,
                 preprocessing_fn=None):

        logs_path = os.path.join(dataroot, dataset, log_file)
        with open(logs_path, 'r') as f:
            self.dialogs = json.load(f)

        self.labels = None
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
        if isinstance(item, int):
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



def random_exclusion(start, stop, excluded) -> int:
    """Function for getting a random number with some numbers excluded"""
    excluded = set(excluded)
    value = random.randint(start, stop - len(excluded)) # Or you could use randrange
    for exclusion in tuple(excluded):
        if value < exclusion:
            break
        value += 1
    return value

class NegativeKnowledgeSampler():
    def __init__(self, knowledge_base: KnowledgeBase):
        self.knowledge_base = knowledge_base
        pass

    def sample_domain(self, domain):
        ind_min, ind_max = self.knowledge_base.ranges[domain]
        l = len(self.knowledge_base)
        i = random_exclusion(0, l-1, range(ind_min, ind_max+1))
        print(i)
        return kb[i]['domain']

import random
def sample(min, max, exclude_min, exclude_max):
    remove = exclude_max - exclude_min
    new_max = max - remove

    if min < new_max:
        id = random.randint(min, new_max)
    else:
        return None

    if id < exclude_min:
        return id
    else:
        return (id - exclude_min) + 1+ exclude_max


def test_sampling():
    for _ in range(100000):
        m = random.randint(0, 1000)
        M = random.randint(m, m+999)
        exclude_min = random.randint(m, M)
        exclude_max = random.randint(exclude_min, M)

        id = sample(m, M , exclude_min, exclude_max)
        try:
            assert id >= m
        except:
            print(f"not {id} >= {m}", id, m)
            raise AssertionError

        try:
            assert id < exclude_min
        except:
            print(f"not {id} < {exclude_min}", id, exclude_min)
            raise AssertionError

        try:
            assert id > exclude_max
        except:
            print(f"not {id} > {exclude_max}", id, exclude_max)
            raise AssertionError


        try:
            assert id <= M
        except:
            print(f"not {id} <= {M}", id, M)
            raise AssertionError



class TripletSelectionDataset(Dataset):

    def __init__(self, dialog_context_dataset: DialogContext, knowledge_base: KnowledgeBase):
        self.data = dialog_context_dataset
        self.k_base = knowledge_base
        self.selection_method = self.select_negative_domain
        self.selection_filter = 'random'

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        if isinstance(item, int):
            anchor, label = self.data[item]
            pos, neg = self.selection_method(**label)
            return {'anchor': anchor, 'pos': pos, 'neg': neg}
        elif isinstance(item, slice):
            return [self[ii] for ii in range(*item.indices(len(self)))]
        elif isinstance(item, list):
            return [self[ii] for ii in item]
        else:
            raise IndexError

    def set_selection_method(self, method='domain'):
        self.selection_method = self.__getattribute__('select_negative_' + method)

    def set_selection_filter(self, filter='random'):
        self.selection_filter = filter

    def select_negative_domain(self, domain, entity_id, doc_id):
        domains = list(self.k_base.knowledge.keys())
        domains.remove(domain)
        return {'domain': domain}, {'domain': random.choice(domains)}


    def select_negative_entity(self, domain, entity_id, doc_id):
        # select randomly across all domains
        if self.selection_filter == 'random':
            # select random domain and corresponding possible entities
            _, neg = self.select_negative_domain(domain, entity_id, doc_id)
            neg_domain = neg['domain']
            entities = list(self.k_base[neg_domain].keys())

        # select entity only among the ones in the same domain
        elif self.selection_filter == 'in-domain':
            if entity_id == '*':
                # return self.select_negative_document(domain, entity_id, doc_id)
                self.set_selection_filter('random')
                r = self.select_negative_entity(domain, entity_id, doc_id)
                self.set_selection_filter('in-domain')
                return r

            neg_domain = domain
            entities = list(self.k_base[neg_domain].keys())
            entities.remove(entity_id)  # remove the same entity

        neg_entity_id = random.choice(entities)
        neg_entity_name = self.k_base[neg_domain][neg_entity_id]['name']
        neg_entity_name = neg_entity_name if neg_entity_name is not None else neg_domain

        pos_entity_name = self.k_base[domain][entity_id]['name']
        pos_entity_name = pos_entity_name if pos_entity_name is not None else domain

        return {'domain': domain, 'entity_id': entity_id, 'name': pos_entity_name}, \
               {'domain': neg_domain, 'entity_id': neg_entity_id, 'name': neg_entity_name}

    def select_negative_document(self, domain, entity_id, doc_id):

        if self.selection_filter == 'random':
            pos, neg = self.select_negative_entity(domain, entity_id, doc_id)
            neg_domain = neg['domain']
            neg_entity_id = neg['entity_id']
            docs = list(self.k_base[neg_domain][neg_entity_id]['docs'].keys())
        elif self.selection_filter == 'in-domain' and entity_id != '*':
            neg_domain = domain
            entities = list(self.k_base[domain].keys())
            entities.remove(entity_id)
            neg_entity_id = random.choice(entities)
            docs = list(self.k_base[neg_domain][neg_entity_id]['docs'].keys())
        else:
            neg_domain = domain
            neg_entity_id = entity_id
            docs = list(self.k_base[neg_domain][neg_entity_id]['docs'].keys())
            docs.remove(doc_id)

        neg_doc_id = random.choice(docs)

        return self.k_base[(domain, entity_id, doc_id)], self.k_base[(neg_domain, neg_entity_id, neg_doc_id)]






if __name__ == '__main__':
    dataroot = './DSTC9/data/'
    dataset = 'test'

    kb = KnowledgeBase(dataroot)

    import random
    for i in range(10):
        print(kb[random.randint(0, len(kb))])

    # from transforms import ConcatenateDialogContext
    #
    # dc = DialogContext(dataroot, dataset, selection_turns_only=True, preprocessing_fn=None)
    #
    # train_dataset = TripletSelectionDataset(dc, kb)
    # import numpy as np
    #
    # d_pre = ConcatenateDialogContext(limit=1)
    # for method in ['domain', 'entity', 'document']:
    #     if method == 'domain':
    #         filters = ['']
    #         k_pre = ConcatenateKnowledge('</s>', keys=['domain'])
    #     elif method == 'entity':
    #         filters = ['random', 'in-domain']
    #         k_pre = ConcatenateKnowledge('</s>', keys=['domain', 'name'])
    #     else:
    #         filters = ['random', 'in-domain', 'in-entity']
    #         k_pre = ConcatenateKnowledge('</s>', keys=['domain', 'name', 'question', 'answer'])
    #
    #     for filter in filters:
    #         train_dataset.set_selection_method(method)
    #         train_dataset.set_selection_filter(filter)
    #         print(f'========= {method}-{filter} ==========')
    #         for i in range(10):
    #             anchor, pos, neg = train_dataset[np.random.randint(len(train_dataset))].values()
    #
    #             print(f"Anchor : {d_pre(anchor)}")
    #             print(f"pos : {k_pre(pos)}")
    #             print(f"neg : {k_pre(neg)}")
    #             print()
    #
    # # import json
    # # x = json.load(open('./out.tmp.json'))
    # r = []
    # for label in x:
    #     if label['target'] == False:
    #         r.append(label)
    #     else:
    #         label['knowledge'] = [{'domain':x['domain'}
