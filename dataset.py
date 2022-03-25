from torch.utils.data import Dataset
from scripts.knowledge_reader import KnowledgeReader
from scripts.dataset_walker import DatasetWalker
import random
import torch

#todo cambiare transforms
class KnowledgeBase(KnowledgeReader):
    def __init__(self, dataroot, knowledge_file):
        super(KnowledgeBase, self).__init__(dataroot, knowledge_file)
        self.doc_list = [{'domain':doc['domain'],
                           'entity_id':doc['entity_id'],
                           'doc_id':doc['doc_id']}
                           for doc in self.get_doc_list()]

    def __len__(self):
        return len(self.doc_list)

    def __getitem__(self, item):
        if type(item) is int:
            if item < 0 or item >= len(self.doc_list):
                raise ValueError("index out of bounds: %d" % item)

            item = self.doc_list[item]
        elif isinstance(item, slice):
            # Get the start, stop, and step from the slice
            return [self[ii] for ii in range(*item.indices(len(self)))]
        res = self.get_doc(**item)
        return res

    def __iter__(self):
        for item in self.doc_list:
            yield self.__getitem__(item)



class VectorizedKnowledgeBase(KnowledgeBase):
    def __init__(self, encoder, *args):
        super(VectorizedKnowledgeBase, self).__init__(*args)
        self.encoder = encoder
        self.vectorize()

    def vectorize(self):
        self.knowledge_vectors = torch.cat([self.encoder(self.__getitem__(doc_ids)) for doc_ids in self.doc_list[:10]])



class DSTCDataset(DatasetWalker, Dataset):
    def __init__(self, dataset, dataroot, knowledge_base, labels=False, labels_file=None, encoder=None):
        super(DSTCDataset, self).__init__(dataset, dataroot, labels, labels_file)
        self.encoder = encoder
        self.knowledge_base = knowledge_base

        if self.labels is not None:
            self.ids = [idx for idx, val in enumerate(iter(self)) if val[1]['target'] is True]

        # knowledge_labels = []
        # logs = []
        # for dialog_context, label in iter(self):
        #     if labels is False:
        #         logs.append(dialog_context)
        #     else:
        #         if label['target'] is True:
        #             knowledge_labels.append(label['knowledge'][0])
        #             logs.append(dialog_context)
        #
        # self.logs = logs
        # self.labels = knowledge_labels

    def __len__(self):
        return len(self.logs)

    def __getitem__(self, idx):
        if self.labels is not None:
            i = self.ids[idx]
        else:
            i = idx

        data = self.logs[i]

        if self.encoder is not None:
            data = self.encoder(data)

        if len(self.labels) > 0:
            label = self.labels[i]['knowledge']
            return (data, label)

        else:
            return (data,)

#
#             domain = label['domain']
#             entity_id = label['entity_id']
#             doc_id = label['doc_id']
#             doc = self.knowledge_walker.get_doc(domain, entity_id, doc_id)
#             entity_name = doc['entity_name']
#             doc = doc['doc']
#
#             if self.sampling_method == 'domain':
#                 neg_domain = self.__get_random_domain(domain_to_exclude=doc_id)
#                 neg_sample = (neg_domain,)
#                 pos_sample = (domain,)
#             elif self.sampling_method == 'entity':
#                 neg_entity = self.__get_random_entity(domain, entity_to_exclude=entity_id)
#                 neg_entity_name = neg_entity['name']
#                 neg_sample = (domain, neg_entity_name)
#                 pos_sample = (domain, entity_name)
#             elif self.sampling_method == 'document':
#                 neg_doc = self.__get_random_document(domain, entity_id, document_to_exclude=doc_id)['doc']
#                 neg_sample = (domain, entity_name, *neg_doc.values())
#                 pos_sample = (domain, entity_name, *doc.values())
#
#
#             if self.target_transform is not None:
#                 pos_sample = self.target_transform(pos_sample)
#                 neg_sample = self.target_transform(neg_sample)
#
#             return (anchor, pos_sample, neg_sample)

# class DSTCDataset(Dataset):
#     def __init__(self, dataset_walker, knowledge_walker, sampling_method='document', data_transform=None, target_transform=None):
#         self.data = []
#         self.labels = []
#         self.knowledge_walker = knowledge_walker
#         self.data_transform = data_transform
#         self.target_transform = target_transform
#         self.sampling_method = sampling_method
#
#         for dialog_context, label in dataset_walker:
#             if label is None:
#                 self.data.append(dialog_context)
#             else:
#                 if label['target'] is True:
#                     self.labels.append(label['knowledge'][0])
#                     self.data.append(dialog_context)
#
#     def __len__(self):
#         return len(self.data)
#
#     def __get_random_domain(self, domain_to_exclude=None):
#         domains = self.knowledge_walker.get_domain_list()
#         try:
#             domains.remove(domain_to_exclude)
#         except:
#             pass
#         return random.choice(domains)
#
#
#     def __get_random_entity(self, domain=None, entity_to_exclude=None):
#         if domain is None:
#             domain = self.__get_random_domain()
#
#         entities = self.knowledge_walker.get_entity_list(domain)
#         try:
#             entities.remove(entity_to_exclude)
#         except:
#             pass
#         return random.choice(entities)
#
#     # def get_document(self, domain=None, entity_id=None, doc_id=None, domain_to_exclude=None, entity_to_exclude=None, document_to_exclude=None):
#     #     if domain is None:
#     #         domain = self.__get_random_domain(domain_to_exclude=domain_to_exclude)
#     #
#     #         if entity_id is None:
#     #             entity_id = self.__get_random_entity(domain)
#     #
#     #             if doc_id is None:
#
#
#     def __get_random_document(self, domain, entity, document_to_exclude=None):
#         documents = self.knowledge_walker.get_doc_list(domain, entity)
#         doc = random.choice(documents)
#         while doc['doc_id'] == document_to_exclude:
#             doc = random.choice(documents)
#         return doc
#
#
#     def __getitem__(self, idx):
#
#         anchor = self.data[idx]
#
#         if self.data_transform is not None:
#             anchor = self.data_transform(anchor)
#
#         if len(self.labels) > 0:
#             label = self.labels[idx]
#
#             domain = label['domain']
#             entity_id = label['entity_id']
#             doc_id = label['doc_id']
#             doc = self.knowledge_walker.get_doc(domain, entity_id, doc_id)
#             entity_name = doc['entity_name']
#             doc = doc['doc']
#
#             if self.sampling_method == 'domain':
#                 neg_domain = self.__get_random_domain(domain_to_exclude=doc_id)
#                 neg_sample = (neg_domain,)
#                 pos_sample = (domain,)
#             elif self.sampling_method == 'entity':
#                 neg_entity = self.__get_random_entity(domain, entity_to_exclude=entity_id)
#                 neg_entity_name = neg_entity['name']
#                 neg_sample = (domain, neg_entity_name)
#                 pos_sample = (domain, entity_name)
#             elif self.sampling_method == 'document':
#                 neg_doc = self.__get_random_document(domain, entity_id, document_to_exclude=doc_id)['doc']
#                 neg_sample = (domain, entity_name, *neg_doc.values())
#                 pos_sample = (domain, entity_name, *doc.values())
#
#
#             if self.target_transform is not None:
#                 pos_sample = self.target_transform(pos_sample)
#                 neg_sample = self.target_transform(neg_sample)
#
#             return (anchor, pos_sample, neg_sample)
#
#
#     def set_sampling_method(self, sampling_method):
#         self.sampling_method = sampling_method


if __name__ == '__main__':

    d_train_dataset = DSTCDataset('train', 'DSTC9/data/', labels=True)

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