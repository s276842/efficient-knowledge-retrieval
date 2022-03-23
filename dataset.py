from torch.utils.data import Dataset
import random


class DSTCDataset(Dataset):
    def __init__(self, dataset_walker, knowledge_walker, data_transform=None, target_transform=None):
        self.data = []
        self.labels = []
        self.knowledge_walker = knowledge_walker
        self.data_transform = data_transform
        self.target_transform = target_transform
        self.sampling_method = 'domain'#self.set_sampling_method('domain')

        for dialog_context, label in dataset_walker:
            if label is None:
                self.data.append(dialog_context)
            else:
                if label['target'] is True:
                    self.labels.append(label['knowledge'][0])
                    self.data.append(dialog_context)

    def __len__(self):
        return len(self.data)

    def __get_random_domain(self, domain_to_exclude=None):
        domains = self.knowledge_walker.get_domain_list()
        try:
            domains.remove(domain_to_exclude)
        except:
            pass
        return random.choice(domains)


    def __get_random_entity(self, domain, entity_to_exclude=None):
        entities = self.knowledge_walker.get_entity_list(domain)
        try:
            entities.remove(entity_to_exclude)
        except:
            pass
        return random.choice(entities)


    def __get_random_document(self, domain, entity, document_to_exclude=None):
        documents = self.knowledge_walker.get_doc_list(domain, entity)
        doc = random.choice(documents)
        while doc['doc_id'] == document_to_exclude:
            doc = random.choice(documents)
        return doc


    def __getitem__(self, idx):

        anchor = self.data[idx]

        if self.data_transform is not None:
            anchor = self.data_transform(anchor)

        if len(self.labels) > 0:
            label = self.labels[idx]

            domain = label['domain']
            entity_id = label['entity_id']
            doc_id = label['doc_id']
            doc = self.knowledge_walker.get_doc(domain, entity_id, doc_id)
            entity_name = doc['entity_name']
            doc = doc['doc']

            if self.sampling_method == 'domain':
                neg_domain = self.__get_random_domain(domain_to_exclude=doc_id)
                neg_sample = (neg_domain,)
                pos_sample = (domain,)
            elif self.sampling_method == 'entity':
                neg_entity = self.__get_random_entity(domain, entity_to_exclude=entity_id)
                neg_entity_name = neg_entity['name']
                neg_sample = (domain, neg_entity_name)
                pos_sample = (domain, entity_name)
            elif self.sampling_method == 'document':
                neg_doc = self.__get_random_document(domain, entity_id, document_to_exclude=doc_id)['doc']
                neg_sample = (domain, entity_name, neg_doc)
                pos_sample = (domain, entity_name, doc)


            if self.target_transform is not None:
                pos_sample = self.target_transform(pos_sample)
                neg_sample = self.target_transform(neg_sample)

            return (anchor, pos_sample, neg_sample)


    def set_sampling_method(self, sampling_method):
        self.sampling_method = sampling_method


if __name__ == '__main__':
    from scripts.dataset_walker import DatasetWalker
    from scripts.knowledge_reader import KnowledgeReader

    dw = DatasetWalker('train', 'DSTC9/data', labels=True)
    kr = KnowledgeReader("DSTC9/data", "knowledge.json")



    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('distilroberta-base')
    ds = DSTCDataset(dw, kr)
    print(ds[0])

    print(ds[1])