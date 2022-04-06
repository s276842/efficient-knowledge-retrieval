import json


class KnowledgeBase:
    def __init__(self, knowledge_file='knowledge.json', knowledge_transform=None):

        self.knowledge_transform = knowledge_transform

        # load knowledge database as dictionary
        with open(knowledge_file, 'r') as f:
            self.knowledge = json.load(f)

        self.ids = []

        for domain in self.knowledge:
            d_dict = self.knowledge[domain]
            for entity_id in d_dict:
                for doc_id in d_dict[entity_id]['docs']:
                    self.ids.append((domain, entity_id, doc_id))

    def getinfo(self, item):
        if isinstance(item, int):
            domain, entity_id, doc_id = self.ids[item]
            return {'domain': domain, 'entity_id': entity_id, 'doc_id': doc_id}

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
            entity_name = e_dict['name'] if e_dict['name'] is not None else domain
            question, answer = e_dict['docs'][doc_id].values()
            r = {'domain': domain, 'entity_id': entity_id, "entity_name": entity_name, 'doc_id': doc_id,
                 'question': question, 'answer': answer}
        elif isinstance(item, str):
            return self.knowledge[item]
        else:
            raise IndexError

        if self.knowledge_transform is not None:
            r = self.knowledge_transform(r)

        return r

    def __len__(self):
        return len(self.ids)

    def __iter__(self):
        for id in self.ids:
            yield self[id]


class DialogContext:

    def __init__(self, log_file_path, label_file_path=None, selection_turns_only=False, data_transform=None):

        with open(log_file_path, 'r') as f:
            self.logs = json.load(f)

        self.labels = None
        if label_file_path is not None:
            with open(label_file_path, 'r') as f:
                self.labels = json.load(f)

            if selection_turns_only:
                self.__filter_turns()

        self.data_transform = data_transform

    def __filter_turns(self):
        s_dialogs = []
        s_labels = []
        for dialog_context, label in zip(self.logs, self.labels):
            if label['target'] == True:
                s_dialogs.append(dialog_context)

                label = label['knowledge'][0]
                label['entity_id'] = str(label['entity_id'])
                label['doc_id'] = str(label['doc_id'])
                s_labels.append(label)

        self.logs = s_dialogs
        self.labels = s_labels

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

        if self.data_transform is not None:
            data = self.data_transform(data)

        return data, label

    def __iter__(self):
        for i in range(self.__len__()):
            yield self[i]
