import itertools

def flatmap(func, *iterable):
    return list(itertools.chain.from_iterable(map(func, *iterable)))


class CLSExtractor:
    def __call__(self, out):
        return out.last_hidden_state[:, 0]

class PoolerExtractor:
    def __call__(self, out):
        return out['pooler_output']

class ConcatenateKnowledge:
    def __init__(self, sep_token=' ', add_domain=False, add_entity_name=False, add_answer=True, separate_question_answer=False):
        self.sep_token = ' ' + sep_token + ' ' if sep_token != ' ' else sep_token
        self.add_domain = add_domain
        self.add_entity_name = add_entity_name
        self.add_answer = add_answer
        self.separate_question_answer = separate_question_answer

    def __call__(self, target):
        domain = target['domain']
        entity_name = target['entity_name']
        doc = target['doc']
        question, answer = doc['title'], doc['body']

        x = []
        if self.add_domain:
            x.append(domain)
            x.append(self.sep_token)

        if self.add_entity_name:
            x.append(entity_name)
            x.append(self.sep_token)

        x.append(question)
        if self.add_answer:
            if self.separate_question_answer:
                x.append(self.sep_token)

            x.append(answer)

        return ' '.join(x)


class ConcatenateDialogContext:
    def __init__(self, user_special_token='', agent_special_token = ''):
        self.speaker_special_tokens = {'U': user_special_token, 'S':agent_special_token}

    def __call__(self, dialog_context):
        # try:
        #     iter(item)
        #     dialog_context = item
        # except:
        #     dialog_context = [item]


        f = lambda utterance: (self.speaker_special_tokens[utterance['speaker']], utterance['text'])
        r = flatmap(f, dialog_context)
        return ' '.join(r)