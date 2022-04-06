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
    def __init__(self, sep_token=' ', keys=['domain', 'entity_name', 'question', 'answer']):
        self.sep_token = ' ' + sep_token + ' ' if sep_token != ' ' else sep_token
        self.keys = keys

    def __call__(self, knowledge_snippet):
        info = [knowledge_snippet.get(key, '') for key in self.keys]
        return self.sep_token.join(info)


class ConcatenateDialogContext:
    def __init__(self, user_special_token='', agent_special_token = '', reverse=False, limit=0):
        self.speaker_special_tokens = {'U': user_special_token, 'S':agent_special_token}
        self.reverse = reverse
        self.limit = limit
        self.f = lambda utterance: (self.speaker_special_tokens[utterance['speaker']], utterance['text'])

    def __call__(self, dialog_context):
        turns = dialog_context[-self.limit:]
        if self.reverse is True:
            turns = turns[::-1]

        r = flatmap(self.f, turns)
        return ' '.join(r)

