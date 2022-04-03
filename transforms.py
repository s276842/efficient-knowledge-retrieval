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
        question = target['question']
        answer = target['answer']

        x = []
        if self.add_domain:
            x.append(domain)
            x.append(self.sep_token)

        if self.add_entity_name and entity_name is not None: #None to consider enity_id '*'
            x.append(entity_name)
            x.append(self.sep_token)

        x.append(question)
        if self.add_answer:
            if self.separate_question_answer:
                x.append(self.sep_token)

            x.append(answer)
        try:
            return ' '.join(x)
        except:
            pass


class ConcatenateDialogContext:
    def __init__(self, user_special_token='', agent_special_token = '', reverse=False, limit=None):
        self.speaker_special_tokens = {'U': user_special_token, 'S':agent_special_token}
        self.reverse = reverse
        self.limit = limit
        self.f = lambda utterance: (self.speaker_special_tokens[utterance['speaker']], utterance['text'])

    def __call__(self, dialog_context):
        # try:
        #     iter(item)
        #     dialog_context = item
        # except:
        #     dialog_context = [item]


        if self.reverse is False:
            r = flatmap(self.f, dialog_context[:self.limit])
        else:
            r = flatmap(self.f, dialog_context[::-1][:self.limit])
        return ' '.join(r)


if __name__ == '__main__':
    from transformers import AutoTokenizer, AutoModel
    import torch

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model_path = 'distilroberta-base'
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path).to(device)

    from models import Encoder
    from transforms import ConcatenateKnowledge, CLSExtractor, PoolerExtractor
    from dataset import VectorizedKnowledgeBase, KnowledgeBase
    import torch

    k_preprocessing = ConcatenateKnowledge(sep_token=tokenizer.sep_token, add_domain=True)
    k_postprocessing = PoolerExtractor()
    k_encoder = Encoder(model, tokenizer, k_preprocessing, k_postprocessing)
    k_base = KnowledgeBase("DSTC9/data", "knowledge.json")
    k_preprocessing = ConcatenateKnowledge(sep_token=tokenizer.sep_token, add_domain=True, add_entity_name=True)
    knowledge_docs = [k_preprocessing(doc) for doc in k_base]

