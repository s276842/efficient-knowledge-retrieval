import sys
import pandas as pd
from dataset import KnowledgeBase
import itertools
import json
import os

def flatmap(func, *iterable):
    return list(itertools.chain.from_iterable(map(func, *iterable)))


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



SPECIAL_TOKENS = {'domain':'<D>', 'name': '<E>', 'question': '<Q>', 'answer': '<A>', 'U': '<U>', 'S': '<S>'}
NO_SPECIAL_TOKENS = {'domain':'', 'name': '', 'question': '', 'answer': '', 'U': '', 'S': ''}

class ConcatenateKnowledge:
    def __init__(self, keys=['domain', 'name', 'question', 'answer'], sep_tokens=NO_SPECIAL_TOKENS, replace_domain_name=False):
        self.keys = keys
        self.sep_tokens = sep_tokens
        self.replace_domain_name = replace_domain_name
        # if both domain and entity_name are required, we need to do an additional check
        # for those domains that does not have entities, such as train and taxi
        if 'name' in self.keys:
            self.check_entity_name = True
            self.alt_keys = ['domain-entity'] + [key for key in keys if key != 'domain' and key!= 'name']
            self.sep_tokens['domain-entity'] = self.sep_tokens['name']
        else:
            self.check_entity_name = False

    def __call__(self, knowledge_snippet):

        # if the current knowledge snippet belongs to a domain without entities
        #     keys = self.keys
        #     if self.replace_domain_name:
        #         ks['name'] = ks['name'].replace(ks['domain'], '')
        if isinstance(knowledge_snippet, pd.core.frame.DataFrame):
            pairs_sep_key = [f'{self.sep_tokens[key]} {knowledge_snippet.get(key).item()} ' if knowledge_snippet.get(key).item() is not None else '' for key in self.keys]
        else:
            pairs_sep_key = [f'{self.sep_tokens[key]} {knowledge_snippet.get(key)} ' if knowledge_snippet.get(key) is not None else '' for key in self.keys]
        return ''.join(pairs_sep_key).strip()


# ==============================================================


def check_domain(true, pred):
    return true['domain'] == pred['domain']

def check_entity(true, pred):
    return str(true['entity_id']) == str(pred['entity_id']) and check_domain(true, pred)

def check_name(true, pred):
    return str(true['entity_id']) == str(pred['entity_id']) and check_domain(true, pred)

def check_doc(true, pred):
    return int(true['doc_id']) == int(pred['doc_id']) and check_entity(true, pred)

def check_top_knowledge_snippets(true_label, retrieved_snippets):
    topk = len(retrieved_snippets)
    for i, k in enumerate(retrieved_snippets):
        if check_doc(true_label, k):
            return True, (topk - i) % (topk + 1)

    return (False, 0)


def store_results(predictions, output_path):
    '''The labels for entity_id and doc_id are of type <str> in the knowledge base and of type <int> in the labels'''

    r = []
    for label in predictions:
        if label['target'] == False:
            r.append(label)
        else:
            best_ks = []
            for k in label['knowledge']:
                ks = {'domain' : k['domain']}

                ks['entity_id'] = int(entity_id) if (entity_id := k.get('entity_id')) != None and entity_id != '*' else '*'
                if (doc_id := k.get('doc_id')) != None:
                    ks['doc_id'] = int(doc_id)

                best_ks.append(ks)

            label['knowledge'] = best_ks
            r.append(label)

    with open(output_path, 'w') as f:
        json.dump(predictions, f)

    return


def score_results(dataroot, dataset, out_file, score_file):
    os.system(
        f'python scripts/scores.py --dataset "{dataset}" --dataroot "{dataroot}" --outfile "{out_file}" --scorefile "{score_file}"')
    with open(score_file, 'r') as f:
        scores = json.load(f)
        mrr = scores['selection']['mrr@5']
        r1 = scores['selection']['r@1']
        r5 = scores['selection']['r@5']

    return {'r@1':r1, 'r@5':r5, 'mrr':mrr}



# ====================================================================


def get_domain_corpus(knowledge_base: KnowledgeBase, knowledge_preprocessing):
    corpus_ids = [{'domain':domain} for domain in knowledge_base.keys()]
    corpus = [knowledge_preprocessing(ks) for ks in corpus_ids]

    return corpus, corpus_ids

def get_hierarchical_domain_corpus(knowledge_base: KnowledgeBase):
    corpus_ids = {'': [{'domain':domain} for domain in knowledge_base.keys()]}
    corpus = corpus_ids

    return corpus, corpus_ids

def get_hierarchical_entity_corpus(knowledge_base: KnowledgeBase):
    corpus = {}
    corpus_ids = {}
    for domain in knowledge_base.keys():
        corpus[domain] = []
        corpus_ids[domain] = []
        for entity_id in knowledge_base[domain]:
            name = knowledge_base[domain][entity_id]['name']
            corpus_ids[domain].append({'domain':domain, 'entity_id':entity_id, 'name':name})
            corpus[domain].append(name)

    return corpus, corpus_ids


def get_entity_corpus(knowledge_base: KnowledgeBase, knowledge_preprocessing):
    corpus_ids = []
    for domain in knowledge_base.keys():
        domain_dict = knowledge_base[domain]
        for entity_id in domain_dict:
            entity_dict = domain_dict[entity_id]
            d = {'domain': domain, 'entity_id':entity_id}
            for key, val in entity_dict.items():
                if key != 'docs':
                    d[key] = val
            corpus_ids.append(d)

    corpus = [knowledge_preprocessing(ks) for ks in corpus_ids]
    return corpus, corpus_ids


def get_hierarchical_document_corpus(knowledge_base: KnowledgeBase):
    corpus = {}
    corpus_ids = {}
    for domain in knowledge_base.keys():
        for entity_id in knowledge_base[domain]:
            docs_dict = knowledge_base[domain][entity_id]['docs']
            corpus_ids[(domain, entity_id)] = [{'domain':domain, 'entity_id':entity_id, 'doc_id':doc_id} for doc_id in docs_dict.keys()]
            corpus[(domain, entity_id)] = [' '.join(docs_dict[doc_id].values()) for doc_id in docs_dict.keys()]

    return corpus, corpus_ids


def get_document_corpus(knowledge_base, knowledge_preprocessing=None):
    if knowledge_preprocessing is None:
        corpus = [doc for doc in knowledge_base]
    else:
        corpus = [knowledge_preprocessing(doc) for doc in knowledge_base]

    corpus_ids = [knowledge_base.get_keys(i) for i in range(len(knowledge_base))]
    return corpus, corpus_ids



#========================= Print ======================

def print_example(method, query, candidates, scores, knowledge_base, label=None, fp=sys.stdout, knowledge_preprocessing=None):

    if method == 'document':
        doc_candidates = [knowledge_base.get_doc(**candidate) for candidate in candidates]
        doc_label = knowledge_base.get_doc(**label) if label is not None else None
    else:
        doc_candidates = candidates
        doc_label = label

    if knowledge_preprocessing is not None:
        doc_candidates = [knowledge_preprocessing(doc) for doc in doc_candidates]
        if doc_label is not None:
            doc_label = knowledge_preprocessing(doc_label)



    print('_'*40, 'Example', '_'*40, file=fp)
    print("Query:", file=fp)
    for i in range(0, len(query), 80):
        print('\t', query[i:i+80], file=fp)


    print(f"\n{method} predictions:", file=fp)

    for candidate, score in zip(doc_candidates, scores):
        print("(Score: {:.4f})".format(score), candidate, file=fp)

    if label is not None:
        print(f'TRUE: {doc_label}', file=fp)
    print('_' *89, file=fp)


import random
def print_title(title, sep='='):
    print('\n', sep*20, title, sep*20)

def print_triplet(anchor, pos, neg):
    print(f'{"anchor":<7}:', anchor[:100] + ('...' if len(anchor)>=100 else ''))
    print(f'{"pos":<7}:', pos[:100]+ ('...' if len(pos)>=100 else ''))
    print(f'{"neg":<7}:', neg[:100]+ ('...' if len(neg)>=100 else ''))
    print()

if __name__ == '__main__':
    k_pre = ConcatenateKnowledge()
    kb = KnowledgeBase('./DSTC9/data')

    x,y = get_document_corpus(kb, k_pre)
