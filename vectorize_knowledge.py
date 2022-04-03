import configparser
import os

from dataset import KnowledgeBase
from models import BaseTransformerEncoder
import models
import torch


def main(dataroot, device, output_path, config):

    # data
    knowledge_file = os.path.join(dataroot, config['DATA']['knowledge_file'])
    k_base = KnowledgeBase(knowledge_file)

    # model
    encoder_cls = getattr(models, config['MODEL']['encoder_model'])
    model_name = config['MODEL']['model_name_or_path']
    outputlayer = config['MODEL']['outputlayer']
    k_encoder = encoder_cls(model_name, outputlayer, device)
    sep_token = k_encoder.tokenizer.sep_token

    # domains = list(k_base.knowledge.keys())
    # entities = [(d, e) for d in domains for e in k_base.knowledge[d].keys()]
    # docs = [(d, e, k) for d in domains for e in k_base.knowledge[d].keys() for k in k_base.knowledge[d][e]['docs'].keys()]

    domains = []
    entities = []
    docs = []

    projections = ['D', 'E', 'DE', 'DEK', 'DK', 'EK', 'K']
    res = {k: {'matrix': [], 'keys': []} for k in projections}

    for domain in k_base.knowledge:
        domain_dict = k_base.knowledge[domain]
        domains.append({domain})
        res['D']['matrix'].append(k_encoder(domain))

        for entity in domain_dict:
            entity_dict = domain_dict[entity]

            entities.append((domain, entity))
            entity_name = name if (name := entity_dict['name']) is not None else domain
            res['E']['matrix'].append(k_encoder(entity_name))
            domain_entity_name = f'{domain} {sep_token} {entity_name}'
            res['DE']['matrix'].append(k_encoder(domain_entity_name))

            for doc in entity_dict['docs']:
                docs.append((domain, entity, doc))
                question, answer = entity_dict['docs'][doc].values()
                knowledge = question + ' ' + answer
                res['K']['matrix'].append(k_encoder(knowledge))
                entity_knowledge = f'{entity_name} {sep_token} {knowledge}'
                res['EK']['matrix'].append(k_encoder(entity_knowledge))
                domain_knowledge = f'{domain} {sep_token} {knowledge}'
                res['DK']['matrix'].append(k_encoder(domain_knowledge))
                domain_entity_knowledge = f'{domain} {sep_token} {entity_knowledge}'
                res['DEK']['matrix'].append(k_encoder(domain_entity_knowledge))


    for p in projections:
        res[p]['matrix'] = torch.cat(res[p]['matrix'])

        if 'K' in p:
            res[p]['keys'] = docs
        elif 'E' in p:
            res[p]['keys'] = entities
        else:
            res[p]['keys'] = domains

    name = os.path.basename(model_name).split('.')[0] + '_' + outputlayer + '.pt'
    torch.save(res, os.path.join(output_path, name))


if __name__ == '__main__':
    config =configparser.ConfigParser()
    config.read('./vectorized_knowledge/conf.ini')
    main('./DSTC9/data_eval/', 'cpu', './DSTC9/data_eval/vectorized_knowledge/', config)