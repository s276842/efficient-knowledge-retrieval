import json
import time
from tqdm import tqdm
from models import BaseTransformerEncoder, DenseRetriever, Cosine, DotProd, Euclidean
from dataset import DialogContext
import torch
import os
import models
import transforms

def main(dataroot, dataset, device, output_path, config):

    # data
    log_file = config['DATA']['log_file']
    label_file = config['DATA']['label_file']
    log_file_path = os.path.join(dataroot, dataset, log_file)
    label_file_path = os.path.join(dataroot, dataset, label_file)
    v_knowledge_file = os.path.join(dataroot, config['DATA']['vectorized_knowledge_file'])

    # dialog
    data_transform_cls = getattr(transforms, config['DIALOG']['data_transform'])
    reverse = config.getboolean('DIALOG', 'reverse')
    limit = config.getint('DIALOG', 'limit')
    d_transform = data_transform_cls(reverse = reverse, limit=limit)
    d_data = DialogContext(log_file_path, label_file_path, data_transform=d_transform)

    # model
    encoder_cls = getattr(models, config['MODEL']['encoder_model'])
    model_name = config['MODEL']['model_name_or_path']
    outputlayer = config['MODEL']['outputlayer']
    d_encoder = encoder_cls(model_name, outputlayer, device)

    score_function = config['MODEL']['score_function']
    if score_function == 'cos':
        score_function = Cosine()
    elif score_function == 'dot':
        score_function = DotProd()
    elif score_function == 'euclidean':
        score_function = Euclidean()

    projections = config['MODEL']['matrices'].split('-')
    s_model = DenseRetriever(v_knowledge_file, score_function, projections=projections, topk=5)

    res = []
    sel_times = []

    with torch.no_grad():
        for data, label in tqdm(d_data):

            # non-knowledge-seeking turns
            if label['target'] == False:
                res.append(label)
            # knowledge-seeking turns
            else:
                start_selection_time = time.time()
                e_data = d_encoder(data)
                k_index = s_model(e_data)
                label['knowledge'] = [{'domain':k[0], 'entity_id':int(k[1]) if k[1] != '*' else k[1], 'doc_id':int(k[2])} for k in k_index]
                res.append(label)
                sel_times.append(time.time() - start_selection_time)
                break

    res = {'pred':res, 'sel_times':sel_times}
    with open(output_path, 'w') as f: json.dump(res, f)
    return




if __name__ == '__main__':
    import configparser
    config = configparser.ConfigParser()
    config.read('./conf.ini')
    main('./DSTC9/data_eval/', 'test', 'cpu', './configurations/zeroshot/prova0.json', config)

