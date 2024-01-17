from torch.utils.data import Dataset
from PIL import Image
import torch
import json
import h5py
import bisect
import pandas as pd
import os 
CAPTION_LENGTH = 35
SIMPLE_PREFIX = "This image shows "

def prep_strings(text, tokenizer, template=None, retrieved_caps=None, k=None, is_test=False, max_length=None):

    if is_test:
        padding = False
        truncation = False
    else:
        padding = True 
        truncation = True

    if retrieved_caps is not None:
        infix = '\n\n'.join(retrieved_caps[:k]) + '.'
        prefix = template.replace('||', infix)
    else:
        prefix = SIMPLE_PREFIX

    prefix_ids = tokenizer.encode(prefix)
    len_prefix = len(prefix_ids)

    text_ids = tokenizer.encode(text)
    if truncation:
        text_ids = text_ids[:CAPTION_LENGTH]
    input_ids = prefix_ids + text_ids if not is_test else prefix_ids
    
    # we ignore the prefix (minus one as the first subtoken in the prefix is not predicted)
    label_ids = [-100] * (len_prefix - 1) + text_ids + [tokenizer.eos_token_id] 
    if padding:
        input_ids += [tokenizer.pad_token_id] * (max_length - len(input_ids))
        label_ids += [-100] * (max_length - len(label_ids))
    
    if is_test:
        return input_ids
    else:  
        return input_ids, label_ids

def postprocess_preds(pred, tokenizer):
    # print("postprocess_preds")
    # print(pred)
    # print("hello")
    pred = pred.split(SIMPLE_PREFIX)[-1]
    
    
    pred = pred.replace(tokenizer.pad_token, '')
    if pred.endswith(tokenizer.eos_token):
        pred = pred[:-1]
    return pred

class TrainDataset(Dataset):
    def __init__(self, df, features_path, tokenizer, rag=False, template_path=None, k=None, max_target_length=150):
        self.df = df
        self.tokenizer = tokenizer
        self.max_target_length = max_target_length 
        self.features = h5py.File(features_path, 'r')

        if rag:
            self.template = open(template_path).read().strip() + ' '
            assert k is not None 
            self.k = k
        self.rag = rag

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        #print(self.df['cocoid'])
        text = self.df['text'][idx]
        #print(text)
        if self.rag: 
            caps = self.df['caps'][idx]
            decoder_input_ids, labels = prep_strings(text, self.tokenizer, template=self.template,
                                                     retrieved_caps=caps, k=self.k, max_length=self.max_target_length)
        else:
            decoder_input_ids, labels = prep_strings(text, self.tokenizer, max_length=self.max_target_length)
        # load precomputed features
        #print(self.df['cocoid'][idx])
        image_name=self.df['cocoid'][idx]+'.jpg'
        if os.path.isfile('vietcap/images/'+image_name):
            index=self.df['cocoid'][idx]+".jpg"
        else:
            index=self.df['cocoid'][idx]+".png"
        encoder_outputs = self.features[index][()]
        if len(decoder_input_ids)>143:
            #print(text)
            #print(labels)
        
            decoder_input_ids=decoder_input_ids[:143]
            labels=labels[:143]
            #print(len(decoder_input_ids))
            
        #print(encoder_outputs.shape)
        encoding = {"encoder_outputs": torch.tensor(encoder_outputs), 
                    "decoder_input_ids": torch.tensor(decoder_input_ids),
                    "labels": torch.tensor(labels)}
        
            
        return encoding


def load_data_for_training(annot_path, caps_path=None):
    labels=json.load(open("vietcap/captions_smallcap.json","r",encoding="utf-8"))
    labels={i["id"]:i["captions"] for i in labels}
    annotations = pd.read_csv(annot_path)
    if caps_path is not None:
        retrieved_caps = json.load(open(caps_path))
    #print(annotations)
    #print(retrieved_caps)
    #exit()
    data = {'train': [], 'val': []}
    for item in annotations.values:
        file_name=item[0][:-4]
        #print(file_name)
        #exit()
        if caps_path is not None:
            caps = retrieved_caps[file_name]
            #print(caps)
        else:
            caps = None
        samples=[]
        #print(item[0])
        #print()
        #print(labels)
        setences=labels[item[0]].split("\n")
        value=min(len(setences),3)
        for sentence in setences:
            samples.append({'file_name': file_name, 'cocoid':file_name, 'text': sentence, 'caps': caps})
        if item[2] == 'train':
            data['train']+=samples
        elif item[2] == 'val':
            data['val']+=samples
    return data
def load_data_for_inference(annot_path, caps_path=None):
    annotations =pd.read_csv(annot_path)
    if caps_path is not None:
        retrieved_caps = json.load(open(caps_path))
    data = {'test': [], 'val': []}

    for item in annotations.values:
        #file_name = item['filename'].split('_')[-1]
        file_name=item[0]
        if caps_path is not None:
            caps = retrieved_caps[file_name[:-4]]
            #print(caps)
        else:
            caps = None
        image = {'file_name': file_name, 'caps': caps, 'image_id':file_name[:-4]}
        if item[2] == 'test':
            data['test'].append(image)
        elif item[2] == 'val':
            data['val'].append(image)

    return data      

