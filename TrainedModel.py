from Model import BertClassifier as BC
import typing
from transformers import BertModel, BertTokenizer
from torch import nn
import re
import emoji
import torch
import pickle


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

class BertClassifier():
    def __init__ (self):
        self.model = BC().to(device)
        self.model.load_state_dict(torch.load("airline_sentiment.pth"))
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    

    def preprocess(self,sentence):
        pattern=r"@\s[a-zA-Z]*" # this is to cover for places where there is a space next to @ example '@ '
        sentence = re.sub(pattern, "",emoji.demojize(sentence, delimiters=("", "")))
        return sentence
        

    def predict(self,sentence):
        test_input = self.tokenizer(self.preprocess(sentence),padding='max_length', max_length = 512, truncation=True,return_tensors="pt").to(device)
        mask = test_input['attention_mask'].to(device)
        input_id = test_input['input_ids'].squeeze(1).to(device)
        output = self.model(input_id, mask)
        value = output.argmax(dim=1).item()
        if(value == 1):
            value = "Positive"
        else:
            value = "Negative"
        del test_input
        del mask
        del output
        return value
    def save(self):
        self.model.save_pretrained('pytorch1_model.bin')
    def generate(self):
        self.model.config.to_json_file("config.json")
# BertClassifier().save()