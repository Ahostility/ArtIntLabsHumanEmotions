import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

class BERTClassifierModel(nn.Module):

    def __init__(self, num_classes = 2, hidden_size = 768):
        super(BERTClassifierModel, self).__init__()
        self.number_of_classes = num_classes
        self.dropout = nn.Dropout(0.01)
        self.hidden_size = hidden_size
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, inputs):
        _, embedding = self.bert(inputs[0], token_type_ids=None, attention_mask=inputs[1])
        output = self.classifier(self.dropout(embedding))
        return F.sigmoid(output)