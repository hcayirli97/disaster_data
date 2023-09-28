import logging

import pandas as pd
import torch
import torch.nn as nn
from transformers import AdamW, BertModel, BertTokenizer

logging.basicConfig(level=logging.ERROR)
import argparse
import warnings

from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

train_maxlen = 140
dev_maxlen = 140
batch_size = 16
epochs = 10
bert_model = 'bert-base-uncased'
learning_rate = 1e-5

class Tokenize_dataset:
  """
  This class tokenizes the dataset using bert tokenizer
  """

  def __init__(self, text, targets, tokenizer, max_len):
    self.text = text
    self.tokenizer = tokenizer
    self.max_len = max_len
    self.targets = targets

  def __len__(self):
    return len(self.targets)

  def __getitem__(self, item):
    text = str(self.text[item])
    targets = self.targets[item]
    """
    Using encode_plus instead of encode as it helps to provide additional information that we need
    """
    inputs = self.tokenizer.encode_plus(
        str(text),
        add_special_tokens = True,
        max_length = self.max_len,
        pad_to_max_length = True
    )

    ids = inputs["input_ids"]
    mask = inputs["attention_mask"]
    token_type_ids = inputs["token_type_ids"]

    return {
        "ids": torch.tensor(ids, dtype=torch.long),
        "mask": torch.tensor(mask, dtype=torch.long),
        "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
        "targets": torch.tensor(targets, dtype=torch.long)
    }
  
def loss_function(outputs, targets):
	return nn.CrossEntropyLoss()(outputs, targets)


def train_function(data_loader, model, optimizer, device):
  """
  Function defines the training that we will happen over the entire dataset
  """
  model.train()

  running_loss = 0.0
  """
  looping over the entire training dataset
  """
  for i, data in enumerate(data_loader):
    mask = data["mask"].to(device, dtype=torch.long)
    ids = data["ids"].to(device, dtype=torch.long)
    token_type_ids = data["token_type_ids"].to(device, dtype=torch.long)
    target = data["targets"].to(device, dtype=torch.long)
    optimizer.zero_grad()

    output = model(ids=ids, mask=mask, token_type_ids=token_type_ids)

    loss = loss_function(output, target)
    loss.backward()
    optimizer.step()
    """
    calculating loss and running loss
    """
    running_loss += loss.item()

    temp = "\r{}/{} Loss: {:.4f}".format( i, data_loader.__len__(), running_loss/(batch_size*(i+1)))
    print(temp, end= "")

def eval_function(data_loader, model, device):
  """
  This function defines the loop over the dev set.
  """
  model.eval()
  correct_labels = 0
  tot = 0
  """
  no_grad as this is evaluation set and we dont want the model to update weights
  """
  with torch.no_grad():
    for i, data in enumerate(data_loader):
      mask = data["mask"].to(device, dtype=torch.long)
      ids = data["ids"].to(device, dtype=torch.long)
      token_type_ids = data["token_type_ids"].to(device, dtype=torch.long)
      targets = data["targets"].to(device, dtype=torch.long)
      outputs = model(ids=ids, mask=mask, token_type_ids=token_type_ids)
      val_loss = loss_function(outputs, targets)
      max_probs, predicted = torch.max(outputs, 1)
      tot = tot + targets.size(0)
      correct_labels = correct_labels + torch.sum(predicted==targets)
    """
    basic metrics for accuracy calculation
    """
    accuracy = correct_labels / tot * 100
  return accuracy, val_loss

class CompleteModel(nn.Module):
  """
  The model architecture is defined here which is a fully connected layer + normalization on top of a BERT model
  """

  def __init__(self, bert):
    super(CompleteModel, self).__init__()
    self.bert = BertModel.from_pretrained(bert)
    self.drop = nn.Dropout(p=0.25)
    self.out = nn.Linear(self.bert.config.hidden_size, 2) # Number of output classes = 3, positive, negative and N(none)

  def forward(self, ids, mask, token_type_ids):
    _, pooled_output = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids, return_dict=False)
    output = self.drop(pooled_output)
    return self.out(output)
  

def run(args):
  train_maxlen = args.train_maxlen
  dev_maxlen = args.dev_maxlen
  batch_size = args.batch_size
  epochs = args.epochs
  bert_model = args.bert_model
  learning_rate = args.learning_rate


  data_path = args.data_url
  df_data = pd.read_csv(data_path)
  df_train, df_valid = train_test_split(df_data, test_size= 0.25, random_state= 42)

  df_train['target'] = df_train['target']
  df_valid['target'] = df_valid['target']
  df_train = df_train.reset_index(drop=True)
  df_valid = df_valid.reset_index(drop=True)

  tokenizer = BertTokenizer.from_pretrained(bert_model)
  train_dataset = Tokenize_dataset(
        text = df_train['text'].values,
        targets = df_train['target'].values,
        tokenizer = tokenizer,
        max_len = train_maxlen
  )

  class_counts = []
  for i in range(3):
    class_counts.append(df_train[df_train['target']==i].shape[0])

  num_samples = sum(class_counts)
  labels = df_train['target'].values
  class_weights = []
  for i in range(len(class_counts)):
      if class_counts[i] != 0:
          class_weights.append(num_samples/class_counts[i])
      else:
          class_weights.append(0)
  weights = [class_weights[labels[i]] for i in range(int(num_samples))]
  sampler = torch.utils.data.sampler.WeightedRandomSampler(torch.DoubleTensor(weights), int(num_samples))
  train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = batch_size,
        shuffle = False,
        sampler = sampler
    )
  valid_dataset = Tokenize_dataset(
        text = df_valid['text'].values,
        targets = df_valid['target'].values,
      tokenizer = tokenizer,
        max_len = dev_maxlen
    )
  valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size = batch_size,
        shuffle = False
    )
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  print(f"Device: {device}")
  model = CompleteModel(bert_model).to(device)
  optimizer = AdamW(model.parameters(), lr=learning_rate)

  best_acc = 0
  for epoch in range(epochs):
    print("\nEpoch = "+ str(epoch))
    train_function(data_loader=train_data_loader, model=model, optimizer=optimizer, device=device )
    accuracy, val_loss = eval_function(data_loader=valid_data_loader, model=model, device=device)
    print(" Val Loss: {:.2f} Val Accuracy: {:.2f}".format(val_loss, accuracy/100))
    if float(accuracy) > best_acc:     
      torch.save(model, "Model_" + str(epoch) + '.bin')
      best_acc = float(accuracy)

if __name__ == "__main__":

  parser = argparse.ArgumentParser(description='Run')
  parser.add_argument('--model', default='bert-base-uncased', help='model type')
  parser.add_argument('--train_maxlen',  default=140, type=int)
  parser.add_argument('--dev_maxlen',  default=140, type=int)
  parser.add_argument('--data_url', default="./data/train.csv", type=str, help='the training and validation data path')
  parser.add_argument('--train_url', default="./output/", type=str, help='the path to save training outputs')

  parser.add_argument('--epochs', default=10, type=int)
  parser.add_argument('--batch_size', default=16, type=int)
  parser.add_argument('--learning_rate', default=1e-5, type=float)

  args, unknown = parser.parse_known_args()
  run(args)