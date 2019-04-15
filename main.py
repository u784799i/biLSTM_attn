import torch

lr = 0.001
epochs = 1000
batch_size = 16
seed = 1111
cuda_able = True
save = './bilstm_attn_model'
data = './data/corpus.pt'
dropout = 0.5
embed_dim = 64
hidden_size = 32
bidirectional = True
weight_decay = 0.001
attention_size = 16
sequence_length = 16

torch.manual_seed(seed)

use_cuda = torch.cuda.is_available() and cuda_able


###################################################
#load data

from data_loder import DataLoader

data = torch.load(data)
max_len = data["max_len"]
vocab_size = data['dict']['vocab_size']
output_size = data['dict']['label_size']


training_data = DataLoader(data['train']['src'],
                           data['train']['label'],
                           max_len,
                           batch_size=batch_size,
                           cuda=use_cuda)
validation_data = DataLoader(data['valid']['src'],
                             data['valid']['label'],
                             max_len,
                             batch_size=batch_size,
                             shuffle=False,
                             cuda=use_cuda)

###############################################
#build model

import model
lstm_attn = model.bilstm_attn(batch_size=batch_size,
                                  output_size=output_size,
                                  hidden_size=hidden_size,
                                  vocab_size=vocab_size,
                                  embed_dim=embed_dim,
                                  bidirectional=bidirectional,
                                  dropout=dropout,
                                  use_cuda=use_cuda,
                                  attention_size=attention_size,
                                  sequence_length=sequence_length)
if use_cuda:
    lstm_attn = lstm_attn.cuda()

optimizer = torch.optim.Adam(lstm_attn.parameters(), lr=lr, weight_decay=weight_decay)
criterion = torch.nn.CrossEntropyLoss()

###################################################
#training
import time
from tqdm import tqdm

train_loss = []
valid_loss = []
accuracy = []




def evaluate():
    lstm_attn.eval()
    corrects = eval_loss = 0
    _size = validation_data.sents_size

    for data, label in tqdm(validation_data, mininterval=0.2,
                desc='Evaluate Processing', leave=False):

        pred = lstm_attn(data)
        loss = criterion(pred, label)

        eval_loss += loss.data
        corrects += (torch.max(pred, 1)[1].view(label.size()).data == label.data).sum()
    return eval_loss[0]/_size, corrects, corrects*100.0/_size, _size


def train():
    lstm_attn.train()
    total_loss = 0
    for data, label in tqdm(training_data, mininterval=1,
                desc='Train Processing', leave=False):
        optimizer.zero_grad()

        target = lstm_attn(data)
        loss = criterion(target, label)

        loss.backward()
        optimizer.step()

        total_loss += loss.data
    return total_loss[0]/training_data.sents_size

#################################################
#saving
best_acc = None
total_start_time = time.time()

try:
    print('-' * 90)
    for epoch in range(1, epochs+1):
        epoch_start_time = time.time()
        loss = train()
        train_loss.append(loss*1000.)

        print('| start of epoch {:3d} | time: {:2.2f}s | loss {:5.6f}'.format(epoch,
                                                                              time.time() - epoch_start_time,
                                                                              loss))

        loss, corrects, acc, size = evaluate()
        valid_loss.append(loss*1000.)
        accuracy.append(acc)

        print('-' * 10)
        print('| end of epoch {:3d} | time: {:2.2f}s | loss {:.4f} | accuracy {}%({}/{})'.format(epoch,
                                                                                                 time.time() - epoch_start_time,
                                                                                                 loss,
                                                                                                 acc,
                                                                                                 corrects,
                                                                                                 size))
        print('-' * 10)
        if not best_acc or best_acc < corrects:
            best_acc = corrects
            model_state_dict = lstm_attn.state_dict()
            model_source = {
                "model": model_state_dict,
                "src_dict": data['dict']['train']
            }
            torch.save(model_source, save)
except KeyboardInterrupt:
    print("-"*90)
    print("Exiting from training early | cost time: {:5.2f}min".format((time.time() - total_start_time)/60.0))