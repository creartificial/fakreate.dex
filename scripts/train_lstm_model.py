"""
BSD 4-Clause License

Copyright (c) 2021, creartificial
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.
3. All advertising materials mentioning features or use of this software
   must display the following acknowledgement:
   This product includes software developed by creartificial.
4. Neither the name of creartificial nor the
   names of its contributors may be used to endorse or promote products
   derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER ''AS IS'' AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""


import os
import time

import torch
import pandas as pd
from torch import nn, optim
from torch.utils.data import DataLoader
from lstm.model import LSTMModel
from lstm.pokemon_dataset import PokemonDataset

from constants import (
    DATASET_PATH,
    LSTM_NUMBER_OF_EPOCHS,
    LSTM_BATCH_SIZE,
    LSTM_SEQUENCE_LENGTH,
    LSTM_CHECKPOINT_PATH,
)


def train_lstm_model():
    # Load the dataset
    dataset = PokemonDataset(dataset=pd.read_csv(DATASET_PATH))

    # Create the dataloader
    dataloader = DataLoader(dataset, batch_size=LSTM_BATCH_SIZE)

    # Create the model
    model = LSTMModel(vocabulary_size=dataset.vocabulary_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Load the checkpoints if they exist
    if os.path.exists(LSTM_CHECKPOINT_PATH):
        print("Loading existing checkpoint")

        # Generator
        checkpoint = torch.load(LSTM_CHECKPOINT_PATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        last_loss = checkpoint['loss']
        last_epoch = checkpoint['epoch']

        print(f'Loaded Epoch {last_epoch}, Loss: {last_loss}')
    else:
        # Initialize the variables
        last_epoch = 0
        last_loss = None

    model.train()

    start_time = time.time()
    for epoch in range(last_epoch, LSTM_NUMBER_OF_EPOCHS):
        state_h, state_c = model.init_state(LSTM_SEQUENCE_LENGTH)

        for batch, (x, y) in enumerate(dataloader):
            optimizer.zero_grad()

            y_pred, (state_h, state_c) = model(x, (state_h, state_c))
            loss = criterion(y_pred.transpose(1, 2), y)

            state_h = state_h.detach()
            state_c = state_c.detach()

            loss.backward()
            optimizer.step()

            # Show loss
            if batch % 50 == 0:
                elapsed_time = time.time() - start_time
                elapsed_time_formatted = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
                print(f'Epoch {epoch}, Batch: {batch}, Loss: {loss.item()}, Elapsed time: {elapsed_time_formatted}')

            last_epoch = epoch
            last_loss = loss.item()

    print(f'Saving LSTM model checkpoint to {LSTM_CHECKPOINT_PATH}. Epoch: {last_epoch}. Loss: {last_loss}')
    torch.save({
        'epoch': last_epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': last_loss,
        'vocabulary_size': dataset.vocabulary_size,
    }, LSTM_CHECKPOINT_PATH)


if __name__ == '__main__':
    train_lstm_model()
