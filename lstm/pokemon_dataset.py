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


import torch

from typing import List
from pandas import DataFrame
from collections import Counter

from constants import (
    LSTM_SEQUENCE_LENGTH,
    NUMBER_OF_POKEMON,
)


class PokemonDataset(torch.utils.data.Dataset):
    def __init__(self, dataset: DataFrame):
        self.words = self.load_words(dataset)
        self.vocabulary = self.get_vocabulary()
        self.vocabulary_size = len(self.vocabulary)

        self.index_to_word = {index: word for index, word in enumerate(self.vocabulary)}
        self.word_to_index = {word: index for index, word in enumerate(self.vocabulary)}

        self.words_indexes = [self.word_to_index[w] for w in self.words]

    def load_words(self, dataset: DataFrame) -> List[str]:
        text = ''

        for i in range(NUMBER_OF_POKEMON):
            # Add the color
            text = text + ' ' + dataset.iloc[[i]]['color'].str.cat(sep=' ')

            # Add the classification (Removing the String "Pokémon")
            text = text + ' ' + dataset.iloc[[i]]['classification'].str.cat(sep=' ')[:-8]

            # Add the type
            text = text + ' ' + dataset.iloc[[i]]['type_1'].str.cat(sep=' ')

            # Add the flavor text
            text = text + ' ' + dataset.iloc[[i]]['flavor_text'].str.cat(sep=' ')

        return text.split(' ')

    def get_vocabulary(self) -> List[str]:
        word_counts = Counter(self.words)
        return sorted(word_counts, key=word_counts.get, reverse=True)

    def __len__(self):
        return len(self.words_indexes) - LSTM_SEQUENCE_LENGTH

    def __getitem__(self, index):
        return (
            torch.tensor(self.words_indexes[index:index+LSTM_SEQUENCE_LENGTH]),
            torch.tensor(self.words_indexes[index+1:index+LSTM_SEQUENCE_LENGTH+1]),
        )
