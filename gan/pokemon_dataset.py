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
from torch.utils.data import Dataset
from pandas import DataFrame, isna
from bidict import bidict

from typing import List, Tuple

from constants import (
    NUMBER_OF_POKEMON,
    NUMBER_OF_ATTRIBUTES
)


class PokemonDataset(Dataset):
    # Creates the string to int mappings
    TYPE_SET = set()
    TYPE_MAP = bidict()

    ABILITY_SET = set()
    ABILITY_MAP = bidict()

    COLOR_SET = set()
    COLOR_MAP = bidict()

    WORD_SET = set()
    WORD_MAP = bidict()

    CLASSIFICATION_SET = set()
    CLASSIFICATION_MAP = bidict()

    HEIGHT_SET = set()
    HEIGHT_MAP = bidict()

    WEIGHT_SET = set()
    WEIGHT_MAP = bidict()

    SHAPE_SET = set()
    SHAPE_MAP = bidict()

    EGG_GROUP_SET = set()
    EGG_GROUP_MAP = bidict()

    HELD_ITEM_SET = set()
    HELD_ITEM_MAP = bidict()

    def __init__(self, dataset: DataFrame):
        dataset = self.prepare_dataset(dataset)

        self.x_train = torch.tensor(dataset.iloc[0:NUMBER_OF_POKEMON, 0:NUMBER_OF_ATTRIBUTES].values)

    def __len__(self) -> int:
        return NUMBER_OF_POKEMON

    def __getitem__(self, index) -> Tuple:
        # We don't have labels for this case, so we simply return the label for the real data
        return self.x_train[index]

    @classmethod
    def convert_string_to_int(cls, string: str, string_set: set, string_to_int_map: bidict) -> int:
        if isna(string):
            return -1

        if string not in string_set:
            string_set.add(string)
            string_to_int_map[len(string_to_int_map)] = string

        return string_to_int_map.inverse[string]

    @classmethod
    def convert_boolean_to_int(cls, boolean: bool) -> int:
        return 1 if boolean else 0

    @classmethod
    def convert_words_to_int(cls, words: str) -> List:
        if isna(words):
            return [-1]

        words = words.split()
        word_array = []

        for word in words:
            word_array.append(cls.convert_string_to_int(
                string=word,
                string_set=cls.WORD_SET,
                string_to_int_map=cls.WORD_MAP,
            ))

        return word_array

    def convert_dataset_string_field_to_int(self,
                                            dataset: DataFrame,
                                            field: str,
                                            string_set: set,
                                            string_to_int_map: bidict
                                            ):
        dataset[field] = dataset[field].map(lambda string: self.convert_string_to_int(
            string=string,
            string_set=string_set,
            string_to_int_map=string_to_int_map
        ))

    def prepare_dataset(self, dataset: DataFrame) -> DataFrame:
        # Convert the strings to integers

        # Type
        self.convert_dataset_string_field_to_int(
            dataset=dataset, field='type_1', string_set=self.TYPE_SET, string_to_int_map=self.TYPE_MAP)
        self.convert_dataset_string_field_to_int(
            dataset=dataset, field='type_2', string_set=self.TYPE_SET, string_to_int_map=self.TYPE_MAP)

        # Ability
        self.convert_dataset_string_field_to_int(
            dataset=dataset, field='ability_1', string_set=self.ABILITY_SET, string_to_int_map=self.ABILITY_MAP)
        self.convert_dataset_string_field_to_int(
            dataset=dataset, field='ability_2', string_set=self.ABILITY_SET, string_to_int_map=self.ABILITY_MAP)
        self.convert_dataset_string_field_to_int(
            dataset=dataset, field='hidden_ability', string_set=self.ABILITY_SET, string_to_int_map=self.ABILITY_MAP)

        # Classification
        self.convert_dataset_string_field_to_int(
            dataset=dataset,
            field='classification',
            string_set=self.CLASSIFICATION_SET,
            string_to_int_map=self.CLASSIFICATION_MAP
        )

        # Color
        self.convert_dataset_string_field_to_int(
            dataset=dataset, field='color', string_set=self.COLOR_SET, string_to_int_map=self.COLOR_MAP)

        # Height and Weight
        self.convert_dataset_string_field_to_int(
            dataset=dataset, field='height(m)', string_set=self.HEIGHT_SET, string_to_int_map=self.HEIGHT_MAP)
        self.convert_dataset_string_field_to_int(
            dataset=dataset, field='weight(kg)', string_set=self.WEIGHT_SET, string_to_int_map=self.WEIGHT_MAP)

        # Shape
        self.convert_dataset_string_field_to_int(
            dataset=dataset, field='shape', string_set=self.SHAPE_SET, string_to_int_map=self.SHAPE_MAP)

        # Egg Groups
        self.convert_dataset_string_field_to_int(
            dataset=dataset, field='egg_group_1', string_set=self.EGG_GROUP_SET, string_to_int_map=self.EGG_GROUP_MAP)
        self.convert_dataset_string_field_to_int(
            dataset=dataset, field='egg_group_2', string_set=self.EGG_GROUP_SET, string_to_int_map=self.EGG_GROUP_MAP)

        # Held Item
        self.convert_dataset_string_field_to_int(
            dataset=dataset, field='held_items', string_set=self.HELD_ITEM_SET, string_to_int_map=self.HELD_ITEM_MAP)

        # Is Legendary
        dataset['is_legendary'] = dataset['is_legendary'].map(lambda is_legendary: self.convert_boolean_to_int(
            boolean=is_legendary
        ))

        # Remove unused fields
        dataset.drop('name', inplace=True, axis=1)
        dataset.drop('flavor_text', inplace=True, axis=1)
        dataset.drop('habitat', inplace=True, axis=1)

        return dataset
