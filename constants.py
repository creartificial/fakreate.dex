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


# Basic constants used throughout the code
NUMBER_OF_POKEMON = 898
DATASET_PATH = 'dataset/pokemon.csv'
FIELD_NAMES = [
    'id',
    'name',
    'type_1',
    'type_2',
    'ability_1',
    'ability_2',
    'hidden_ability',
    'classification',
    'hp',
    'attack',
    'defense',
    'special_attack',
    'special_defense',
    'speed',
    'color',
    'height(m)',
    'weight(kg)',
    'shape',
    'habitat',
    'egg_group_1',
    'egg_group_2',
    'held_items',
    'is_legendary',
    'flavor_text',
]

# PokeAPI related contants
BASE_API_ROUTE = 'https://pokeapi.co/api/v2'
POKEMON_BASE_ROUTE = 'pokemon'
SPECIES_INFO_ROUTE = 'pokemon-species'

# GAN Training parameters
GAN_BATCH_SIZE = 32
GAN_NUMBER_OF_EPOCHS = 900

# LSTM Training parameters
LSTM_BATCH_SIZE = 256
LSTM_NUMBER_OF_EPOCHS = 80
LSTM_SEQUENCE_LENGTH = 7

# Labels
GAN_TRUE_DATA_LABEL = 1
GAN_GENERATED_DATA_LABEL = 0

# Attributes size
NUMBER_OF_ATTRIBUTES = 21

# Text size
FLAVOR_TEXT_LENGTH = 75

# Checkpoints paths
LSTM_CHECKPOINT_PATH = 'checkpoints/lstm.pt'
GAN_GENERATOR_CHECKPOINT_PATH = 'checkpoints/gan_generator.pt'
GAN_DISCRIMINATOR_CHECKPOINT_PATH = 'checkpoints/gan_discriminator.pt'

# Conversion Values
ONE_METER_IN_FEET = 3.280839895
ONE_KILOGRAM_IN_POUNDS = 2.20462262185

# Final output file
OUTPUT_FILE = 'out/generated_pokemon.txt'
