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
import numpy as np
import pandas as pd

from typing import List, Dict

from lstm.model import LSTMModel
from lstm.pokemon_dataset import PokemonDataset as FlavorTextPokemonDataset

from gan.pokemon_generator import PokemonGenerator
from gan.pokemon_dataset import PokemonDataset as AttributePokemonDataset
from gan.generated_pokemon_indexes import GeneratedPokemonFieldIndex

from constants import (
    DATASET_PATH,
    LSTM_CHECKPOINT_PATH,
    GAN_GENERATOR_CHECKPOINT_PATH,
    FLAVOR_TEXT_LENGTH,
    NUMBER_OF_ATTRIBUTES,
    ONE_METER_IN_FEET,
    ONE_KILOGRAM_IN_POUNDS,
    OUTPUT_FILE
)


def load_lstm_model() -> LSTMModel:
    checkpoint = torch.load(LSTM_CHECKPOINT_PATH)

    model = LSTMModel(checkpoint['vocabulary_size'])
    model.load_state_dict(checkpoint['model_state_dict'])

    return model


def load_gan_generator_model() -> PokemonGenerator:
    checkpoint = torch.load(GAN_GENERATOR_CHECKPOINT_PATH)

    model = PokemonGenerator()
    model.load_state_dict(checkpoint['model_state_dict'])

    return model


def generate_flavor_text(seed: str, dataset: FlavorTextPokemonDataset, model: LSTMModel) -> str:
    model.eval()

    words = seed.split(' ')
    state_h, state_c = model.init_state(len(words))

    for i in range(0, FLAVOR_TEXT_LENGTH):
        x = torch.tensor([[dataset.word_to_index[w] for w in words[i:]]])
        y_pred, (state_h, state_c) = model(x, (state_h, state_c))

        last_word_logits = y_pred[0][-1]
        p = torch.nn.functional.softmax(last_word_logits, dim=0).detach().numpy()
        word_index = np.random.choice(len(last_word_logits), p=p)
        words.append(dataset.index_to_word[word_index])

    return ' '.join(words)


def generate_attributes(model: PokemonGenerator) -> List:
    model.eval()

    latent_space_samples = torch.randn(1, NUMBER_OF_ATTRIBUTES)
    generated_samples = model(latent_space_samples)

    return generated_samples[0]


def format_type(generated_pokemon: List) -> str:
    type_1 = int(generated_pokemon[GeneratedPokemonFieldIndex.TYPE_1])
    type_2 = int(generated_pokemon[GeneratedPokemonFieldIndex.TYPE_2])
    types = [AttributePokemonDataset.TYPE_MAP.get(type_1), AttributePokemonDataset.TYPE_MAP.get(type_2)]
    type = '/'.join(filter(lambda type: type is not None, types))

    return type


def format_abilities(generated_pokemon: List) -> str:
    ability_1 = int(generated_pokemon[GeneratedPokemonFieldIndex.ABILITY_1])
    ability_2 = int(generated_pokemon[GeneratedPokemonFieldIndex.ABILITY_2])
    abilities = [AttributePokemonDataset.ABILITY_MAP.get(ability_1), AttributePokemonDataset.ABILITY_MAP.get(ability_2)]
    ability = '/'.join(filter(lambda ability: ability is not None, abilities))

    return ability


def format_height(generated_pokemon: List) -> str:
    height = AttributePokemonDataset.HEIGHT_MAP.get((int(generated_pokemon[GeneratedPokemonFieldIndex.HEIGHT])))

    return f'{height} (m) / {height*ONE_METER_IN_FEET:.2f} (ft)'


def format_weight(generated_pokemon: List) -> str:
    weight = AttributePokemonDataset.WEIGHT_MAP.get((int(generated_pokemon[GeneratedPokemonFieldIndex.WEIGHT])))

    return f'{weight} (kg) / {weight*ONE_KILOGRAM_IN_POUNDS:.1f} (lbs)'


def format_egg_group(generated_pokemon: List) -> str:
    egg_group_1 = int(generated_pokemon[GeneratedPokemonFieldIndex.EGG_GROUP_1])
    egg_group_2 = int(generated_pokemon[GeneratedPokemonFieldIndex.EGG_GROUP_2])
    egg_groups = [AttributePokemonDataset.EGG_GROUP_MAP.get(egg_group_1),
                  AttributePokemonDataset.EGG_GROUP_MAP.get(egg_group_2)]

    egg_group = '/'.join(filter(lambda egg_group: egg_group is not None, egg_groups))

    return egg_group


def format_is_legendary(generated_pokemon: List) -> str:
    is_legendary = int(generated_pokemon[GeneratedPokemonFieldIndex.IS_LEGENDARY])

    if is_legendary == 1:
        return 'True'

    return 'False'


def format_generated_pokemon(generated_pokemon: List) -> Dict:
    result = {
        'type': format_type(generated_pokemon),
        'abilities': format_abilities(generated_pokemon),
        'hidden_ability':
            AttributePokemonDataset.ABILITY_MAP.get(int(generated_pokemon[GeneratedPokemonFieldIndex.HIDDEN_ABILITY])),
        'classification':
            AttributePokemonDataset.CLASSIFICATION_MAP.get(
                int(generated_pokemon[GeneratedPokemonFieldIndex.CLASSIFICATION])),
        'color': AttributePokemonDataset.COLOR_MAP.get(int(generated_pokemon[GeneratedPokemonFieldIndex.COLOR])),
        'shape': AttributePokemonDataset.SHAPE_MAP.get(int(generated_pokemon[GeneratedPokemonFieldIndex.SHAPE])),
        'height': format_height(generated_pokemon),
        'weight': format_weight(generated_pokemon),
        'egg_group': format_egg_group(generated_pokemon),
        'held_items': AttributePokemonDataset.HELD_ITEM_MAP.get(
            int(generated_pokemon[GeneratedPokemonFieldIndex.HELD_ITEMS])),
        'is_legendary': format_is_legendary(generated_pokemon),
        'hp': int(generated_pokemon[GeneratedPokemonFieldIndex.HP]),
        'attack': int(generated_pokemon[GeneratedPokemonFieldIndex.ATK]),
        'defense': int(generated_pokemon[GeneratedPokemonFieldIndex.DEF]),
        'special_attack': int(generated_pokemon[GeneratedPokemonFieldIndex.SATK]),
        'special_defense': int(generated_pokemon[GeneratedPokemonFieldIndex.SDEF]),
        'speed': int(generated_pokemon[GeneratedPokemonFieldIndex.SPD]),
    }

    return result


def generate_flavor_text_seed(generated_pokemon: Dict) -> str:
    color = generated_pokemon['color']
    classification = generated_pokemon['classification'][:-8]  # Remove the word "Pokémon"
    type = generated_pokemon['type'].split('/')[0]

    return f'{color} {classification} {type}'


def save_generated_pokemon_to_text(generated_pokemon: Dict) -> None:
    with open(OUTPUT_FILE, 'w') as output_file:
        # Type
        output_file.write('[Type]\n')
        output_file.write(str.title(generated_pokemon['type']))

        # Classification
        output_file.write('\n\n[Classification]\n')
        output_file.write(generated_pokemon['classification'])

        # Fakedex entry
        output_file.write('\n\n[Fakedex Entry]\n')
        import textwrap
        for line in textwrap.wrap(generated_pokemon['flavor_text']):
            output_file.write(line + '\n')

        # Is Legendary
        if generated_pokemon['is_legendary'] == 'True':
            output_file.write('\n***A legendary Fakemon!***\n')

        # Abilities
        output_file.write('\n[Abilities]\n')
        abilities = str.title(generated_pokemon['abilities'])
        hidden_ability = generated_pokemon['hidden_ability']

        if hidden_ability is not None:
            abilities += f'/{str.title(hidden_ability)} (Hidden)'

        output_file.write(abilities)

        # Base stats
        output_file.write('\n\n[Base Stats]\n')
        hp = generated_pokemon['hp']
        attack = generated_pokemon['attack']
        defense = generated_pokemon['defense']
        s_attack = generated_pokemon['special_attack']
        s_defense = generated_pokemon['special_defense']
        speed = generated_pokemon['speed']

        stats = f'{hp}/{attack}/{defense}/{s_attack}/{s_defense}/{speed}'
        output_file.write(stats)

        # Height
        output_file.write('\n\n[Height]\n')
        output_file.write(generated_pokemon['height'])

        # Weight
        output_file.write('\n\n[Weight]\n')
        output_file.write(generated_pokemon['weight'])

        # Color
        output_file.write('\n\n[Color]\n')
        output_file.write(str.title(generated_pokemon['color']))

        # Shape
        output_file.write('\n\n[Shape]\n')
        output_file.write(str.title(generated_pokemon['shape']))

        # Egg groups
        output_file.write('\n\n[Egg-Groups]\n')
        output_file.write(str.title(generated_pokemon['egg_group']))

    output_file.close()


def generate_pokemon() -> None:
    dataset = pd.read_csv(DATASET_PATH)

    # Load the attribute mappings and flavor text mappings
    lstm_dataset = FlavorTextPokemonDataset(dataset=dataset)
    AttributePokemonDataset(dataset=dataset)

    # Load the models
    lstm_model = load_lstm_model()
    gan_generator_model = load_gan_generator_model()

    # Generates the attibutes of the Pokemon
    attributes = generate_attributes(model=gan_generator_model)
    generated_pokemon = format_generated_pokemon(attributes)

    # Generate the flavor text
    generated_pokemon['flavor_text'] = generate_flavor_text(
        seed=generate_flavor_text_seed(generated_pokemon),
        dataset=lstm_dataset,
        model=lstm_model)

    import pprint
    pprint.pprint(generated_pokemon, sort_dicts=False)

    # Save the output to text file
    save_generated_pokemon_to_text(generated_pokemon)


if __name__ == '__main__':
    generate_pokemon()
