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


from typing import List, Dict
from enum import IntEnum

escapes = ''.join([chr(char) for char in range(1, 32)])
translator = str.maketrans(' ', ' ', escapes)


class Stats(IntEnum):
    HP = 0
    ATK = 1
    DEF = 2
    SATK = 3
    SDEF = 4
    SPD = 5


def extract_flavor_text(flavor_text_entries: List[Dict]) -> str:
    english_entries = [entry['flavor_text'] for entry in flavor_text_entries if entry['language']['name'] == 'en']
    parsed_entries = set()

    for entry in english_entries:
        entry = entry.replace('\n', ' ').replace('\xad', ' ').translate(translator)
        parsed_entries.add(entry)

    return ' '.join(parsed_entries)


def extract_genera(genus_entries: List[Dict]) -> str:
    return ''.join([genus['genus'] for genus in genus_entries if genus['language']['name'] == 'en'])


def extract_name(name_entries: List[Dict]) -> str:
    return ''.join([name['name'] for name in name_entries if name['language']['name'] == 'en'])


def extract_egg_groups(egg_groups: List[Dict]) -> List[str]:
    return [egg_group['name'] for egg_group in egg_groups]


def extract_types(types: List[Dict]) -> List[str]:
    return [type['type']['name'] for type in types]


def extract_abilities(abilities: List[Dict], is_hidden: bool = False) -> List[str]:
    return [ability['ability']['name'] for ability in abilities if ability['is_hidden'] == is_hidden]


def extract_held_items(held_items: List[Dict]) -> str:
    if len(held_items) == 0:
        return ''

    return ','.join(item['item']['name'] for item in held_items)


def extract_pokemon_data(id: int, base_data: Dict, species_info: Dict) -> Dict:
    egg_groups = extract_egg_groups(species_info['egg_groups'])
    types = extract_types(base_data['types'])
    abilities = extract_abilities(base_data['abilities'])
    hidden_ability = extract_abilities(base_data['abilities'], is_hidden=True)

    return {
        'id': id,
        'name': extract_name(species_info['names']),
        'type_1': types[0],
        'type_2': types[1] if len(types) > 1 else None,
        'ability_1': abilities[0],
        'ability_2': abilities[1] if len(abilities) > 1 else None,
        'hidden_ability': hidden_ability[0] if len(hidden_ability) > 0 else None,
        'classification': extract_genera(species_info['genera']),
        'hp': base_data['stats'][Stats.HP]['base_stat'],
        'attack': base_data['stats'][Stats.ATK]['base_stat'],
        'defense': base_data['stats'][Stats.DEF]['base_stat'],
        'special_attack': base_data['stats'][Stats.SATK]['base_stat'],
        'special_defense': base_data['stats'][Stats.SDEF]['base_stat'],
        'speed': base_data['stats'][Stats.SPD]['base_stat'],
        'color': species_info['color']['name'],
        'height(m)': float(base_data['height']) / 10.0,
        'weight(kg)': float(base_data['weight']) / 10.0,
        'shape': species_info['shape']['name'],
        'habitat': species_info['habitat']['name'] if species_info['habitat'] is not None else None,
        'egg_group_1': egg_groups[0] if len(egg_groups) > 0 else None,
        'egg_group_2': egg_groups[1] if len(egg_groups) > 1 else None,
        'held_items': extract_held_items(base_data['held_items']),
        'is_legendary': species_info['is_legendary'],
        'flavor_text': extract_flavor_text(species_info['flavor_text_entries'])
    }
