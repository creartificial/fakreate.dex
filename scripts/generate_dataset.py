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


import requests
import csv

from scripts.data_extractor import extract_pokemon_data
from constants import (
    NUMBER_OF_POKEMON,
    BASE_API_ROUTE,
    POKEMON_BASE_ROUTE,
    SPECIES_INFO_ROUTE,
    FIELD_NAMES,
    DATASET_PATH
)


def generate_dataset():
    pokemon_info = []

    # Iterate through all the Pokémon extracting their data
    for i in range(1, NUMBER_OF_POKEMON + 1):
        print(f'Processing Pokémon {i} Data')

        pokemon_base_data = requests.get(f'{BASE_API_ROUTE}/{POKEMON_BASE_ROUTE}/{i}').json()
        species_info_data = requests.get(f'{BASE_API_ROUTE}/{SPECIES_INFO_ROUTE}/{i}').json()
        pokemon_info.append(extract_pokemon_data(
            id=i,
            base_data=pokemon_base_data,
            species_info=species_info_data
        ))

    print('Generating CSV')

    # Write the pokemon data to csv
    with open(DATASET_PATH, 'w') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=FIELD_NAMES)

        writer.writeheader()

        for pokemon in pokemon_info:
            writer.writerow(pokemon)


if __name__ == '__main__':
    generate_dataset()
