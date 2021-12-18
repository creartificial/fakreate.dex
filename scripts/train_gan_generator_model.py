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
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from gan.pokemon_dataset import PokemonDataset
from gan.pokemon_discriminator import PokemonDiscriminator
from gan.pokemon_generator import PokemonGenerator

from constants import (
    DATASET_PATH,
    GAN_BATCH_SIZE,
    GAN_NUMBER_OF_EPOCHS,
    NUMBER_OF_ATTRIBUTES,
    GAN_GENERATOR_CHECKPOINT_PATH,
    GAN_DISCRIMINATOR_CHECKPOINT_PATH
)


def create_data_loader() -> DataLoader:
    dataset = pd.read_csv(DATASET_PATH)

    train_loader = DataLoader(
        PokemonDataset(dataset),
        batch_size=GAN_BATCH_SIZE,
        shuffle=True,
    )

    return train_loader


def train_gan_models():
    print('Loading data')
    data_loader = create_data_loader()

    print('Creating GAN models and optimizers')
    pokemon_discriminator = PokemonDiscriminator()
    pokemon_generator = PokemonGenerator()
    discriminator_optimizer = torch.optim.Adam(pokemon_discriminator.parameters())
    generator_optimizer = torch.optim.Adam(pokemon_generator.parameters())
    loss_function = nn.BCELoss()

    # Load the checkpoints if they exist
    if os.path.exists(GAN_GENERATOR_CHECKPOINT_PATH) and os.path.exists(GAN_DISCRIMINATOR_CHECKPOINT_PATH):
        print("Loading existing checkpoints")

        # Generator
        generator_checkpoint = torch.load(GAN_GENERATOR_CHECKPOINT_PATH)
        pokemon_generator.load_state_dict(generator_checkpoint['model_state_dict'])
        generator_optimizer.load_state_dict(generator_checkpoint['optimizer_state_dict'])
        generator_loss = generator_checkpoint['loss']

        # Discriminator
        discriminator_checkpoint = torch.load(GAN_DISCRIMINATOR_CHECKPOINT_PATH)
        pokemon_discriminator.load_state_dict(discriminator_checkpoint['model_state_dict'])
        discriminator_optimizer.load_state_dict(discriminator_checkpoint['optimizer_state_dict'])
        discriminator_loss = discriminator_checkpoint['loss']

        last_epoch = generator_checkpoint['epoch']
        pokemon_generator.train()
        pokemon_discriminator.train()
    else:
        # Initialize the variables
        last_epoch = 0
        generator_loss = None
        discriminator_loss = None

    for epoch in range(last_epoch, GAN_NUMBER_OF_EPOCHS):
        for n, real_samples in enumerate(data_loader):
            # We need to break if the sample size is smaller than the batch size
            if len(real_samples) < GAN_BATCH_SIZE:
                continue

            # Data for training the discriminator
            real_samples_labels = torch.ones((GAN_BATCH_SIZE, 1))
            latent_space_samples = torch.randn((GAN_BATCH_SIZE, NUMBER_OF_ATTRIBUTES))
            generated_samples = pokemon_generator(latent_space_samples)
            generated_samples_labels = torch.zeros((GAN_BATCH_SIZE, 1))
            all_samples = torch.cat((real_samples, generated_samples))
            all_samples_labels = torch.cat((real_samples_labels, generated_samples_labels))

            # Training the discriminator
            pokemon_discriminator.zero_grad()
            discriminator_output = pokemon_discriminator(all_samples)
            discriminator_loss = loss_function(discriminator_output, all_samples_labels)
            discriminator_loss.backward()
            discriminator_optimizer.step()

            # Data for training the generator
            latent_space_samples = torch.randn((GAN_BATCH_SIZE, NUMBER_OF_ATTRIBUTES))

            # Training the generator
            pokemon_generator.zero_grad()
            generated_samples = pokemon_generator(latent_space_samples)
            discriminator_generated_output = pokemon_discriminator(generated_samples)
            generator_loss = loss_function(discriminator_generated_output, real_samples_labels)
            generator_loss.backward()
            generator_optimizer.step()

        # Show loss
        if epoch % 10 == 0:
            print(f"Epoch: {epoch} Discriminator Loss: {discriminator_loss}")
            print(f"Epoch: {epoch} Generator Loss: {generator_loss}")

        last_epoch = epoch
        last_generator_loss = generator_loss
        last_discriminator_loss = discriminator_loss

    print(f'Saving GAN generator model checkpoint to {GAN_GENERATOR_CHECKPOINT_PATH}. '
          f'Epoch: {last_epoch}. Loss: {last_generator_loss}')

    torch.save({
        'epoch': last_epoch + 1,
        'model_state_dict': pokemon_generator.state_dict(),
        'optimizer_state_dict': generator_optimizer.state_dict(),
        'loss': last_generator_loss,
    }, GAN_GENERATOR_CHECKPOINT_PATH)

    print(f'Saving GAN discriminator model checkpoint to {GAN_DISCRIMINATOR_CHECKPOINT_PATH}. '
          f'Epoch: {last_epoch}. Loss: {last_discriminator_loss}')

    torch.save({
        'epoch': last_epoch + 1,
        'model_state_dict': pokemon_discriminator.state_dict(),
        'optimizer_state_dict': discriminator_optimizer.state_dict(),
        'loss': last_discriminator_loss,
    }, GAN_DISCRIMINATOR_CHECKPOINT_PATH)


if __name__ == '__main__':
    train_gan_models()
