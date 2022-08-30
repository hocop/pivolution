from abc import ABC, abstractmethod
import numpy as np

import torch

from . import networks


class Creature(ABC):
    ACTION_NAMES = [
        'nothing',
        'go forward',
        'go backward',
        'turn right',
        'turn left',
        'heal',
        'reproduce',
    ]
    ACTION_COSTS = {
        'nothing': 0.0,
        'go forward': 0.01,
        'go backward': 0.01,
        'turn right': 0.0,
        'turn left': 0.0,
        'heal': 0.1,
        'reproduce': 0.5,
    }
    OMNIVORY = 0.5

    def __init__(self, genes=None):
        self.health = 1.0
        self.energy = 0.25
        self.age = 0

        if genes is None:
            self.genes = self.completely_new_genes()
        else:
            self.genes = genes
        self.predatory = np.clip(self.genes[0], 0, 1)

        self.base_color = np.array([self.predatory, (1 - self.predatory), 0])
        self.base_color = (self.base_color / max(self.base_color) * 200).astype('uint8')

        self.gain_from_sun = 0.004 * np.clip(0.5 + 0.5 * self.OMNIVORY - self.predatory, 0, 1) * (2 - self.OMNIVORY)
        self.gain_from_meat = 0.5 * np.clip(self.predatory - 0.5 + 0.5 * self.OMNIVORY, 0, 1) * (2 - self.OMNIVORY)

        self.features = None
        self.action = 'nothing'

    def compute_action(self):
        '''
        Returns one of actions
        '''
        water_level, air_level, meat_in_this_cell, creature_in_front, local_feats = self.features

        # Photosynthesize
        self.energy += self.gain_from_sun * np.exp(-(water_level * 5 + air_level * 2 + meat_in_this_cell * 5))
        # Eat meat
        self.energy += self.gain_from_meat * meat_in_this_cell
        self.energy = min(self.energy, 1.0)

        # Get action
        action = self.action_from_feats(local_feats)

        # Take action cost
        cost = self.ACTION_COSTS[action]
        if 'go' in action:
            cost += min(water_level, 0.5)
            cost = cost * np.exp(-air_level * 2)
        if cost > self.energy:
            action = 'nothing'
            cost = 0
        self.energy = self.energy - cost

        # Kick someone's ass (D&D rules)
        if action == 'go forward' and creature_in_front is not None:
            # Compute hit or miss
            strength = 5 * self.predatory
            roll = 1 + np.random.randint(1, 21) + strength
            hit = roll >= (10 + creature_in_front.predatory * 4)
            crit = roll == 20
            if hit:
                # Compute damage
                dmg = strength + np.random.randint(1, 5)
                if crit:
                    dmg = dmg + np.random.randint(1, 5)
                # Deal damage
                creature_in_front.health = creature_in_front.health - dmg * 0.1

        # Heal yourself
        if action == 'heal':
            self.health = min(self.health + 0.01, 1.0)

        # Get damage from low energy
        if self.energy < 0.1:
            self.health -= 0.01
        
        # Get damage from age
        self.age += 1
        if self.age > 600 and self.age % 200 == 0:
            dmg = np.random.randint(1, 5)
            self.health -= dmg * 0.1

        self.action = action


    @property
    def color(self):
        return (self.base_color * self.health).astype('uint8')

    def get_offspring(self):
        genes = self.genes.copy()
        # Mutation
        mutant_mask = np.random.random(size=len(genes)) < 0.01
        mutant_genes = self.completely_new_genes()
        new_genes = genes * (1 - mutant_mask) + mutant_genes * mutant_mask
        # new_genes = new_genes + np.random.normal(size=len(genes)) * 0.01
        # Create creature
        offspring = type(self)(genes=new_genes)
        return offspring

    @abstractmethod
    def completely_new_genes(self):
        pass
    
    @abstractmethod
    def action_from_feats(self, feats):
        pass


class CreatureLinear(Creature):
    def action_from_feats(self, feats):
        feats = feats.flatten()
        w_idx = 1 + len(feats) * len(self.ACTION_NAMES)
        logits = feats[None] @ self.genes[1: w_idx].reshape(-1, len(self.ACTION_NAMES))
        logits = logits.flatten() + self.genes[w_idx:]
        action = self.ACTION_NAMES[np.argmax(logits)]
        return action

    def completely_new_genes(self):
        genes = np.random.normal(size=1 + 6 * 5 * 5 * len(self.ACTION_NAMES) + len(self.ACTION_NAMES))
        genes[0] = np.random.random()
        return genes


class CreatureNeural2(Creature):
    def __init__(self, genes=None):
        hidden_size = 32
        self.weight_sizes = [1, 6 * 5 * 5 * hidden_size, hidden_size, hidden_size * len(self.ACTION_NAMES), len(self.ACTION_NAMES)]
        super().__init__(genes)
        self.weights = np.split(self.genes, np.cumsum(self.weight_sizes))[1:]
        self.weights[0] = self.weights[0].reshape(-1, hidden_size)
        self.weights[0] = self.weights[0] / np.sqrt(self.weights[0].shape[0])
        self.weights[1] = self.weights[1][None] / 2
        self.weights[2] = self.weights[2].reshape(-1, len(self.ACTION_NAMES))
        self.weights[2] = self.weights[2] / np.sqrt(self.weights[2].shape[0])
        self.weights[3] = self.weights[3][None] / 2

    def action_from_feats(self, feats):
        feats = feats.flatten()[None]
        hidden = feats @ self.weights[0] + self.weights[1]
        hidden = np.clip(hidden, 0, None)
        logits = hidden @ self.weights[2] + self.weights[3]
        logits = logits.flatten()
        action = self.ACTION_NAMES[np.argmax(logits)]
        return action

    def completely_new_genes(self):
        genes = np.random.normal(size=sum(self.weight_sizes))
        genes[0] = np.random.random()
        return genes


class CreatureRandom(Creature):
    def action_from_feats(self, feats):
        action = np.random.choice(self.ACTION_NAMES, p=self.genes[1:])
        return action

    def completely_new_genes(self):
        genes = np.random.random(size=1 + len(self.ACTION_NAMES))
        genes = np.clip(genes, 0, None)
        genes[1:] = genes[1:] / genes[1:].sum()
        return genes


class CreatureNeural(Creature):
    def __init__(self, genes=None):
        self.net = networks.CNN(6, 5, len(self.ACTION_NAMES))
        super().__init__(genes)
        networks.set_params(self.net, self.genes[1:])
        self.net = torch.jit.trace(self.net, torch.randn([1, 6, 5, 5]))

    def action_from_feats(self, feats):
        feats = torch.tensor(feats[None].copy(), dtype=torch.float)
        logits = self.net(feats).numpy().flatten()
        action = self.ACTION_NAMES[np.argmax(logits)]
        return action

    def completely_new_genes(self):
        params = networks.get_vector(self.net)
        genes = np.concatenate([
            np.random.random(size=1),
            params,
        ], 0)
        return genes