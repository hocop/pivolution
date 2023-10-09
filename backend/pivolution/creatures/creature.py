from abc import ABC, abstractmethod
import numpy as np
import colorsys
import copy

from ..networks import Perceptron, Recurrent
from .gene_arrays import GeneArray

NUM_FEATURES = 8
FEAT_WINDOW = 5


class Creature(ABC):
    ACTION_NAMES = [
        'nothing',
        'go forward',
        'go backward',
        'turn right',
        'turn left',
        'reproduce',
    ]
    ACTION_COSTS = {
        'nothing': 0.0,
        'go forward': 0.0,
        'go backward': 0.001,
        'turn right': 0.001,
        'turn left': 0.001,
        'reproduce': 0.5,
        'attack': 0.1 * 16/20 * 0.1 * (5 + 2.5),
    }
    OMNIVORY = 1/3

    def __init__(self, genes: GeneArray = None):
        self.health = 1.0
        self.energy = 0.25
        self.age = 0
        self.action_costs = copy.deepcopy(self.ACTION_COSTS)
        self.healing_efficiency = 0.01 * 0.25

        if genes is None:
            self.genes = self.completely_new_genes()
            self.genes['predatory'] = 0
        else:
            self.genes = genes
        self.predatory = np.clip(self.genes['predatory'], 0, 1)

        self.face_color = [255 - self.color[1]] * 3
        self.middle_color = None

        self.gain_from_sun = np.clip(0.5 + 0.5 * self.OMNIVORY - self.predatory, 0, 1) / (0.5 + 0.5 * self.OMNIVORY)
        self.gain_from_meat = np.clip(self.predatory - 0.5 + 0.5 * self.OMNIVORY, 0, 1) / (0.5 + 0.5 * self.OMNIVORY)
        self.gain_from_sun = 0.002 * self.gain_from_sun
        self.gain_from_meat = 0.8 * (self.gain_from_meat > 0)

        self.features = None
        self.action = 'nothing'

    def compute_action(self):
        '''
        Returns one of actions
        '''
        water_level, air_level, meat_in_this_cell, creature_in_front, local_feats = self.features

        # Photosynthesize
        self.energy += self.gain_from_sun * np.exp(-(meat_in_this_cell * 4)) * (2 - np.exp(-air_level))
        # Eat meat
        self.energy += self.gain_from_meat * meat_in_this_cell
        self.energy = min(self.energy, 1.0)

        # Get action
        action = self.action_from_feats(local_feats)

        # Check if action is attack
        if action == 'go forward' and creature_in_front is not None:
            action = 'attack'

        # Take action cost
        cost = self.action_costs[action]
        if 'go' in action:
            cost += min(water_level, 0.1)
        if cost > self.energy:
            action = 'nothing'
            cost = 0
        if action != 'reproduce':
            self.energy = self.energy - cost

        # Kick someone's ass (D&D rules)
        if action == 'attack':
            # Compute hit or miss
            strength = int(np.round(5 * self.predatory))
            armor_class = 10
            roll = np.random.randint(1, 21) + strength
            hit = roll >= armor_class
            crit = roll == 20
            if hit:
                # Compute damage
                dmg = strength + np.random.randint(1, 5)
                if crit:
                    dmg = dmg + np.random.randint(1, 5)
                # Deal damage
                creature_in_front.health = creature_in_front.health - dmg * 0.1

        # Heal yourself
        self.health = min(self.health + self.healing_efficiency, 1.0)

        # Get damage from age
        self.age += 1
        if self.age > 0 and self.age % 100 == 0:
            dmg = np.random.randint(1, 5)
            self.health -= dmg * 0.1
            self.healing_efficiency = self.healing_efficiency * 0.5

        self.action = action


    @property
    def color(self):
        # Green - herbavores; Yellow & Orange - omnivores; Red - carnivores
        return (np.array(colorsys.hsv_to_rgb((1 - self.predatory) / 3, 0.5 + 0.5 * self.health, 0.25 + 0.5 * self.health)) * 255).astype('uint8')


    def reproduce(self):
        # Mutation
        new_genes = self.mutate_genes(self.genes)
        # Create creature
        offspring = type(self)(genes=new_genes)
        self.energy = self.energy - self.action_costs['reproduce']
        return offspring

    def mutate_genes(self, genes: GeneArray):
        mutant_mask = np.random.random(size=len(genes)) < 0.01
        mutant_genes = self.completely_new_genes()
        new_arr = np.where(mutant_mask, mutant_genes.asarray(), genes.asarray())
        mutant_genes.set_array(new_arr)
        return mutant_genes

    @abstractmethod
    def completely_new_genes(self) -> GeneArray:
        pass

    @abstractmethod
    def action_from_feats(self, feats):
        pass

    @abstractmethod
    def genes_config(self) -> dict[str, int]:
        pass

    def new_empty_genes(self) -> GeneArray:
        return GeneArray(self.genes_config())


class CreatureNeural(Creature):
    def __init__(self, genes=None):
        self.net = Perceptron(
            n_inputs=NUM_FEATURES * FEAT_WINDOW * FEAT_WINDOW,
            n_outputs=len(self.ACTION_NAMES),
            hidden_size=32
        )

        super().__init__(genes)

        self.net.set_weights(self.genes['net_params'])

    def action_from_feats(self, feats):
        feats = feats.flatten()
        logits = self.net.forward(feats)
        action = self.ACTION_NAMES[np.argmax(logits)]
        return action

    def genes_config(self) -> dict[str, int]:
        return {
            'predatory': 1,
            'net_params': self.net.n_params,
        }

    def completely_new_genes(self):
        genes = self.new_empty_genes()
        genes['net_params'] = self.net.get_new_params()
        genes['predatory'] = np.random.random(size=1)
        return genes


class CreatureRecurrent(Creature):
    def __init__(self, genes=None):

        self.net = Recurrent(
            n_inputs=NUM_FEATURES * FEAT_WINDOW * FEAT_WINDOW,
            n_outputs=len(self.ACTION_NAMES),
            hidden_size=48,
            state_size=8,
        )

        super().__init__(genes)

        self.net.set_weights(self.genes['net_params'])

    def action_from_feats(self, feats):
        feats = feats.flatten()
        logits = self.net.forward(feats)
        action = self.ACTION_NAMES[np.argmax(logits)]
        return action

    def genes_config(self) -> dict[str, int]:
        return {
            'predatory': 1,
            'net_params': self.net.n_params,
        }

    def completely_new_genes(self):
        genes = self.new_empty_genes()
        genes['net_params'] = self.net.get_new_params()
        genes['predatory'] = np.random.random(size=1)
        return genes


class CreatureRandom(Creature):
    def action_from_feats(self, feats):
        action = np.random.choice(self.ACTION_NAMES, p=self.genes['action_probs'])
        return action

    def genes_config(self) -> dict[str, int]:
        return {
            'predatory': 1,
            'action_probs': len(self.ACTION_NAMES),
        }

    def completely_new_genes(self):
        genes = self.new_empty_genes()
        genes['predatory'] = np.random.random(size=1)
        genes['action_probs'] = np.random.random(size=len(self.ACTION_NAMES))
        genes['action_probs'] = genes['action_probs'] / genes['action_probs'].sum()
        return genes
