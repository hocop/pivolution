from abc import ABC, abstractmethod
import numpy as np
import colorsys
import copy

from .np_networks import Perceptron, Recurrent

NUM_FEATURES = 8
FEAT_WINDOW = 3


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
        'go forward': 0.0,
        'go backward': 0.001,
        'turn right': 0.001,
        'turn left': 0.001,
        'heal': 0.1,
        'reproduce': 0.5,
    }
    OMNIVORY = 1/3

    def __init__(self, genes=None):
        self.health = 1.0
        self.energy = 0.25
        self.age = 0
        self.healing_efficiency = 0.05
        self.action_costs = copy.deepcopy(self.ACTION_COSTS)

        if genes is None:
            self.genes = self.completely_new_genes()
        else:
            self.genes = genes
        self.predatory = np.clip(self.genes[0], 0, 1)

        self.face_color = [255 - self.color[1]] * 3
        self.middle_color = None

        self.gain_from_sun = np.clip(0.5 + 0.5 * self.OMNIVORY - self.predatory, 0, 1) / (0.5 + 0.5 * self.OMNIVORY)
        self.gain_from_meat = np.clip(self.predatory - 0.5 + 0.5 * self.OMNIVORY, 0, 1) / (0.5 + 0.5 * self.OMNIVORY)
        sum_gain = self.gain_from_sun + self.gain_from_meat
        self.gain_from_sun = 0.002 * self.gain_from_sun / sum_gain
        self.gain_from_meat = 0.8 * self.gain_from_meat / sum_gain

        self.features = None
        self.action = 'nothing'

    def compute_action(self):
        '''
        Returns one of actions
        '''
        water_level, air_level, meat_in_this_cell, creature_in_front, local_feats = self.features

        # Photosynthesize
        self.energy += self.gain_from_sun * np.exp(-(water_level * 5 + meat_in_this_cell * 4))
        # Eat meat
        self.energy += self.gain_from_meat * meat_in_this_cell
        self.energy = min(self.energy, 1.0)

        # Get action
        action = self.action_from_feats(local_feats)

        # Check if can reproduce
        if action == 'reproduce':
            if self.health < 0.5:
                action = 'nothing'

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
        if action == 'go forward' and creature_in_front is not None:
            # Compute hit or miss
            strength = 5 * self.predatory
            armor_class = 10 + creature_in_front.predatory * 2
            roll = 1 + np.random.randint(1, 21) + strength
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
        if action == 'heal':
            self.health = min(self.health + self.healing_efficiency, 1.0)

        # Get damage from age
        self.age += 1
        if self.age > 300 and self.age % 100 == 0:
            dmg = np.random.randint(1, 5)
            self.health -= dmg * 0.1
            self.healing_efficiency = self.healing_efficiency * 0.9

        self.action = action


    @property
    def color(self):
        # Green - herbavores; Yellow & Orange - omnivores; Red - carnivores
        return (np.array(colorsys.hsv_to_rgb((1 - self.predatory) / 3, 0.5 + 0.5 * self.health, 0.25 + 0.5 * self.health)) * 255).astype('uint8')


    def reproduce(self):
        genes = self.genes.copy()
        # Mutation
        mutant_mask = np.random.random(size=len(genes)) < 0.01
        mutant_genes = self.completely_new_genes()
        new_genes = genes * (1 - mutant_mask) + mutant_genes * mutant_mask
        # new_genes = new_genes + np.random.normal(size=len(genes)) * 0.01
        # Create creature
        offspring = type(self)(genes=new_genes)
        self.energy = self.energy - self.action_costs['reproduce']
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
        genes = np.random.normal(size=1 + NUM_FEATURES * FEAT_WINDOW * FEAT_WINDOW * len(self.ACTION_NAMES) + len(self.ACTION_NAMES))
        genes[0] = np.random.random()
        return genes


class CreatureNeural(Creature):
    def __init__(self, genes=None):
        self.net = Perceptron(
            n_inputs=NUM_FEATURES * FEAT_WINDOW * FEAT_WINDOW,
            n_outputs=len(self.ACTION_NAMES),
            hidden_size=32
        )

        super().__init__(genes)

        self.net.set_weights(self.genes[1:])

    def action_from_feats(self, feats):
        feats = feats.flatten()
        logits = self.net.forward(feats)
        action = self.ACTION_NAMES[np.argmax(logits)]
        return action

    def completely_new_genes(self):
        net_parms = self.net.get_new_params()
        predatory = np.random.random(size=1)
        genes = np.hstack([predatory, net_parms])
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

        self.net.set_weights(self.genes[1:])

    def action_from_feats(self, feats):
        feats = feats.flatten()
        logits = self.net.forward(feats)
        action = self.ACTION_NAMES[np.argmax(logits)]
        return action

    def completely_new_genes(self):
        net_parms = self.net.get_new_params()
        predatory = np.random.random(size=1)
        genes = np.hstack([predatory, net_parms])
        return genes


class CreatureGendered(Creature):
    def __init__(self, genes_a=None, genes_b=None):

        # Create neural network (without weights yet)
        self.net = Recurrent(
            n_inputs=NUM_FEATURES * FEAT_WINDOW * FEAT_WINDOW + 2,
            n_outputs=len(self.ACTION_NAMES),
            hidden_size=48,
            state_size=8,
        )

        # Make new genes
        self.genes_a = genes_a
        self.genes_b = genes_b
        if self.genes_b is None:
            assert self.genes_a is None
        if self.genes_a is None:
            assert self.genes_b is None
            self.genes_a = self.completely_new_genes()
            self.genes_b = self.genes_a

        # Merge genes
        genes = np.empty_like(self.genes_a)
        # There are 3 major genes: "predatory", "boy_prob" and "boy_reproduce_cost"
        n_major = 3
        # Average between major genes
        genes[:n_major] = 0.5 * (self.genes_a[:n_major] + self.genes_b[:n_major])
        # For neural genes, take lesser weight as dominant
        dominant_mask = np.abs(self.genes_a[n_major:]) < np.abs(self.genes_b[n_major:])
        genes[n_major:] = dominant_mask * self.genes_a[n_major:] + (1 - dominant_mask) * self.genes_b[n_major:]

        super().__init__(genes)

        # Parse genes
        boy_prob = self.genes[1]
        boy_reproduce_cost = self.genes[2]
        self.net.set_weights(self.genes[2:])

        self.gender = 'boy' if np.random.random() < boy_prob else 'girl'

        # Reset shared
        self.shared_dist = -1
        self.shared_energy = 0

        if self.gender == 'boy':
            self.face_color = [0, 255, 255]
            self.action_costs['reproduce'] = boy_reproduce_cost * self.ACTION_COSTS['reproduce']

        if self.gender == 'girl':
            self.face_color = [255, 0, 255]
            self.shared_genes = None

    def compute_action(self):
        super().compute_action()

        if self.action == 'reproduce' and self.gender == 'boy':
            # Share genes with creature in front
            creature_in_front = self.features[3]
            if isinstance(creature_in_front, CreatureGendered) and creature_in_front.gender == 'girl':
                dist = (np.abs(creature_in_front.genes - self.genes) > 0.05).mean()
                creature_in_front.shared_genes = self.get_recombined_genes()
                creature_in_front.shared_dist = dist
                creature_in_front.shared_energy = self.action_costs['reproduce']
                self.energy = self.energy - self.action_costs['reproduce']

    def action_from_feats(self, feats):
        feats = np.hstack([self.shared_dist, self.shared_energy, feats.flatten()])
        logits = self.net.forward(feats)
        action = self.ACTION_NAMES[np.argmax(logits)]
        return action

    def reproduce(self):
        if self.gender == 'boy':
            return

        # Create creature
        if self.shared_genes is not None:
            father_genes = self.shared_genes
        else:
            father_genes = self.get_recombined_genes()
        offspring = CreatureGendered(genes_a=self.get_recombined_genes(), genes_b=father_genes)
        if self.shared_genes is not None:
            offspring.middle_color = [0, 0, 0]

        self.energy = self.energy - self.action_costs['reproduce'] + self.shared_energy

        # Reset shared
        self.shared_energy = 0

        return offspring

    def get_recombined_genes(self):
        '''
        This is remotely similar to meiosis
        '''
        # Cross-over genes
        cross_over_mask = np.random.random(size=len(self.genes_a)) < 0.5
        genes = cross_over_mask * self.genes_a + (1 - cross_over_mask) * self.genes_b
        # Mutation
        mutant_mask = np.random.random(size=len(genes)) < 0.01
        mutant_genes = self.completely_new_genes()
        genes = genes * (1 - mutant_mask) + mutant_genes * mutant_mask
        return genes

    def completely_new_genes(self):
        net_parms = self.net.get_new_params()
        predatory = np.random.random(size=1)
        boy_prob = np.random.random(size=1)
        boy_reproduce_cost = np.random.random(size=1)
        genes = np.hstack([predatory, boy_prob, boy_reproduce_cost, net_parms])
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
