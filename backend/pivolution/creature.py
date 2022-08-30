import numpy as np


class Creature:
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
        'nothing': 0.0001,
        'go forward': 0.01,
        'go backward': 0.01,
        'turn right': 0.001,
        'turn left': 0.001,
        'heal': 0.1,
        'reproduce': 0.5,
    }

    def __init__(self, genes=None):
        self.health = 1.0
        self.energy = 0.25

        if genes is None:
            self.genes = self.completely_new_genes()
        else:
            self.genes = genes
        self.predatory = np.clip(self.genes[0], 0, 1)
        # self.genes = np.clip(self.genes, 0, None)
        # self.probs = self.genes[1:] / self.genes[1:].sum()

        self.base_color = np.array([self.predatory, (1 - self.predatory), 0])
        self.base_color = (self.base_color / max(self.base_color) * 200).astype('uint8')

        self.gain_from_sun = 0.01 * np.clip(0.5 - self.predatory, 0, 1)
        self.gain_from_meat = np.clip(self.predatory - 0.5, 0, 1)

        self.features = None
        self.action = 'nothing'

    def compute_action(self):
        '''
        Returns one of actions
        '''
        water_level, meat_in_this_cell, creature_in_front, local_feats = self.features

        # Photosynthesize
        self.energy += self.gain_from_sun * np.exp(-(water_level * 5 + meat_in_this_cell * 5))
        # Eat meat
        self.energy += self.gain_from_meat * meat_in_this_cell
        self.energy = min(self.energy, 1.0)

        # Get action
        # action = np.random.choice(self.ACTION_NAMES, p=self.probs)
        feats = local_feats.flatten()
        w_idx = 1 + len(feats) * len(self.ACTION_NAMES)
        logits = feats[None] @ self.genes[1: w_idx].reshape(-1, len(self.ACTION_NAMES))
        logits = logits.flatten() + self.genes[w_idx:]
        action = self.ACTION_NAMES[np.argmax(logits)]

        # Take action cost
        cost = self.ACTION_COSTS[action]
        if 'go' in action:
            cost += water_level
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
        new_genes = new_genes + np.random.normal(size=len(genes)) * 0.01
        # Create creature
        offspring = Creature(genes=new_genes)
        return offspring

    def completely_new_genes(self):
        # return np.random.random(size=1 + len(self.ACTION_NAMES))
        genes = np.random.normal(size=1 + 5 * 5 * 5 * len(self.ACTION_NAMES) + len(self.ACTION_NAMES))
        genes[0] = np.random.random()
        return genes