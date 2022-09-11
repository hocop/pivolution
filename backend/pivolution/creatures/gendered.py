import numpy as np
import colorsys
import copy

from ..networks import Perceptron, Recurrent
from ..creatures.basic import Creature
from ..creatures.basic import NUM_FEATURES, FEAT_WINDOW


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
            self.genes_a[0] = 0.0 # predatory
            self.genes_a[1] = 0.0 # boy prob
            self.genes_b = self.genes_a

        # Merge genes
        genes = 0.5 * (self.genes_a + self.genes_b)

        super().__init__(genes)

        # Parse genes
        boy_prob = self.genes[1]
        boy_reproduce_cost = self.genes[2]
        self.net.set_weights(self.genes[2:])

        self.gender = 'boy' if np.random.random() < boy_prob else 'girl'

        if self.gender == 'boy':
            self.face_color = [0, 255, 255]
            self.action_costs['reproduce'] = boy_reproduce_cost * self.ACTION_COSTS['reproduce']

        if self.gender == 'girl':
            self.face_color = [255, 0, 255]
            self.shared_genes = None
            self.shared_dist = 0
            self.shared_energy = 0

    def compute_action(self):
        # Look at creature in front
        self.dist_front = -0.1
        creature_in_front = self.features[3]
        if self.gender == 'boy' and isinstance(creature_in_front, CreatureGendered) and creature_in_front.gender == 'girl':
            self.dist_front = (np.abs(creature_in_front.genes - self.genes) > 0.05).mean()

        super().compute_action()

        # Share genes with creature in front
        if self.action == 'reproduce' and self.dist_front >= 0:
            creature_in_front.shared_genes = self.get_recombined_genes()
            creature_in_front.shared_dist = self.dist_front
            creature_in_front.shared_energy = self.action_costs['reproduce']
            self.energy = self.energy - self.action_costs['reproduce']

    def action_from_feats(self, feats):
        if self.gender == 'girl':
            feats = np.hstack([self.shared_dist, self.shared_energy, feats.flatten()])
        elif self.gender == 'boy':
            feats = np.hstack([self.dist_front, 0, feats.flatten()])
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