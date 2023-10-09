import numpy as np
import colorsys
import copy

from .gene_arrays import GeneArray
from ..networks import Recurrent
from .creature import Creature
from .creature import NUM_FEATURES, FEAT_WINDOW
from .inheritance import Genome, get_child


class CreatureGendered(Creature):
    N_APPEARANCE_FEATS = 10

    def __init__(self, genome=None):

        # Create neural network (without weights yet)
        self.net = Recurrent(
            n_inputs=NUM_FEATURES * FEAT_WINDOW * FEAT_WINDOW + 2,
            n_outputs=len(self.ACTION_NAMES),
            hidden_size=48,
            state_size=8,
        )

        # Make new genes
        self.genome = genome
        if genome is None:
            genes_a = self.completely_new_genes()
            genes_a['predatory'] = 0.0
            genes_a['boy_prob'] = 0.0
            genes_b = genes_a
            self.genome = Genome(dom_chromosome=genes_a.asarray(), rec_chromosome=genes_b.asarray())

        # Initialize creature from dominant genes
        super().__init__(
            self.new_empty_genes().set_array(self.genome.dom_chromosome)
        )

        # Parse genes
        self.net.set_weights(self.genes['net_params'])

        self.gender = 'boy' if np.random.random() < self.genes['boy_prob'] else 'girl'

        if self.gender == 'boy':
            self.face_color = [0, 255, 255]
            self.action_costs['reproduce'] = self.genes['energy_sharing'] * self.ACTION_COSTS['reproduce']

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
            self.dist_front = np.abs(creature_in_front.genes['appearance'] - self.genes['appearance']).mean()

        super().compute_action()

        # Share genes with creature in front
        if self.dist_front >= 0 and self.action == 'reproduce':
            creature_in_front.shared_genes = self.genome
            creature_in_front.shared_dist = self.dist_front
            creature_in_front.shared_energy = self.action_costs['reproduce']
            self.energy = self.energy - self.action_costs['reproduce']

        # Action to drop shared genes
        if self.gender == 'girl' and self.action == 'go backward':
            self.shared_genes = None
            self.shared_dist = 0
            self.shared_energy = 0

    def action_from_feats(self, feats):
        if self.gender == 'girl':
            feats = np.hstack([self.shared_dist, self.shared_energy, feats.flatten()])
        elif self.gender == 'boy':
            feats = np.hstack([self.dist_front, -1, feats.flatten()])
        logits = self.net.forward(feats)
        action = self.ACTION_NAMES[np.argmax(logits)]
        return action

    def reproduce(self):
        if self.gender == 'boy':
            return

        # Create new genome
        if self.shared_genes is not None:
            new_genome = get_child(self.genome, self.shared_genes)
        else:
            new_genome = get_child(self.genome, self.genome)
        
        # Mutate new genome
        new_genome = Genome(
            dom_chromosome=self.mutate_genes(self.new_empty_genes().set_array(new_genome.dom_chromosome)).asarray(),
            rec_chromosome=self.mutate_genes(self.new_empty_genes().set_array(new_genome.rec_chromosome)).asarray(),
        )

        offspring = CreatureGendered(genome=new_genome)
        if self.shared_genes is not None:
            offspring.middle_color = [0, 0, 0]

        self.energy = self.energy - self.action_costs['reproduce'] + self.shared_energy

        # Reset shared
        self.shared_energy = 0

        return offspring

    def genes_config(self) -> dict[str, int]:
        return {
            'predatory': 1,
            'boy_prob': 1,
            'energy_sharing': 1,
            'net_params': self.net.n_params,
            'appearance': self.N_APPEARANCE_FEATS,
        }

    def completely_new_genes(self) -> GeneArray:
        genes = self.new_empty_genes()
        genes['predatory'] = np.random.random(size=1)
        genes['boy_prob'] = np.random.random(size=1)
        genes['energy_sharing'] = np.random.random(size=1)
        genes['net_params'] = self.net.get_new_params()
        genes['appearance'] = np.random.normal(size=self.N_APPEARANCE_FEATS)
        return genes
