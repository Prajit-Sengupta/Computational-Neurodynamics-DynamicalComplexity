import numpy as np
from .iznetwork import IzNetwork

class ModularNetwork(IzNetwork):
    """
    A class to create and manage a modular network of Izhikevich neurons.
    Inherits from the IzNetwork class and extends functionality to support
    modular, small-world networks with both excitatory and inhibitory neurons.
    """

    def __init__(self, N_excitatory=800, N_inhibitory=200, Dmax=20):
        """
        Initialize the ModularNetwork with given parameters.

        Parameters:
        - N_excitatory: Number of excitatory neurons in the network (default: 800).
        - N_inhibitory: Number of inhibitory neurons in the network (default: 200).
        - Dmax: Maximum conduction delay in milliseconds (default: 20).
        """
        # Call the parent constructor to initialize common attributes
        super().__init__(N_excitatory + N_inhibitory, Dmax)
        
        # Store the number of excitatory and inhibitory neurons
        self.N_excitatory = N_excitatory
        self.N_inhibitory = N_inhibitory
        
        # Create the initial modular connectivity structure
        self.create_initial_connections()

    def create_initial_connections(self):
        """
        Create the initial connectivity matrix for the modular network.
        The network consists of multiple modules, each with excitatory and inhibitory neurons.
        """
        # Initialize the connectivity matrix with zeros
        self._W = np.zeros((self._N, self._N))
        
        # Divide the network into modules and create connections within each module
        n_modules = 8
        neurons_per_module = self.N_excitatory // n_modules

        for module_idx in range(n_modules):
            # Determine the range of neurons for this module
            start_idx = module_idx * neurons_per_module
            end_idx = start_idx + neurons_per_module
            
            # Create 1000 random excitatory-to-excitatory connections within the module
            for _ in range(1000):
                src = np.random.randint(start_idx, end_idx)
                dest = np.random.randint(start_idx, end_idx)
                self._W[src, dest] = 1.0  # Assign a weight of 1 for excitatory-to-excitatory connections

        # Add inhibitory connections
        used_excitatory_indices = set()
        for i in range(self.N_excitatory, self._N):
            # Each inhibitory neuron receives connections from exactly four excitatory neurons (within a module)
            module_idx = (i - self.N_excitatory) % n_modules
            start_idx = module_idx * neurons_per_module
            end_idx = start_idx + neurons_per_module
            excitatory_indices = set()
            while len(excitatory_indices) < 4:
                candidate = np.random.randint(start_idx, end_idx)
                if candidate not in used_excitatory_indices:
                    excitatory_indices.add(candidate)
                    used_excitatory_indices.add(candidate)
            for src in excitatory_indices:
                self._W[src, i] = np.random.uniform(0.5, 1.0)
            
            # Each inhibitory neuron projects to all other neurons (diffuse inhibition)
            self._W[i, :] = np.random.uniform(-1.0, 0.0, self._N)

    def rewire_network(self, p):
        """
        Rewire the network connections with a given rewiring probability p.

        Parameters:
        - p: Probability of rewiring each connection.
        """
        for i in range(self._N):
            for j in range(self._N):
                # Skip self-connections
                if i == j:
                    continue
                
                # Rewire with probability p
                if np.random.rand() < p:
                    # Remove the existing connection
                    self._W[i, j] = 0
                    
                    # Create a new connection to a randomly chosen neuron
                    new_dest = np.random.randint(0, self._N)
                    while new_dest == j:
                        new_dest = np.random.randint(0, self._N)
                    self._W[i, new_dest] = np.random.uniform(0, 1) if i < self.N_excitatory else np.random.uniform(-1.0, 0.0)
                    if new_dest < self.N_excitatory and i < self.N_excitatory:
                      self._W[i, new_dest] = 1

    def set_neuron_parameters(self):
        """
        Set parameters for the excitatory and inhibitory neurons based on Izhikevich's model.
        """
        # Set different parameters for excitatory and inhibitory populations
        a = np.array([0.02] * self.N_excitatory + [0.1] * self.N_inhibitory)
        b = np.array([0.2] * self.N_excitatory + [0.2] * self.N_inhibitory)
        c = np.array([-65.0] * self.N_excitatory + [-65.0] * self.N_inhibitory)
        d = np.array([8.0] * self.N_excitatory + [2.0] * self.N_inhibitory)

        # Use the parent class method to set the parameters
        self.setParameters(a, b, c, d)

    def run_simulation(self, duration_ms):
        """
        Run the simulation for the specified duration in milliseconds.

        Parameters:
        - duration_ms: Duration of the simulation in milliseconds.
        """
        firings = []
        for t in range(duration_ms):
            fired_indices = self.update()
            firings.append((t, fired_indices))
        return firings
