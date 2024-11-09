import numpy as np
from Networks.Izhikevich_neuron import IzNetwork
import matplotlib.pyplot as plt

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
        self.set_neuron_parameters()


    def create_initial_connections(self):
        """
        Create the initial connectivity matrix for the modular network.
        The network consists of multiple modules, each with excitatory and inhibitory neurons.
        (Before Rewiring)
        """
        # Initialize the connectivity matrix with zeros
        self._W = np.zeros((self._N, self._N))

        #Intialize Delay Matrix
        self._D = np.ones((self._N, self._N), dtype = int)

        
        # Divide the network into modules and create connections within each module
        n_modules = 8
        neurons_per_module = self.N_excitatory // n_modules  #100

        for module_index in range(n_modules):
            # Determine the range of neurons for this module
            start_index = module_index * neurons_per_module
            end_index = start_index + neurons_per_module
            
            # Create 1000 random excitatory-to-excitatory connections within the module
            for _ in range(1000):
                src = np.random.randint(start_index, end_index)
                dest = np.random.randint(start_index, end_index)
                while src == dest: 
                    dest = np.random.randint(start_index, end_index)         #For avoiding Self connection
                self._W[src, dest] = 1.0 * 17         #Weight+ Scaled Matrix
                self._D[src, dest] = np.random.randint(1,21)  #Delay Matrix

        # Add inhibitory connections
        used_excitatory_neurons = set()
        for i in range(self.N_excitatory, self._N):
            # Each inhibitory neuron receives connections from exactly four excitatory neurons (within a module)
            module_index = (i - self.N_excitatory) % n_modules #0-7
            start_index = module_index * neurons_per_module
            end_index = start_index + neurons_per_module
            excitatory_indices = set()
            while len(excitatory_indices) < 4:
                candidate = np.random.randint(start_index, end_index)
                if candidate not in used_excitatory_neurons:
                    excitatory_indices.add(candidate)
                    used_excitatory_neurons.add(candidate)
            for src in excitatory_indices:
                self._W[src, i] = np.random.uniform(0, 1.0) * 50  #Scaled Weight
                
            
            # Each inhibitory neuron projects to all other neurons (diffuse inhibition)
            for e in range(self.N_excitatory):
                self._W[i, e] = np.random.uniform(-1.0, 0.0) * 2   #Scaled Wt
            for n in range(self.N_excitatory, self._N):
                self._W[i,n] = np.random.uniform(-1.0, 0.0) * 1          

    def rewire_network(self, p):
        """
        Rewire the network connections with a given rewiring probability p.

        Parameters:
        - p: Probability of rewiring each connection.
        """
        for i in range(self.N_excitatory):
            for j in range(self.N_excitatory):
                # Skip self-connections
                if i == j:
                    continue
                
                # Rewire with probability p
                if np.random.rand() < p:
                    # Remove the existing connection
                    self._W[i, j] = 0
                    self._D[i, j] = 1
                    
                    # Create a new connection to a randomly chosen neuron
                    new_dest = np.random.randint(0, self.N_excitatory)
                    while new_dest == j:
                        new_dest = np.random.randint(0, self.N_excitatory)
                    self._W[i, new_dest] = 1.0 * 17 
                    self._D[i, new_dest] = np.random.randint(1,21)


    def set_neuron_parameters(self):
        """
        Set parameters for the excitatory and inhibitory neurons based on Izhikevich's model.
        """
        # Set different parameters for excitatory and inhibitory populations
        a = np.array([0.02] * self.N_excitatory + [0.02] * self.N_inhibitory)
        b = np.array([0.2] * self.N_excitatory + [0.25] * self.N_inhibitory)
        c = np.array([-65.0] * self.N_excitatory + [-65.0] * self.N_inhibitory)
        d = np.array([8.0] * self.N_excitatory + [2.0] * self.N_inhibitory)
        
        # Use the parent class method to set the parameters
        self.setParameters(a, b, c, d)


    def run_simulation(self, duration_ms,p):
        """
        Run the simulation for the specified duration in milliseconds.

        Parameters:
        - duration_ms: Duration of the simulation in milliseconds.
        """
        if p is not None:
            self.rewire_network(p)

        firings = []
        for t in range(duration_ms):
            # Add background firing using Poisson process
            poisson_vals = np.random.poisson(0.01, size=self._N)
            for neuron_index, pos_val in enumerate(poisson_vals):
                if pos_val > 0:
                    current = np.array([15 if i == neuron_index else 0 for i in range(self._N)])
                    self.setCurrent(current)
            # Update the network state and collect firing indices
            fired_indices = self.update()
            if len(fired_indices) > 0:  # Only log if any neurons fired
                print(f"Time {t}: Neurons fired: {fired_indices}")
            firings.append((t, fired_indices))
        return firings
    
    def plot_connectivity_matrix(self):
        """
        Generate a plot of the matrix connectivity.
        """
        plt.figure(figsize=(10, 8))
        plt.imshow(self._W, cmap='binary', interpolation='nearest')
        plt.colorbar(label='Connection Weight')
        plt.xlabel('Neuron Index')
        plt.ylabel('Neuron Index')
        plt.title('Connectivity Matrix of the Modular Network')
        plt.show()

    def plot_raster(self, firings):
        """
        Generate a raster plot of the neuron firing in a 1000ms run.

        Parameters:
        - firings: List of tuples containing (time, fired_indices) pairs.
        """
        times_exc = []
        neurons_exc = []
        times_inh = []
        neurons_inh = []

        for t, fired_indices in firings:
            # Separate excitatory and inhibitory neurons
            excitatory_fired = [neuron for neuron in fired_indices if neuron < self.N_excitatory]
            inhibitory_fired = [neuron for neuron in fired_indices if neuron >= self.N_excitatory]
            
            times_exc.extend([t] * len(excitatory_fired))
            neurons_exc.extend(excitatory_fired)
            
            times_inh.extend([t] * len(inhibitory_fired))
            neurons_inh.extend(inhibitory_fired)

        plt.figure(figsize=(12, 6))

        # Plot excitatory neurons in blue
        plt.scatter(times_exc, neurons_exc, s=2, c='blue', label='Excitatory Neurons')

        # Plot inhibitory neurons in red
        plt.scatter(times_inh, neurons_inh, s=2, c='red', label='Inhibitory Neurons')

        plt.xlabel('Time (ms)')
        plt.ylabel('Neuron Index')
        plt.title('Raster Plot of Neuron Firing')
        plt.legend(loc='upper right')
        plt.tight_layout()
        print("Delay matrix:", self._D)  # Check the delays
        plt.show()


if __name__ == "__main__":
        network = ModularNetwork()
        firings = network.run_simulation(1000,0.1)
        network.plot_raster(firings)
        network.plot_connectivity_matrix()
         # Check the delays

# Run the simulation for 1000 ms and generate a raster plot


#[Action Required]
'''


1-3 or 3-1 -> Weighted Connection should be availiable  
1. External continoius Noise - Activation 

3. Average Firing Rate

5. Chcek graphs matching with slides (create)

'''