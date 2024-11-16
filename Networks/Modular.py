import numpy as np
from .Izhikevich_neuron import IzNetwork   #From Default Path
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
        This is initial connectivity matrix for the modular network.
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
            for i in range(1000):
                src = np.random.randint(start_index, end_index)
                dest = np.random.randint(start_index, end_index)
                while src == dest: 
                    dest = np.random.randint(start_index, end_index)      #For avoiding Self connection
                self._W[src, dest] = 1.0 * 17.0         #Scaled Weight as per the table
                self._D[src, dest] = np.random.randint(1,21)  #Delay Matrix
        
        # Add excitatory to inhibitory connections
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
                self._W[src, i] = np.random.uniform(0, 1.0) * 50.0  #Scaled Weight as per the table
            
            
        # Each inhibitory neuron projects to all other neurons 
        for i in range(self.N_excitatory, self._N):
            for e in range(self.N_excitatory):
                self._W[i, e] = np.random.uniform(-1.0, 0.0) * 2.0   #Scaled Weight as per the table
            for n in range(self.N_excitatory, self._N):
                self._W[i,n] = np.random.uniform(-1.0, 0.0) * 1.0   #Scaled Weight as per the table
        
    
    def rewire_network(self, p):
        """
        Rewire the network connections with a given rewiring probability p.
        Parameters:
        - p: Probability of rewiring each connection.
        """
        # Initialize rewired weight and delay matrices
        rewired_W = np.copy(self._W)
        rewired_D = np.copy(self._D)

        # Rewiring process for each excitatory module
        n_modules = 8
        neurons_per_module = self.N_excitatory // n_modules

        for module_index in range(n_modules):
            start_index = module_index * neurons_per_module
            end_index = start_index + neurons_per_module
            
            for i in range(start_index, end_index):
                for j in range(start_index, end_index):
                    if i != j and self._W[i, j] > 0:  # Skip self-connections and non-existent connections
                        if np.random.rand() < p:
                            # Attempt to rewire the connection
                            rewired = False
                            while not rewired:
                                # Generate a random target module different from the source module
                                target_module_index = np.random.choice([idx for idx in range(n_modules) if idx != module_index])
                                target_start = target_module_index * neurons_per_module
                                target_end = target_start + neurons_per_module
                                new_dest = np.random.randint(target_start, target_end)
                                # Check if there is no existing connection between the selected source and target neurons
                                if rewired_W[i, new_dest] == 0:
                                    # Rewire the connection
                                    rewired_W[i, new_dest] = self._W[i, j]
                                    rewired_D[i, new_dest] = self._D[i, j]
                                    # Remove the original connection
                                    rewired_W[i, j] = 0
                                    rewired_D[i, j] = 1
                                    rewired = True
        # Update the network with rewired connections
        self._W = rewired_W
        self._D = rewired_D

    def set_neuron_parameters(self):
        """
        Set parameters for the excitatory and inhibitory neurons based on Izhikevich's model.
        """
        a = np.zeros(self.N_excitatory + self.N_inhibitory)
        b = np.zeros(self.N_excitatory + self.N_inhibitory)
        c = np.zeros(self.N_excitatory + self.N_inhibitory)
        d = np.zeros(self.N_excitatory + self.N_inhibitory)

        # Set parameters for excitatory and inhibitory neurons
        for i in range(self.N_excitatory + self.N_inhibitory):
            r = np.random.uniform(0,1)
            if i < self.N_excitatory:
                # Excitatory neuron parameters
                a[i] = 0.02
                b[i] = 0.2
                c[i] = -65 + 15 * (r ** 2)
                d[i] = 8 - 6 * (r ** 2)
            else:
                # Inhibitory neuron parameters
                a[i] = 0.02 + 0.08 * r
                b[i] = 0.25 - 0.05 * r
                c[i] = -65
                d[i] = 2

        self.setParameters(a, b, c, d)
  

    def run_simulation(self, duration_ms, p):
        """
        Run the simulation for the specified duration in milliseconds.

        Parameters:
        - duration_ms: Duration of the simulation in milliseconds.
        """
        if p is not None:
            self.rewire_network(p)
            
        #Add Transient Time to improve the results of the Neurone Spikes 
        transient_time = 100
        for t in range(transient_time):
        # Add background firing using Poisson process
            poisson_vals = (np.random.poisson(0.01, size=self._N) > 0) *15
            self.setCurrent(poisson_vals)
            # Update the network state without collecting data
            self.update()
        firings = []
        for t in range(duration_ms):
            # Add background firing using Poisson process
            poisson_vals = (np.random.poisson(0.01, size=self._N) > 0) *15
            self.setCurrent(poisson_vals)

            fired_indices = self.update()
            if len(fired_indices) > 0:  # Only log if any neurons fired
                print(f"Time {t}: Neurons fired: {fired_indices}")
            firings.append((t, fired_indices))
        return firings
    
    def plot_connectivity_matrix(self,p):
        """
        Generate a plot of the matrix connectivity.
        """
        # Create a mask: 0 where weights are 0, 1 where weights are non-zero
        mask = np.where(self._W == 0, 1, 0)
        # Create the plot
        plt.imshow(mask, cmap="gray", interpolation='nearest')
        plt.title('Connectivity Matrix of the Modular Network of p={}'.format(p))
        plt.xlabel('Target Neuron')
        plt.ylabel('Source Neuron')
        plt.xlim(0,800)
        plt.ylim(800,0)
        plt.savefig('Connectivity Matrix_p={}.pdf'.format(p), format='pdf')     #For Saving the Plot
        plt.show()


    def plot_raster(self, firings, p):
        """
        Generate a raster plot of the neuron firing in a 1000ms run.
        Parameters:
        - firings: List of tuples containing (time, fired_indices) pairs.
        """
        times_exc = []
        neurons_exc = []

        for t, fired_indices in firings:
            excitatory_fired = [neuron for neuron in fired_indices if neuron < self.N_excitatory]
            times_exc.extend([t] * len(excitatory_fired))
            neurons_exc.extend(excitatory_fired)
        plt.figure(figsize=(12, 3))

        # Excitatory neurons
        plt.scatter(times_exc, neurons_exc, s=16, c='blue', label='Excitatory Neurons')

        plt.xlabel('Time (ms)')
        plt.ylabel('Neuron Index')
        plt.title('Raster Plot of Neuron Firing of p={}'.format(p))
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig('Raster_p={}.pdf'.format(p), format='pdf')
        plt.show()


    def plot_mean_firing_rate(self, firings, T,p, modules_to_plot=8, window_size=50, shift=20):
        """
        Plot the mean firing rate for each module over time.
        Parameters:
        - firings: List of tuples (time, neuron_indices) from run_simulation()
        - T: Total simulation time (in ms)
        - modules_to_plot: Number of modules to plot (default: 8)
        - window_size: Window size for calculating mean firing rate (default: 50 ms) 
        - shift: Step size for sliding window (default: 20 ms)
        """
        # Initialize firing count per module
        neurons_per_module = self.N_excitatory // modules_to_plot
        mean_firing_rates = np.zeros((modules_to_plot, T // shift))

        # Count neuron firings for each module within each time window
        for time_window_start in range(0, T - window_size, shift):
            window_end = time_window_start + window_size
            time_index = time_window_start // shift

            # Get neurons that fired in this time window
            neurons_in_window = [neuron for t, neurons in firings if time_window_start <= t < window_end for neuron in neurons]

            # Calculate mean firing rate for each module
            for module_index in range(modules_to_plot):
                start_idx = module_index * neurons_per_module
                end_idx = start_idx + neurons_per_module

                # Count how many neurons in this module fired in this time window
                count_firings = sum(start_idx <= neuron < end_idx for neuron in neurons_in_window)
                mean_firing_rate = count_firings / window_size

                # Store mean firing rate
                mean_firing_rates[module_index, time_index] = mean_firing_rate

        # Plot the mean firing rates for each module over time
        plt.figure(figsize=(12, 6))
        for module_index in range(modules_to_plot):
            plt.plot(np.arange(0, T, shift), mean_firing_rates[module_index, :], label=f'Module {module_index + 1}')
        
        plt.xlabel('Time (ms)')
        plt.ylabel('Mean Firing Rate')
        plt.title('Mean Firing Rate per Module Over Time of p={}'.format(p))
        plt.legend()
        plt.tight_layout()
        plt.savefig('Firing_Rate_p={}.pdf'.format(p), format='pdf')  
        plt.show()
    

#For Running this function
if __name__ == "__main__":
        network = ModularNetwork()
        p=[0.4]
        for i in p:
            firings = network.run_simulation(1000,i)
            network.plot_raster(firings,i)
            network.plot_mean_firing_rate(firings, 1000,i)
            network.plot_connectivity_matrix(i)