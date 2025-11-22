
"""
Infectious Disease Spread Simulation in Social Networks
A comprehensive simulation tool for modeling COVID-19-like disease spread
through synthetic social networks with force-directed visualization.
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import numpy as np
import random
import math
from collections import deque, defaultdict
import json
from datetime import datetime
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

# --- Original Classes (SocialNetwork, ForceDirectedLayout, DiseaseSpreadSimulator) ---

class SocialNetwork:
    """
    Generates and manages synthetic social networks using various models.
    Implements efficient data structures for network representation.
    """

    def __init__(self, n_nodes, model='small_world', **params):
        """
        Initialize social network.

        Args:
            n_nodes: Number of individuals in the network
            model: 'small_world', 'scale_free', or 'random'
            params: Model-specific parameters
        """
        self.n_nodes = n_nodes
        self.model = model
        self.adjacency_list = defaultdict(set)  # O(1) lookup for neighbors
        self.edges = []
        self.nodes = list(range(n_nodes))

        # Node attributes
        self.positions = {}  # Force-directed layout positions
        self.velocities = {}  # For spring embedder algorithm
        self.status = {}  # 'S' (Susceptible), 'I' (Infected), 'R' (Recovered)
        self.infection_time = {}

        # Generate network based on model
        if model == 'small_world':
            self._generate_small_world(
                params.get('k', 6), params.get('p', 0.1))
        elif model == 'scale_free':
            self._generate_scale_free(params.get('m', 3))
        elif model == 'random':
            self._generate_random(params.get('p', 0.01))

        # Initialize node states
        for node in self.nodes:
            self.status[node] = 'S'
            self.infection_time[node] = -1
            # Random initial positions
            self.positions[node] = [
                random.uniform(-1, 1), random.uniform(-1, 1)]
            self.velocities[node] = [0.0, 0.0]

    def _generate_small_world(self, k, p):
        """
        Watts-Strogatz small-world network model.
        High clustering with short average path lengths.
        """
        n = self.n_nodes

        # Create ring lattice
        for i in range(n):
            for j in range(1, k // 2 + 1):
                neighbor = (i + j) % n
                self._add_edge(i, neighbor)

        # Rewire edges with probability p
        edges_copy = list(self.edges)
        for u, v in edges_copy:
            if random.random() < p:
                # Remove edge
                self.adjacency_list[u].discard(v)
                self.adjacency_list[v].discard(u)

                # Add new random edge
                new_neighbor = random.randint(0, n - 1)
                while new_neighbor == u or new_neighbor in self.adjacency_list[u]:
                    new_neighbor = random.randint(0, n - 1)

                self._add_edge(u, new_neighbor)

        self.edges = [
            (u, v) for u in self.adjacency_list for v in self.adjacency_list[u] if u < v]

    def _generate_scale_free(self, m):
        """
        Barabási-Albert scale-free network model.
        Preferential attachment: rich get richer.
        """
        # Start with small complete graph
        initial_nodes = min(m + 1, self.n_nodes)
        for i in range(initial_nodes):
            for j in range(i + 1, initial_nodes):
                self._add_edge(i, j)

        # Add remaining nodes with preferential attachment
        for new_node in range(initial_nodes, self.n_nodes):
            # Calculate attachment probabilities based on degree
            degrees = [len(self.adjacency_list[node])
                       for node in range(new_node)]
            total_degree = sum(degrees)

            if total_degree == 0:
                probabilities = [1.0 / new_node] * new_node
            else:
                probabilities = [d / total_degree for d in degrees]

            # Select m nodes to connect to
            targets = set()
            while len(targets) < min(m, new_node):
                target = random.choices(
                    range(new_node), weights=probabilities)[0]
                targets.add(target)

            for target in targets:
                self._add_edge(new_node, target)

    def _generate_random(self, p):
        """
        Erdős-Rényi random graph model.
        """
        for i in range(self.n_nodes):
            for j in range(i + 1, self.n_nodes):
                if random.random() < p:
                    self._add_edge(i, j)

    def _add_edge(self, u, v):
        """Add undirected edge between nodes u and v."""
        self.adjacency_list[u].add(v)
        self.adjacency_list[v].add(u)

    def get_neighbors(self, node):
        """Get neighbors of a node in O(1) time."""
        return self.adjacency_list[node]

    def get_degree(self, node):
        """Get degree of a node in O(1) time."""
        return len(self.adjacency_list[node])

    def get_network_stats(self):
        """Calculate network statistics."""
        degrees = [self.get_degree(node) for node in self.nodes]
        return {
            'nodes': self.n_nodes,
            'edges': len(self.edges),
            'avg_degree': np.mean(degrees),
            'max_degree': max(degrees),
            'min_degree': min(degrees),
            'clustering': self._calculate_clustering()
        }

    def _calculate_clustering(self):
        """Calculate average clustering coefficient."""
        coefficients = []
        for node in self.nodes:
            neighbors = list(self.get_neighbors(node))
            k = len(neighbors)
            if k < 2:
                continue

            # Count triangles
            triangles = 0
            for i in range(len(neighbors)):
                for j in range(i + 1, len(neighbors)):
                    if neighbors[j] in self.get_neighbors(neighbors[i]):
                        triangles += 1

            coeff = 2 * triangles / (k * (k - 1))
            coefficients.append(coeff)

        return np.mean(coefficients) if coefficients else 0.0


class ForceDirectedLayout:
    """
    Implements Spring Embedder algorithm for graph visualization.
    Uses Fruchterman-Reingold force-directed placement.
    """

    def __init__(self, network, width=800, height=600):
        self.network = network
        self.width = width
        self.height = height
        # Optimal distance
        self.k = math.sqrt((width * height) / network.n_nodes)
        self.temperature = width / 10  # Initial temperature for simulated annealing
        self.iterations = 0
        self.max_iterations = 500

    def calculate_repulsive_force(self, dist):
        """Repulsive force between all node pairs (Coulomb's law)."""
        if dist < 0.01:
            dist = 0.01
        return (self.k * self.k) / dist

    def calculate_attractive_force(self, dist):
        """Attractive force between connected nodes (Hooke's law)."""
        return (dist * dist) / self.k

    def iterate(self, iterations=1):
        """
        Perform force-directed layout iterations.
        """
        for _ in range(iterations):
            if self.iterations >= self.max_iterations:
                return False

            # Calculate repulsive forces between all pairs
            displacements = {node: [0.0, 0.0] for node in self.network.nodes}

            for i, v in enumerate(self.network.nodes):
                for u in self.network.nodes[i + 1:]:
                    delta_x = self.network.positions[v][0] - \
                        self.network.positions[u][0]
                    delta_y = self.network.positions[v][1] - \
                        self.network.positions[u][1]
                    dist = math.sqrt(delta_x**2 + delta_y**2)

                    if dist > 0:
                        force = self.calculate_repulsive_force(dist)
                        displacements[v][0] += (delta_x / dist) * force
                        displacements[v][1] += (delta_y / dist) * force
                        displacements[u][0] -= (delta_x / dist) * force
                        displacements[u][1] -= (delta_y / dist) * force

            # Calculate attractive forces for edges
            for u, v in self.network.edges:
                delta_x = self.network.positions[v][0] - \
                    self.network.positions[u][0]
                delta_y = self.network.positions[v][1] - \
                    self.network.positions[u][1]
                dist = math.sqrt(delta_x**2 + delta_y**2)

                if dist > 0:
                    force = self.calculate_attractive_force(dist)
                    displacements[v][0] -= (delta_x / dist) * force
                    displacements[v][1] -= (delta_y / dist) * force
                    displacements[u][0] += (delta_x / dist) * force
                    displacements[u][1] += (delta_y / dist) * force

            # Update positions with temperature cooling
            for node in self.network.nodes:
                disp_length = math.sqrt(
                    displacements[node][0]**2 + displacements[node][1]**2)
                if disp_length > 0:
                    self.network.positions[node][0] += (
                        displacements[node][0] / disp_length) * min(disp_length, self.temperature)
                    self.network.positions[node][1] += (
                        displacements[node][1] / disp_length) * min(disp_length, self.temperature)

                # Keep within bounds
                self.network.positions[node][0] = max(-self.width/2, min(
                    self.width/2, self.network.positions[node][0]))
                self.network.positions[node][1] = max(-self.height/2, min(
                    self.height/2, self.network.positions[node][1]))

            # Cool temperature
            self.temperature *= 0.95
            self.iterations += 1

        return True


class DiseaseSpreadSimulator:
    """
    Simulates infectious disease spread using SIR model with social network interactions.
    """

    def __init__(self, network, transmission_prob=0.05, recovery_time=14,
                 initial_infected=5, interaction_model='uniform'):
        """
        Initialize disease spread simulator.
        """
        self.network = network
        self.transmission_prob = transmission_prob
        self.recovery_time = recovery_time
        self.interaction_model = interaction_model
        self.current_day = 0

        # Statistics tracking
        self.susceptible_count = [network.n_nodes]
        self.infected_count = [0]
        self.recovered_count = [0]
        self.daily_new_infections = [0]

        # Initialize infections
        self._initialize_infections(initial_infected)

    def _initialize_infections(self, n_infected):
        """
        Initialize patient zero(s).
        """
        # Degree-based selection: infect high-degree nodes (superspreaders)
        if self.interaction_model == 'degree_based':
            degrees = [(node, self.network.get_degree(node))
                       for node in self.network.nodes]
            degrees.sort(key=lambda x: x[1], reverse=True)
            infected_nodes = [node for node, _ in degrees[:n_infected]]
        else:
            # Random selection
            infected_nodes = random.sample(self.network.nodes, n_infected)

        for node in infected_nodes:
            self.network.status[node] = 'I'
            self.network.infection_time[node] = self.current_day

        self.infected_count[0] = n_infected
        self.susceptible_count[0] = self.network.n_nodes - n_infected

    def simulate_day(self):
        """
        Simulate one day of disease spread.
        """
        new_infections = []
        nodes_to_recover = []

        # Find currently infected nodes
        infected_nodes = [node for node in self.network.nodes
                          if self.network.status[node] == 'I']

        # Simulate interactions and transmission
        for infected_node in infected_nodes:
            # Check recovery
            if self.current_day - self.network.infection_time[infected_node] >= self.recovery_time:
                nodes_to_recover.append(infected_node)
                continue

            # Simulate interactions with neighbors
            neighbors = list(self.network.get_neighbors(infected_node))

            # Determine number of interactions based on model
            if self.interaction_model == 'uniform':
                n_interactions = len(neighbors)
            elif self.interaction_model == 'degree_based':
                # High-degree nodes interact more
                n_interactions = min(
                    len(neighbors), max(1, len(neighbors) // 2))
            else:
                n_interactions = len(neighbors)

            # Randomly select neighbors to interact with
            if n_interactions < len(neighbors):
                interacting_neighbors = random.sample(
                    neighbors, n_interactions)
            else:
                interacting_neighbors = neighbors

            # Attempt transmission
            for neighbor in interacting_neighbors:
                if self.network.status[neighbor] == 'S':
                    if random.random() < self.transmission_prob:
                        new_infections.append(neighbor)

        # Apply new infections
        for node in new_infections:
            self.network.status[node] = 'I'
            self.network.infection_time[node] = self.current_day

        # Apply recoveries
        for node in nodes_to_recover:
            self.network.status[node] = 'R'

        # Update statistics
        self.current_day += 1
        susceptible = sum(
            1 for node in self.network.nodes if self.network.status[node] == 'S')
        infected = sum(
            1 for node in self.network.nodes if self.network.status[node] == 'I')
        recovered = sum(
            1 for node in self.network.nodes if self.network.status[node] == 'R')

        self.susceptible_count.append(susceptible)
        self.infected_count.append(infected)
        self.recovered_count.append(recovered)
        self.daily_new_infections.append(len(new_infections))

        return len(new_infections), len(nodes_to_recover)

    def get_statistics(self):
        """Get current simulation statistics."""
        if self.network is None:
            return {
                'day': 0,
                'susceptible': 0,
                'infected': 0,
                'recovered': 0,
                'total_infected': 0,
                'new_infections': 0,
                'attack_rate': 0.0
            }
            
        total_nodes = self.network.n_nodes
        if total_nodes == 0:
            attack_rate = 0.0
        else:
            attack_rate = (total_nodes - self.susceptible_count[-1]) / total_nodes
            
        return {
            'day': self.current_day,
            'susceptible': self.susceptible_count[-1],
            'infected': self.infected_count[-1],
            'recovered': self.recovered_count[-1],
            'total_infected': total_nodes - self.susceptible_count[-1],
            'new_infections': self.daily_new_infections[-1],
            'attack_rate': attack_rate
        }


# --- New Tooltip Class for enhanced interactivity ---

class ToolTip:
    """
    Class to create a tooltip.
    A helper for making the GUI more layman-friendly.
    """
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tipwindow = None
        self.id = None
        self.x = self.y = 0
        widget.bind('<Enter>', self.enter)
        widget.bind('<Leave>', self.leave)

    def enter(self, event=None):
        self.schedule()

    def leave(self, event=None):
        self.unschedule()
        self.hidetip()

    def schedule(self):
        self.unschedule()
        self.id = self.widget.after(500, self.showtip)

    def unschedule(self):
        id = self.id
        self.id = None
        if id:
            self.widget.after_cancel(id)

    def showtip(self, event=None):
        x = y = 0
        x, y, cx, cy = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 20

        self.tipwindow = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry("+%d+%d" % (x, y))

        label = tk.Label(tw, text=self.text, justify=tk.LEFT,
                         background="#ffffe0", relief=tk.SOLID, borderwidth=1,
                         font=("tahoma", "8", "normal"))
        label.pack(ipadx=1)

    def hidetip(self):
        tw = self.tipwindow
        self.tipwindow = None
        if tw:
            tw.destroy()


# --- Refactored GUI Class (with fix) ---

class SimulatorGUI:
    """
    Main GUI application for disease spread simulation with improved aesthetics.
    """
    def __init__(self, root):
        self.root = root
        self.root.title("🦠 Infectious Disease Spread Simulation (SIR Model)")
        self.root.geometry("1400x900")
        
        # Apply a modern theme
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.style.configure('TLabel', font=('Arial', 10))
        self.style.configure('TButton', font=('Arial', 10, 'bold'), padding=6)
        self.style.configure('TFrame', background='#f0f0f0')
        self.style.configure('TCombobox', font=('Arial', 10))
        
        # Color mapping for visualization
        self.COLOR_MAP = {
            'S': 'deepskyblue',  # Susceptible
            'I': 'red',          # Infected
            'R': 'forestgreen'   # Recovered
        }

        self.network = None
        self.simulator = None
        self.layout = None
        self.running = False

        # --- FIX: INITIALIZE TKINTER VARIABLES HERE ---
        # The original code called self._add_setting which tried to access
        # these variables before they were created.
        self.size_var = tk.IntVar(value=500)
        self.model_var = tk.StringVar(value='small_world')
        self.trans_var = tk.DoubleVar(value=0.05)
        self.recovery_var = tk.IntVar(value=14)
        self.infected_var = tk.IntVar(value=5)
        # ---------------------------------------------

        self._create_widgets()

    def _create_widgets(self):
        """Create an aesthetically pleasing GUI layout."""
        
        # Main Layout: Two Columns (Controls/Stats on left, Visualization/Plot on right)
        main_frame = ttk.Frame(self.root, padding="10 10 10 10")
        main_frame.pack(fill='both', expand=True)
        main_frame.grid_columnconfigure(0, weight=0)  # Fixed width for control panel
        main_frame.grid_columnconfigure(1, weight=1)  # Expandable width for viz/plot
        main_frame.grid_rowconfigure(0, weight=1)

        # ------------------- Left Column: Controls and Stats -------------------
        left_panel = ttk.Frame(main_frame, padding="5")
        left_panel.grid(row=0, column=0, sticky='nsew', padx=10, pady=10)
        
        # --- Control Panel ---
        control_frame = ttk.LabelFrame(
            left_panel, text="⚙️ Simulation Settings", padding="10 10 10 10")
        control_frame.pack(fill='x', pady=10)
        control_frame.columnconfigure(1, weight=1)
        
        # Network Parameters
        self._add_setting(control_frame, 0, "Population Size (N):", self.size_var, 500, "The total number of individuals in the simulation (Nodes).")
        self._add_setting(control_frame, 1, "Network Type:", self.model_var, 'small_world', "Model used to connect people. Small-World has high clustering, Scale-Free has hubs (superspreaders).")
        
        # Disease Parameters
        self._add_setting(control_frame, 2, "Transmission Probability (P_T):", self.trans_var, 0.05, "The chance of an infection occurring during a contact (0.0 to 1.0).")
        self._add_setting(control_frame, 3, "Recovery Time (Days):", self.recovery_var, 14, "How long it takes for an infected person to recover (I -> R).")
        self._add_setting(control_frame, 4, "Initial Infected:", self.infected_var, 5, "The starting number of 'Patient Zero' individuals.")
        
        # --- Action Buttons ---
        button_frame = ttk.Frame(left_panel, padding="10 5")
        button_frame.pack(fill='x', pady=5)
        
        ttk.Button(button_frame, text="1. Generate Network", command=self.generate_network).pack(fill='x', pady=3)
        ttk.Button(button_frame, text="2. Start/Pause Simulation", command=self.toggle_simulation).pack(fill='x', pady=3)
        ttk.Button(button_frame, text="3. Step Day", command=self.step_simulation).pack(fill='x', pady=3)
        ttk.Button(button_frame, text="Reset Simulation", command=self.reset_simulation, style='Danger.TButton').pack(fill='x', pady=10)
        self.style.configure('Danger.TButton', background='tomato', foreground='white')

        # --- Statistics Display ---
        stats_frame = ttk.LabelFrame(
            left_panel, text="📊 Current Status", padding="10 10 10 10")
        stats_frame.pack(fill='both', expand=True, pady=10)

        self.stats_labels = {}
        row = 0
        for key, text in [('day', 'Day:'), ('susceptible', 'Susceptible (S):'), ('infected', 'Infected (I):'), 
                          ('recovered', 'Recovered (R):'), ('total_infected', 'Total Cases:'), 
                          ('attack_rate', 'Attack Rate:')]:
            ttk.Label(stats_frame, text=text, font=('Arial', 10, 'bold')).grid(row=row, column=0, sticky='w', pady=2)
            self.stats_labels[key] = ttk.Label(stats_frame, text="N/A", font=('Arial', 10))
            self.stats_labels[key].grid(row=row, column=1, sticky='e', pady=2)
            row += 1
        
        self.log_text = scrolledtext.ScrolledText(
            stats_frame, height=5, width=30, wrap=tk.WORD, state=tk.DISABLED, background='#e8e8e8')
        self.log_text.grid(row=row, column=0, columnspan=2, sticky='nsew', pady=(10, 0))
        ttk.Label(stats_frame, text="Simulation Log:", font=('Arial', 9)).grid(row=row-1, column=0, sticky='sw', pady=(5,0))
        stats_frame.grid_rowconfigure(row, weight=1)

        # ------------------- Right Column: Visualization and Plot -------------------
        right_panel = ttk.Frame(main_frame, padding="5")
        right_panel.grid(row=0, column=1, sticky='nsew', padx=10, pady=10)
        right_panel.grid_rowconfigure(0, weight=3) # Canvas takes more space
        right_panel.grid_rowconfigure(1, weight=1) # Plot takes less space
        right_panel.grid_columnconfigure(0, weight=1)

        # --- Visualization Canvas ---
        viz_frame = ttk.LabelFrame(
            right_panel, text="🌐 Social Network Spread", padding="10")
        viz_frame.grid(row=0, column=0, sticky='nsew', pady=(0, 5))
        
        self.canvas = tk.Canvas(viz_frame, bg='#ffffff')
        self.canvas.pack(fill='both', expand=True)
        self.canvas.bind("<Configure>", self._on_canvas_resize) # Handle dynamic resizing
        self.canvas_width = 900
        self.canvas_height = 600

        # --- SIR Plot ---
        plot_frame = ttk.LabelFrame(
            right_panel, text="📈 SIR Model Over Time", padding="5")
        plot_frame.grid(row=1, column=0, sticky='nsew', pady=(5, 0))

        self.fig = Figure(figsize=(5, 3), dpi=100)
        self.plot = self.fig.add_subplot(111)
        self.plot_canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.plot_canvas_widget = self.plot_canvas.get_tk_widget()
        self.plot_canvas_widget.pack(fill='both', expand=True)
        self.update_plot(reset=True) # Draw initial empty plot

    def _add_setting(self, parent, row, label_text, var_obj, default_value, tooltip_text):
        """Helper to create labeled setting rows with tooltips."""
        ttk.Label(parent, text=label_text).grid(row=row, column=0, sticky='w', pady=2)
        
        if isinstance(default_value, str):
            # This is a Combobox (Network Type)
            widget = ttk.Combobox(parent, textvariable=var_obj, values=['small_world', 'scale_free', 'random'], width=15, state='readonly')
            widget.grid(row=row, column=1, sticky='ew', padx=5, pady=2)
        else:
            # This is an Entry (Number inputs)
            widget = ttk.Entry(parent, textvariable=var_obj, width=15)
            widget.grid(row=row, column=1, sticky='ew', padx=5, pady=2)
        
        # Add ToolTip
        ToolTip(widget, tooltip_text)
        
        info_label = ttk.Label(parent, text="ℹ️", cursor="question_arrow")
        ToolTip(info_label, tooltip_text) # Add info icon with tooltip
        info_label.grid(row=row, column=2, sticky='e', padx=(0, 5))
        
        
    def _on_canvas_resize(self, event):
        """Update canvas dimensions on resize."""
        self.canvas_width = event.width
        self.canvas_height = event.height
        self.visualize_network()

    # --- Simulation Methods (Updated for GUI interaction) ---

    def generate_network(self):
        """Generate social network based on parameters."""
        if self.running:
            messagebox.showwarning("Warning", "Please pause or reset the simulation before generating a new network.")
            return
            
        try:
            n = self.size_var.get()
            model = self.model_var.get()

            if n < 10 or n > 5000:
                messagebox.showerror("Error", "Population Size must be between 10 and 5000")
                return

            self._log_message(f"Generating **{model.replace('_', ' ').title()}** network with **{n}** nodes...")
            self.root.update()

            # Generate network
            if model == 'small_world':
                self.network = SocialNetwork(n, model='small_world', k=6, p=0.1)
            elif model == 'scale_free':
                self.network = SocialNetwork(n, model='scale_free', m=3)
            else:
                self.network = SocialNetwork(n, model='random', p=0.01)

            # Reset layout with new dimensions
            self.layout = ForceDirectedLayout(
                self.network, width=self.canvas_width, height=self.canvas_height)

            # Compute initial layout
            self._log_message("Computing force-directed layout (stabilizing nodes)...")
            self.root.update()

            # Run layout iterations quickly
            for _ in range(30):
                self.layout.iterate(10)
                if _ % 5 == 0:
                    self.visualize_network()
                    self.root.update()
                
            self._log_message("Layout stable.")

            # Display network statistics
            stats = self.network.get_network_stats()
            self._log_message(f"--- Network Stats ---")
            self._log_message(f"Edges: {stats['edges']:,}")
            self._log_message(f"Avg Degree: {stats['avg_degree']:.2f}")
            self._log_message(f"Clustering Coeff: {stats['clustering']:.3f}")
            self._log_message(f"---------------------")
            
            # Reset simulator after new network generation
            self.simulator = None
            self.update_statistics(reset=True)
            self.visualize_network()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate network: {str(e)}")
            self.network = None

    def start_simulation(self):
        """Initialize and start disease spread simulation."""
        if self.network is None:
            messagebox.showerror("Error", "Please generate network first")
            return
            
        if self.simulator and self.simulator.current_day > 0:
            # Already initialized, just pause/unpause
            self.toggle_simulation()
            return

        try:
            trans_prob = self.trans_var.get()
            recovery_time = self.recovery_var.get()
            initial_infected = self.infected_var.get()
            
            if initial_infected <= 0 or initial_infected >= self.network.n_nodes:
                messagebox.showerror("Error", "Initial infected must be > 0 and < population size.")
                return

            self.simulator = DiseaseSpreadSimulator(
                self.network,
                transmission_prob=trans_prob,
                recovery_time=recovery_time,
                initial_infected=initial_infected,
                interaction_model='degree_based'
            )

            self.update_statistics()
            self._log_message(f"Simulation ready: P_T={trans_prob}, Rec.Time={recovery_time} days. Starting spread...")
            self.visualize_network()
            self.running = True
            self.run_simulation_loop()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to start simulation: {str(e)}")
            
    def toggle_simulation(self):
        """Toggle the simulation running state."""
        if self.simulator is None:
            self.start_simulation()
            return
            
        self.running = not self.running
        if self.running:
            self._log_message("Simulation resumed.")
            self.run_simulation_loop()
        else:
            self._log_message("Simulation paused.")

    def step_simulation(self):
        """Perform one simulation step."""
        if self.simulator is None:
            messagebox.showerror("Error", "Please start simulation first")
            return

        if self.simulator.infected_count[-1] == 0 and self.simulator.current_day > 0:
            self._log_message("\nSimulation is already complete!")
            self.running = False
            return
            
        # Temporarily stop the loop if it's running so we only execute one step
        was_running = self.running
        self.running = False
        self.root.update()

        new_inf, new_rec = self.simulator.simulate_day()
        self.update_statistics()
        self.visualize_network()
        
        self._log_message(f"Day {self.simulator.current_day}: **{new_inf}** new infections, **{new_rec}** recoveries.")
        
        if was_running:
            self.running = True # If it was running, the loop will resume after this call returns

    def run_simulation_loop(self):
        """Run simulation continuously."""
        if not self.running:
            return

        if self.simulator.infected_count[-1] == 0 and self.simulator.current_day > 0:
            self.running = False
            self._log_message("\nSimulation complete! Infection has subsided.")
            return

        # Perform one step and schedule the next iteration
        self.step_simulation() 
        self.root.after(100, self.run_simulation_loop) # Faster updates

    def reset_simulation(self):
        """Reset simulation."""
        self.running = False
        self.simulator = None
        
        # Reset network status if it exists
        if self.network:
            for node in self.network.nodes:
                self.network.status[node] = 'S'
                self.network.infection_time[node] = -1
            self.visualize_network()
        
        # Reset GUI components
        self.update_statistics(reset=True)
        self.update_plot(reset=True)
        self.canvas.delete('all')
        self._log_message("\n--- Simulation Reset ---")


    # --- Visualization and Statistics Methods ---

    def visualize_network(self):
        """Draw network on canvas using force-directed layout."""
        if self.network is None:
            self.canvas.delete('all')
            return

        self.canvas.delete('all')

        # Transform coordinates to canvas space
        w, h = self.canvas.winfo_width(), self.canvas.winfo_height()
        if w < 100 or h < 100: # Use default if not fully rendered
            w, h = 900, 700 
        
        margin = 50

        positions = {}
        # Get the current bounds of the layout to ensure scaling is correct
        min_x = min(pos[0] for pos in self.network.positions.values()) if self.network.positions else -400
        max_x = max(pos[0] for pos in self.network.positions.values()) if self.network.positions else 400
        min_y = min(pos[1] for pos in self.network.positions.values()) if self.network.positions else -300
        max_y = max(pos[1] for pos in self.network.positions.values()) if self.network.positions else 300
        
        range_x = max(1.0, max_x - min_x)
        range_y = max(1.0, max_y - min_y)

        # Scale Factor to fit the graph within the canvas margins
        scale_x = (w - 2 * margin) / range_x
        scale_y = (h - 2 * margin) / range_y
        scale = min(scale_x, scale_y)
        
        center_x = (max_x + min_x) / 2
        center_y = (max_y + min_y) / 2
        
        canvas_center_x = w / 2
        canvas_center_y = h / 2

        for node in self.network.nodes:
            x, y = self.network.positions[node]

            # Translate, Scale, and Shift to Center
            canvas_x = (x - center_x) * scale + canvas_center_x
            canvas_y = (y - center_y) * scale + canvas_center_y
            positions[node] = (canvas_x, canvas_y)

        # Draw edges first (background)
        for u, v in self.network.edges:
            x1, y1 = positions[u]
            x2, y2 = positions[v]
            self.canvas.create_line(x1, y1, x2, y2, fill='#cccccc', width=1)

        # Draw nodes (foreground)
        for node in self.network.nodes:
            x, y = positions[node]

            # Color based on status
            color = self.COLOR_MAP.get(self.network.status[node], 'gray')

            # Size based on degree (high degree nodes are larger - 'superspreaders')
            degree = self.network.get_degree(node)
            size = min(12, 4 + degree / 2) # Cap max size for visual clarity

            self.canvas.create_oval(x-size, y-size, x+size, y+size,
                                    fill=color, outline='black', width=1, tags='node')

        # Legend (Clear and simple)
        legend_y = 20
        self.canvas.create_text(50, legend_y, text="Status Legend:", font=('Arial', 10, 'bold'), anchor='w')
        
        # S - Susceptible
        self.canvas.create_oval(150, legend_y-5, 160, legend_y+5, fill=self.COLOR_MAP['S'], outline='black')
        self.canvas.create_text(165, legend_y, text="Susceptible", anchor='w')
        
        # I - Infected
        self.canvas.create_oval(250, legend_y-5, 260, legend_y+5, fill=self.COLOR_MAP['I'], outline='black')
        self.canvas.create_text(265, legend_y, text="Infected", anchor='w')
        
        # R - Recovered
        self.canvas.create_oval(340, legend_y-5, 350, legend_y+5, fill=self.COLOR_MAP['R'], outline='black')
        self.canvas.create_text(355, legend_y, text="Recovered/Immune", anchor='w')
        
    def update_statistics(self, reset=False):
        """Update statistics display."""
        if reset or self.simulator is None:
            stats = self.simulator.get_statistics() if self.simulator else {
                'day': 0, 'susceptible': 0, 'infected': 0, 'recovered': 0, 
                'total_infected': 0, 'new_infections': 0, 'attack_rate': 0.0
            }
            if self.network:
                stats['susceptible'] = self.network.n_nodes
                
            # Use '0' or 'N/A' for reset
            s_text = f"{stats['susceptible']:,}" if self.network else "N/A"
            i_text = "0"
            r_text = "0"
            
        else:
            stats = self.simulator.get_statistics()
            self.update_plot()
            s_text = f"{stats['susceptible']:,}"
            i_text = f"{stats['infected']:,}"
            r_text = f"{stats['recovered']:,}"
        
        # Update labels
        self.stats_labels['day'].configure(text=str(stats['day']))
        self.stats_labels['susceptible'].configure(text=s_text)
        self.stats_labels['infected'].configure(text=i_text)
        self.stats_labels['recovered'].configure(text=r_text)
        self.stats_labels['total_infected'].configure(text=f"{stats['total_infected']:,}")
        self.stats_labels['attack_rate'].configure(text=f"{stats['attack_rate']*100:.1f}%")
        
        # Highlight infected count
        if stats['infected'] > 0:
             self.stats_labels['infected'].configure(foreground=self.COLOR_MAP['I'])
        else:
             self.stats_labels['infected'].configure(foreground='black')


    def update_plot(self, reset=False):
        """Update the SIR curve plot."""
        self.plot.clear()
        
        if reset or self.simulator is None or self.network is None:
            self.plot.set_title("SIR Progression (Day 0)", fontsize=10)
            self.plot.set_xlabel("Time (Days)", fontsize=8)
            self.plot.set_ylabel("Population Fraction", fontsize=8)
            self.plot.plot([0, 1], [0, 0], color='gray', linestyle='--') 
        else:
            days = list(range(self.simulator.current_day + 1))
            N = self.network.n_nodes
            
            # Convert counts to fractions for generalization
            S_frac = np.array(self.simulator.susceptible_count) / N
            I_frac = np.array(self.simulator.infected_count) / N
            R_frac = np.array(self.simulator.recovered_count) / N
            
            self.plot.plot(days, S_frac, label='Susceptible (S)', color=self.COLOR_MAP['S'], linewidth=2)
            self.plot.plot(days, I_frac, label='Infected (I)', color=self.COLOR_MAP['I'], linewidth=2)
            self.plot.plot(days, R_frac, label='Recovered (R)', color=self.COLOR_MAP['R'], linewidth=2)
            
            self.plot.set_title(f"SIR Progression (Day {self.simulator.current_day})", fontsize=10)
            self.plot.set_xlabel("Time (Days)", fontsize=8)
            self.plot.set_ylabel("Population Fraction", fontsize=8)
            self.plot.tick_params(axis='both', which='major', labelsize=8)
            self.plot.legend(loc='center right', fontsize=8)
            self.plot.grid(True, linestyle='--', alpha=0.6)
            self.plot.set_ylim(0, 1)

        self.fig.tight_layout()
        self.plot_canvas.draw()
        
    def _log_message(self, message):
        """Internal helper to add messages to the log display."""
        self.log_text.configure(state=tk.NORMAL)
        # Simple markdown-like rendering for bold text
        rendered_message = message.replace('**', '') 
        self.log_text.insert(tk.END, f"{datetime.now().strftime('%H:%M:%S')} - {rendered_message}\n")
        self.log_text.see(tk.END) # Auto-scroll to bottom
        self.log_text.configure(state=tk.DISABLED)


def main():
    root = tk.Tk()
    # Use a visually appealing color palette
    root.tk_setPalette(background='#f0f0f0', foreground='#333333', 
                       activeBackground='#cccccc', activeForeground='#000000')
    app = SimulatorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()