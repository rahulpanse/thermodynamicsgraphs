import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter


# Energy levels in Joules (e.g., quantum system with discrete levels)
E = np.linspace(0, 1e-20, 10)  # 10 levels from 0 to 1e-20 J

# Boltzmann constant in J/K
k_B = 1.380649e-23

# Realistic temperatures in Kelvin
temperatures = [10, 100, 300, 1000]

plt.figure(figsize=(10, 6))

for T in temperatures:
    beta = 1 / (k_B * T)  # in 1/J

    # Gibbs–Boltzmann probabilities
    boltzmann_factors = np.exp(-beta * E)
    Z = np.sum(boltzmann_factors)
    P = boltzmann_factors / Z

    plt.plot(E, P, marker='o', label=f"T = {T} K")

# Plot details
plt.title("Gibbs–Boltzmann Distribution (SI Units, Realistic T)")
plt.xlabel("Energy Level $E_n$ (Joules)")
plt.ylabel("Probability $P(E_n)$")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
#####################################################################################################################################################################################
#CARNOT CYCLE
# Constants (assume 1 mole of ideal gas, R = 8.314 J/mol·K)
R = 8.314
n = 1.0  # moles

# Temperatures (in Kelvin)
T_H = 500  # High temperature
T_C = 300  # Low temperature

# Volume points (in m^3)
V1 = 1.0   # Initial volume
V2 = 2.0   # Volume after isothermal expansion
V3 = 3.0   # Volume after adiabatic expansion

# Use adiabatic relation: TV^(γ-1) = constant to compute V4
gamma = 5 / 3  # Monatomic gas
V4 = V1 * (T_H / T_C) ** (1 / (gamma - 1))  # Return to initial state

# Volume ranges for each process
V_isoth_exp = np.linspace(V1, V2, 100)
V_adia_exp = np.linspace(V2, V3, 100)
V_isoth_comp = np.linspace(V3, V4, 100)
V_adia_comp = np.linspace(V4, V1, 100)

# Isothermal expansion (1 -> 2)
P_isoth_exp = (n * R * T_H) / V_isoth_exp

# Adiabatic expansion (2 -> 3): P * V^γ = constant
P2 = (n * R * T_H) / V2
K_exp = P2 * V2**gamma
P_adia_exp = K_exp / V_adia_exp**gamma

# Isothermal compression (3 -> 4)
P_isoth_comp = (n * R * T_C) / V_isoth_comp

# Adiabatic compression (4 -> 1): P * V^γ = constant
P4 = (n * R * T_C) / V4
K_comp = P4 * V4**gamma
P_adia_comp = K_comp / V_adia_comp**gamma

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(V_isoth_exp, P_isoth_exp, label='Isothermal Expansion', color='red')
plt.plot(V_adia_exp, P_adia_exp, label='Adiabatic Expansion', color='orange')
plt.plot(V_isoth_comp, P_isoth_comp, label='Isothermal Compression', color='blue')
plt.plot(V_adia_comp, P_adia_comp, label='Adiabatic Compression', color='green')

# Annotate key points
plt.scatter([V1, V2, V3, V4],
            [(n * R * T_H) / V1, (n * R * T_H) / V2, (n * R * T_C) / V3, (n * R * T_C) / V4],
            color='black')
plt.text(V1, (n * R * T_H) / V1, '1', fontsize=12)
plt.text(V2, (n * R * T_H) / V2, '2', fontsize=12)
plt.text(V3, (n * R * T_C) / V3, '3', fontsize=12)
plt.text(V4, (n * R * T_C) / V4, '4', fontsize=12)

# Labels and grid
plt.title('Carnot Cycle (P–V Diagram)', fontsize=14)
plt.xlabel('Volume (m³)')
plt.ylabel('Pressure (Pa)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
########################################################################################################################################################################
#OTTO CYCLE
# Constants
R = 8.314       # J/(mol·K)
n = 1.0         # mol
gamma = 1.4     # for diatomic gas like air

# Compression ratio (V1/V2)
r = 8

# Volumes
V1 = 1.0
V2 = V1 / r
V3 = V2
V4 = V1

# Temperatures (arbitrary consistent units)
T1 = 300        # Initial temperature
T2 = T1 * (r**(gamma - 1))  # From adiabatic relation
T3 = 2000       # After isochoric heat addition
T4 = T3 * (r**(1 - gamma))  # From adiabatic relation

# Pressures using ideal gas law
P1 = (n * R * T1) / V1
P2 = (n * R * T2) / V2
P3 = (n * R * T3) / V3
P4 = (n * R * T4) / V4

# Adiabatic compression: 1 -> 2
V_comp = np.linspace(V1, V2, 100)
P_comp = P1 * (V1 / V_comp) ** gamma

# Isochoric heat addition: 2 -> 3
V_iso_add = np.array([V2, V3])
P_iso_add = np.array([P2, P3])

# Adiabatic expansion: 3 -> 4
V_exp = np.linspace(V3, V4, 100)
P_exp = P3 * (V3 / V_exp) ** gamma

# Isochoric heat rejection: 4 -> 1
V_iso_rej = np.array([V4, V1])
P_iso_rej = np.array([P4, P1])

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(V_comp, P_comp, 'red', label='Adiabatic Compression (1→2)')
plt.plot(V_iso_add, P_iso_add, 'orange', label='Isochoric Heat Addition (2→3)')
plt.plot(V_exp, P_exp, 'blue', label='Adiabatic Expansion (3→4)')
plt.plot(V_iso_rej, P_iso_rej, 'green', label='Isochoric Heat Rejection (4→1)')

# Annotate state points
states = [(V1, P1), (V2, P2), (V3, P3), (V4, P4)]
for i, (V, P) in enumerate(states, start=1):
    plt.plot(V, P, 'ko')
    plt.text(V, P + 10000, f'{i}', fontsize=12)

# Labels and legend
plt.title('Otto Cycle (P–V Diagram)', fontsize=14)
plt.xlabel('Volume (m³)')
plt.ylabel('Pressure (Pa)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
#####################################################################################################################################################################################
#THERMO-MAJORIZATION CURVE
def compute_thermo_majorization_curve(prob_dist, energy_levels, beta):
    # Compute Boltzmann weights
    boltzmann_weights = np.exp(-beta * np.array(energy_levels))

    # Compute the β-ordered indices (descending order of p_i / e^{-βE_i})
    beta_order = np.argsort(-(prob_dist / boltzmann_weights))

    # Order probabilities and Boltzmann weights accordingly
    p_sorted = np.array(prob_dist)[beta_order]
    w_sorted = boltzmann_weights[beta_order]

    # Compute cumulative sums for thermo-majorization curve
    cumulative_x = np.cumsum(w_sorted)
    cumulative_y = np.cumsum(p_sorted)

    # Add (0, 0) to the curve
    curve_x = np.insert(cumulative_x, 0, 0)
    curve_y = np.insert(cumulative_y, 0, 0)

    return curve_x, curve_y


def plot_thermo_majorization(state, diagonal_state, energy_levels, beta):
    """
    Plots thermo-majorization curves for:
    - The given state (assumed diagonal or made diagonal)
    - The diagonalized version (i.e., decohered state)

    Parameters:
    - state: list of probabilities (non-diagonal state can be approximated as a diagonal projection)
    - diagonal_state: explicitly diagonal state
    - energy_levels: energy levels corresponding to each state
    - beta: inverse temperature
    """

    # Compute both curves
    x1, y1 = compute_thermo_majorization_curve(state, energy_levels, beta)
    x2, y2 = compute_thermo_majorization_curve(diagonal_state, energy_levels, beta)

    # Plotting
    plt.figure(figsize=(8, 6))
    plt.plot(x1, y1, 'b-', label='General (possibly coherent) state', linewidth=2)
    plt.plot(x2, y2, 'r--', label='Diagonal state (decohered)', linewidth=2)
    plt.xlabel('Cumulative Boltzmann Weights')
    plt.ylabel('Cumulative Probabilities')
    plt.title('Thermo-Majorization Curves')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# Example use-case
if __name__ == "__main__":
    # Define energy levels (in arbitrary units)
    energy_levels = [0, 1, 2]

    # Inverse temperature β = 1 / (k_B * T)
    beta = 1.0

    # Example state: may include off-diagonal coherence (simplified here as diagonal projection)
    state = [0.5, 0.3, 0.2]

    # Diagonal state: typically the dephased version in energy eigenbasis
    diagonal_state = [0.6, 0.25, 0.15]  # Could come from tracing out coherence

    # Plot the thermo-majorization curves
    plot_thermo_majorization(state, diagonal_state, energy_levels, beta)
###############################################################################################################################################################################
#GIBBS-BOLTZMANN VS HELMHOLTZ
# Constants
k_B = 1.380649e-23  # J/K

# Energy levels
n_levels = 10000
E = np.sort(np.random.exponential(scale=1e-21, size=n_levels))  # J

# Fixed temperature for Gibbs distribution
T_fixed = 300  # K
beta_fixed = 1 / (k_B * T_fixed)
Z_fixed = np.sum(np.exp(-beta_fixed * E))
P_fixed = np.exp(-beta_fixed * E) / Z_fixed

# Temperature range for F(T) and entropy
temperatures = np.linspace(10, 1000, 400)
F_T = []
S_T = []

for T in temperatures:
    beta = 1 / (k_B * T)
    boltzmann_weights = np.exp(-beta * E)
    Z = np.sum(boltzmann_weights)
    P = boltzmann_weights / Z

    # Helmholtz free energy
    F = -k_B * T * np.log(Z)
    F_T.append(F)

    # Entropy of Gibbs distribution
    with np.errstate(divide='ignore'):
        logP = np.log(P, where=(P > 0))
    S = -k_B * np.sum(P * logP)
    S_T.append(S)

F_T = np.array(F_T)
S_T = np.array(S_T)

# Plotting
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'font.family': 'serif',
    'mathtext.fontset': 'cm',
    'figure.dpi': 300
})

fig, axs = plt.subplots(1, 2, figsize=(8.6, 3.3))

# 1. Gibbs Distribution
axs[0].plot(E, P_fixed, color='navy', lw=1.5)
axs[0].set_title(r"Gibbs Distribution at $T = 300$ K")
axs[0].set_xlabel(r"Energy Level $E_n$ (J)")
axs[0].set_ylabel(r"Probability $P(E_n)$")
axs[0].grid(True, linestyle=':', alpha=0.4)
axs[0].spines['top'].set_visible(False)
axs[0].spines['right'].set_visible(False)

# 2. Helmholtz Free Energy vs Gibbs Distribution (via Entropy)
axs[1].plot(S_T, F_T, color='darkgreen', lw=1.5)
axs[1].set_title(r"Helmholtz Free Energy vs Entropy")
axs[1].set_xlabel(r"Gibbs Entropy $S(T)$ (J/K)")
axs[1].set_ylabel(r"Free Energy $F(T)$ (J)")
axs[1].grid(True, linestyle=':', alpha=0.4)
axs[1].spines['top'].set_visible(False)
axs[1].spines['right'].set_visible(False)

# Format y-axis
formatter = ScalarFormatter(useMathText=True)
formatter.set_powerlimits((-2, 2))
axs[0].yaxis.set_major_formatter(formatter)
axs[1].yaxis.set_major_formatter(formatter)

plt.tight_layout()
plt.savefig("helmholtz_vs_gibbs_entropy.pdf")
plt.show()
####################################################################################################################################################################################
