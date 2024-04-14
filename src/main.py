import cProfile
import pstats
import matplotlib
import matplotlib.pyplot as plt
import json
import os
import numpy as np
from sdim.chp_parser import read_circuit
from sdim.program import Program
from sdim.random_circuit import generate_chp_file, circuit_to_cirq_circuit
from sdim.circuit import Circuit
import cirq
import shutil

def simulate_cirq(circuit):
    cirq_circuit = circuit_to_cirq_circuit(circuit, measurement=True)
    simulator = cirq.Simulator()
    result = simulator.simulate(cirq_circuit)
    return result

def simulate_tableau(circuit):
    program = Program(circuit)
    result = program.simulate(show_measurement=False, verbose=False, show_gate=False)
    return result

def plot_from_json(data_file, output_plot):
    shutil.rmtree(matplotlib.get_cachedir())

    with open(data_file, 'r') as f:
        data = json.load(f)

    dimensions = data['dimensions']
    max_qudits_tableau = max(max(num_qudits) for num_qudits in data['num_qudits_tableau'])  # Max for Tableau

    # Create figure with square aspect ratio and larger size
    fig, ax = plt.subplots(figsize=(12, 10))
    plt.rcParams['font.family'] = 'Verdana'

    # Define colors and markers for dot plots
    colors = ['orange', 'blue', 'green', 'red']
    markers = ['o', '^', 'd', 's']  # Unfilled circle, x, plus, and square

    # Line styles and width for averages
    linestyles = ['--', '-']
    linewidth = 4  # Increase line width for better visibility


    num_qudits_range = range(1, max_qudits_tableau + 1)
    avg_tableau_times = []
    for n in num_qudits_range:
        tableau_times_for_n = [data['tableau_times'][i][j] 
                               for i, dim in enumerate(dimensions) 
                               for j, num_qudits in enumerate(data['num_qudits_tableau'][i]) 
                               if num_qudits == n]
        avg_tableau_times.append(np.mean(tableau_times_for_n))
    for i, dimension in enumerate(dimensions):
        # Plot Cirq with thicker lines
        ax.plot(data['num_qudits_cirq'][i], data['cirq_times'][i], 
                linestyle=linestyles[0], color=colors[i], label=f'Cirq (d={dimension})', linewidth=linewidth)

        # Plot Tableau with markers and lines
        ax.plot(data['num_qudits_tableau'][i], data['tableau_times'][i], 
                linestyle='None', marker=markers[i], color=colors[i], label=f'Tableau (d={dimension})')
    ax.plot(num_qudits_range, avg_tableau_times, 
            linestyle=linestyles[1], color='black', linewidth=linewidth, label='Average Tableau', alpha=0.4)
    # Increase font sizes
    ax.set_xlabel('Number of Qudits (n)', fontsize=24)
    ax.set_ylabel('Runtime (seconds)', fontsize=24)
    ax.set_title('Time to Measurement of Randomized Circuit', fontsize=32)

    ax.set_yscale('log')
    ax.set_xticks(range(1, max_qudits_tableau + 1))
    ax.tick_params(axis='both', which='major', labelsize=24)
    ax.legend(fontsize=14)  # Increase legend font size
    ax.grid(True, which='major', axis='both', color='gray', linestyle='-', alpha=0.4, linewidth=0.5)

    plt.tight_layout()
    plt.savefig(output_plot, dpi=300)
    plt.show()

def main():
    dimensions = [2, 3, 5, 7]  # Dimensions to test
    max_qudits = [14, 9, 6, 5]  # Maximum qudits for each dimension
    gate_count = 100000
    measurement_rounds = 1
    c_percentage = 40
    h_percentage = 30
    p_percentage = 30
    m_percentage = 0

    data = {
        'dimensions': dimensions,
        'num_qudits_cirq': [],  # Cirq qudit counts
        'num_qudits_tableau': [],  # Tableau qudit counts
        'gate_counts': gate_count,
        'cirq_times': [],
        'tableau_times': [],
        'gate_distribution': {
            'CNOT': c_percentage,
            'Hadamard': h_percentage,
            'Phase': p_percentage,
            'Measurement': m_percentage
        }
    }

    max_qudit_overall = max(max_qudits)  # Overall maximum for Tableau
    for i, dimension in enumerate(dimensions):
        num_qudits_cirq = list(range(2, max_qudits[i] + 1))  # Cirq range
        data['num_qudits_cirq'].append(num_qudits_cirq)

        num_qudits_tableau = list(range(2, max_qudit_overall + 1))  # Tableau range
        data['num_qudits_tableau'].append(num_qudits_tableau)

        cirq_dim_times = []
        tableau_dim_times = []
        for num_qudit in num_qudits_cirq:  # Cirq simulation loop
            generate_chp_file(c_percentage=c_percentage, h_percentage=h_percentage, p_percentage=p_percentage, m_percentage=m_percentage,
                              num_qudits=num_qudit, num_gates=gate_count, dimension=dimension,
                              measurement_rounds=measurement_rounds, output_file="random_circuit.chp")

            # Read the generated circuit file
            circuit = read_circuit("circuits/random_circuit.chp")

            # Profile Cirq simulation
            print(f"Running Cirq simulation for dimension={dimension} and num_qudit={num_qudit}")
            profiler = cProfile.Profile()
            profiler.enable()
            simulate_cirq(circuit)
            profiler.disable()
            stats = pstats.Stats(profiler).strip_dirs()
            cirq_time = stats.total_tt
            print("Completed Cirq simulation")
        for num_qudit in num_qudits_tableau:  # Tableau simulation loop
            # Profile tableau simulation
            generate_chp_file(c_percentage=c_percentage, h_percentage=h_percentage, p_percentage=p_percentage, m_percentage=m_percentage,
                              num_qudits=num_qudit, num_gates=gate_count, dimension=dimension,
                              measurement_rounds=measurement_rounds, output_file="random_circuit.chp")
            # Read the generated circuit file
            circuit = read_circuit("circuits/random_circuit.chp")
            print(f"Running tableau simulation for dimension={dimension} and num_qudit={num_qudit}")
            profiler = cProfile.Profile()
            profiler.enable()
            simulate_tableau(circuit)
            profiler.disable()
            stats = pstats.Stats(profiler).strip_dirs()
            tableau_time = stats.total_tt
            print("Completed tableau simulation")

            cirq_dim_times.append(cirq_time)
            tableau_dim_times.append(tableau_time)

        data['cirq_times'].append(cirq_dim_times)
        data['tableau_times'].append(tableau_dim_times)

    # Save the data to a JSON file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(current_dir, '..', 'tests', 'testdata')
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'simulation_data.json')
    with open(output_file, 'w') as file:
        json.dump(data, file)

def rerun_tableau_simulations(data_file, output_file):
    with open(data_file, 'r') as f:
        data = json.load(f)

    max_qudits = [14, 9, 6, 5]  # Maximum qudits for each dimension
    max_qudit_overall = max(max_qudits)  # Get the overall maximum

    for i, dimension in enumerate(data['dimensions']):
        tableau_dim_times = []
        for num_qudit in range(2, max_qudit_overall + 1):  # Use overall max
            # Generate circuit (assuming generate_chp_file handles num_qudits correctly)
            generate_chp_file(
                c_percentage=data['gate_distribution']['CNOT'],
                h_percentage=data['gate_distribution']['Hadamard'],
                p_percentage=data['gate_distribution']['Phase'],
                m_percentage=data['gate_distribution']['Measurement'],
                num_qudits=num_qudit, 
                dimension=dimension,
                num_gates=data['gate_counts'],  # Assuming you have this in your data
                measurement_rounds=1,  # Or get this from data if it's variable
                output_file="random_circuit.chp"
            )
            circuit = read_circuit("circuits/random_circuit.chp")

            # Profile tableau simulation
            print(f"Running tableau simulation for dimension={dimension} and num_qudit={num_qudit}")
            profiler = cProfile.Profile()
            profiler.enable()
            simulate_tableau(circuit)
            profiler.disable()
            stats = pstats.Stats(profiler).strip_dirs()
            tableau_time = stats.total_tt
            print("Completed tableau simulation")

            tableau_dim_times.append(tableau_time)

        data['tableau_times'][i] = tableau_dim_times  # Update Tableau times in data

    # Save the updated data
    with open(output_file, 'w') as f:
        json.dump(data, f)

def test():
    circuit = Circuit(2, 3)
    circuit.add_gate("H", 0)
    circuit.add_gate("SUM", 0, 1)
    circuit.add_gate("M", 0)
    circuit.add_gate("M", 1)
    program = Program(circuit)
    program.simulate(show_measurement=True, verbose=True, show_gate=True)

if __name__ == "__main__":
    # # Save the data to a JSON file
    # current_dir = os.path.dirname(os.path.abspath(__file__))
    # output_dir = os.path.join(current_dir, '..', 'tests', 'testdata')
    # data_file = os.path.join(output_dir, 'simulation_data.json')
    # updated_file = os.path.join(output_dir, 'simulation_data.json')
    # # rerun_tableau_simulations(data_file, updated_file)
    # output_graph = os.path.join(output_dir, 'simulation_graph.png')
    # plot_from_json(updated_file, output_graph)
    test()