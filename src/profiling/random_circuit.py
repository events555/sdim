import os
import random

def generate_chp_file(c_percentage, h_percentage, p_percentage, m_percentage, num_qudits, num_gates, dimension, measurement_rounds = 0, output_file="random_circuit.chp", seed=None):
    # Check that the percentages sum to 100
    if c_percentage + h_percentage + p_percentage + m_percentage != 100:
        raise ValueError("The percentages do not sum to 100")

    # Set the seed for random sampling
    if seed is not None:
        random.seed(seed)

    # Calculate the number of each gate
    num_c = int((c_percentage / 100) * num_gates)
    num_h = int((h_percentage / 100) * num_gates)
    num_p = int((p_percentage / 100) * num_gates)
    num_m = int((m_percentage / 100) * num_gates)

    # Generate the gates
    gates = ['c'] * num_c + ['h'] * num_h + ['p'] * num_p + ['m'] * num_m
    random.shuffle(gates)

    # Generate the .chp file content
    chp_content = "Randomly-generated Clifford group quantum circuit\n"
    chp_content += f"# \nd {dimension}\n"
    for gate in gates:
        # qudits = random.sample(range(num_qudits), 1 if gate != 'c' else 2)
        if gate == 'c':
            start_qudit = random.randint(0, num_qudits - 2)  # -2 to ensure we have room for a consecutive qudit
            qudits = [start_qudit, start_qudit + 1]
        else:
            qudits = random.sample(range(num_qudits), 1)
        chp_content += f"{gate} {' '.join(map(str, qudits))}\n"

    # Append measurements across every qubit based on the number of measurement rounds
    for _ in range(measurement_rounds):
        for qubit in range(num_qudits):
            chp_content += f"m {qubit}\n"

    # Define the directory where the file will be saved
    directory = "src/profiling/"

    # Join the directory with the output file name
    output_path = os.path.join(directory, output_file)

    # Write the content to the .chp file
    with open(output_path, "w") as file:
        file.write(chp_content)


# Call the function to generate the .chp file
generate_chp_file(30, 30, 40, 0, 3, 150, 2, 1)