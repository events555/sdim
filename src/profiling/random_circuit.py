import random
import os

def generate_chp_file(c_percentage, h_percentage, p_percentage, m_percentage, num_qudits, num_gates, dimension):
    # Check that the percentages sum to 100
    if c_percentage + h_percentage + p_percentage + m_percentage != 100:
        raise ValueError("The percentages do not sum to 100")

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
        qudits = random.sample(range(num_qudits), 1 if gate != 'c' else 2)
        chp_content += f"{gate} {' '.join(map(str, qudits))}\n"

    # Write the content to the .chp file
    with open(os.path.join("src", "profiling", "random_circuit.chp"), "w") as file:
        file.write(chp_content)

# Call the function to generate the .chp file
generate_chp_file(35, 30, 25, 10, 1000, 10000, 3)