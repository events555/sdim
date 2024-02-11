import re


class PauliString:
    def __init__(self, num_qudits, pauli_string=None, dimension=2):
        """
        Initializes a PauliString object.

        Args:
            num_qudits (int): The number of qudits in the Pauli string. Used to initialize xpow and zpow to all 0, representing the identity product.
            string (str, optional): The string representation of the Pauli string. Defaults to None.
            dimension (int, optional): The dimension of the qudits. Defaults to 2.
        """
        self.num_qudits = num_qudits
        self.dimension = dimension
        self.xpow = [0 for _ in range(num_qudits)]
        self.zpow = [0 for _ in range(num_qudits)]
        self.phase = 0
        if pauli_string is not None:
            self.from_str(pauli_string)

    def __setitem__(self, index: int, new_pauli: object) -> None:
        if index < 0 or index >= self.num_qudits:
            raise IndexError("Index out of range.")
        if isinstance(new_pauli, str):
            match = re.match(r"([A-Z][\d!]*)", new_pauli)
            if match:
                pauli_term = match.group()
                pauli_char = pauli_term[0]
                number = re.search(r"\d+", pauli_term)
                if number:
                    number = int(number.group())
                if pauli_char == "X":
                    if number == "!":
                        self.xpow[index] = self.dimension - 1
                    else:
                        self.xpow[index] = int(number) if number else 1
                    self.zpow[index] = 0
                elif pauli_char == "Z":
                    if number == "!":
                        self.zpow[index] = self.dimension - 1
                    else:
                        self.zpow[index] = int(number) if number else 1
                    self.xpow[index] = 0
                elif pauli_char == "I":
                    self.xpow[index] = 0
                    self.zpow[index] = 0
                else:
                    raise ValueError("Error with regex finding Pauli term.")
            else:
                raise ValueError("Invalid Pauli operator.")
        else:
            raise ValueError("Invalid object types given.")

    def __str__(self):
        pauli_string = ""
        if self.phase != 0:
            pauli_string += f"w{self.phase}"
        for i in range(self.num_qudits):
            term = ""
            if self.xpow[i] != 0:
                term += (
                    f"X{self.xpow[i]}" if self.xpow[i] != self.dimension - 1 else "X!"
                )
            if self.zpow[i] != 0:
                term += (
                    f"Z{self.zpow[i]}" if self.zpow[i] != self.dimension - 1 else "Z!"
                )
            if term == "":
                term = "I"
            pauli_string += f"({term})"
        return pauli_string

    def from_str(self, pauli_string: str):
        """
        Converts a string representation of a Pauli string to its xpow, zpow, and phase representation.
        Expects every qudit to be explicitly represented in the string within it's own parentheses.

        Examples:
        "(X2)(X3Z4)" -> ([2, 3], [0, 4], 0) for xpow, zpow, and phase, respectively.
        Similarly,
        "w2(X)(I)(XZ)" -> ([1, 0, 1], [0, 0, 1], 2) with w being the primitive root of unity.
        Finally,
        "(X!)" -> ([d-1], [0], 1) for d being the dimension of the qudits (2 for qubits, 3 for qutrits, etc.)
        Args:
            string (str): The string representation of the Pauli string.
        """

        # Check if the Pauli string has a phase
        if pauli_string[0] == "w":
            self.phase = int(pauli_string[1])
            pauli_string = pauli_string[2:]
        # Get list of tuples containing every Pauli term
        pauli_terms = get_list_paulis(pauli_string, self.dimension)
        for i, (x, z) in enumerate(pauli_terms):
            # Check if X has value
            # Tuple will be ('X', 'pow')
            if x is not None:
                if "!" in x[1]:
                    self.xpow[i] = self.dimension - 1
                else:
                    self.xpow[i] = int(x[1])
            else:
                self.xpow[i] = 0
            # Check if Z has value
            if z is not None:
                if "!" in z[1]:
                    self.zpow[i] = self.dimension - 1
                else:
                    self.zpow[i] = int(z[1])
            else:
                self.zpow[i] = 0
        return


def get_list_paulis(pauli_string: str, dimension: int):
    """
    Returns a list of Pauli terms in the Pauli string.
    It first separates it by parentheses and then creates tuples in (X,Z) ordering.
    It requires dimension so that the inverse contains the correct number.
    Examples:
    "(X2)(X3Z4)" -> [(('X', '2'), None), (('X', '3'), ('Z', '4'))]
    "(X!)(Z)" -> [(('X', 'd-1'), None), (None, ('Z', 1))]
    """
    pauli_terms = []
    substring_list = re.findall(r"\((.*?)\)", pauli_string)  # Separate parentheses
    for s in substring_list:
        match = re.match(
            r"([A-Z][\d!]*)((?:[A-Z][\d!]*)?)", s
        )  # Split X#/Z# terms inside parentheses where # are numbers or !
        groups = match.groups()
        x_term = None
        z_term = None
        for group in groups:
            if group:
                if group[0] == "X":
                    x_term = (
                        (group[0], str(dimension - 1))
                        if "!" in group
                        else (group[0], group[1:] if len(group) > 1 else "1")
                    )
                elif group[0] == "Z":
                    z_term = (
                        (group[0], str(dimension - 1))
                        if "!" in group
                        else (group[0], group[1:] if len(group) > 1 else "1")
                    )
        pauli_terms.append((x_term, z_term))
    return pauli_terms
