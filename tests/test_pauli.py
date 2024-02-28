from paulistring import PauliString, get_list_paulis
import pytest

class PauliTest:
    dimension = [2, 3, 4]
    num_qudits = [1, 3, 6]
    @pytest.fixture
    def pauli_string(self):
        for d in self.dimension:
            for n in self.num_qudits:
                yield PauliString(n, None, d)

    def test_identity(self, pauli_string):
        """
        Test the identity of the PauliString class
        """
        all_zeros = [0 for _ in range(len(pauli_string.xpow))]
        assert pauli_string.xpow == all_zeros and pauli_string.zpow == all_zeros and pauli_string.phase == 0
    
    def test_setitem(self, pauli_string):
        """
        Test the setitem method of the PauliString class
        """
        for i in range(len(pauli_string.xpow)):
            # Test setting a value within the dimension
            pauli_string[i] = "X1"
            assert pauli_string.xpow[i] == 1 and pauli_string.zpow[i] == 0
            pauli_string[i] = "Z2"
            assert pauli_string.xpow[i] == 0 and pauli_string.zpow[i] == 2
            pauli_string[i] = "I"
            assert pauli_string.xpow[i] == 0 and pauli_string.zpow[i] == 0

            # Test setting a value beyond the dimension
            with pytest.raises(ValueError):
                pauli_string[i] = "X{}".format(pauli_string.dimension + 1)

            # Test setting the inverse
            pauli_string[i] = "X!"
            assert pauli_string.xpow[i] == pauli_string.dimension - 1 and pauli_string.zpow[i] == 0
            pauli_string[i] = "Z!"
            assert pauli_string.xpow[i] == 0 and pauli_string.zpow[i] == pauli_string.dimension - 1

            # Test setting without including the numbers
            pauli_string[i] = "X"
            assert pauli_string.xpow[i] == 1 and pauli_string.zpow[i] == 0
            pauli_string[i] = "Z"
            assert pauli_string.xpow[i] == 0 and pauli_string.zpow[i] == 1

            # Test setting with negative numbers
            with pytest.raises(ValueError):
                pauli_string[i] = "X-1"
            with pytest.raises(ValueError):
                pauli_string[i] = "Z-2"
                
    def test_from_str(pauli_string):
        """
        Test the from_str method of the PauliString class
        """
        # Test with a Pauli string containing no phase
        pauli_string.from_str("(X2)(X3Z4)")
        assert pauli_string.xpow == [2, 3] and pauli_string.zpow == [0, 4] and pauli_string.phase == 0

        # Test with a Pauli string containing a positive phase
        pauli_string.from_str("w2(X)(I)(XZ)")
        assert pauli_string.xpow == [1, 0, 1] and pauli_string.zpow == [0, 0, 1] and pauli_string.phase == 2

        # Test with a Pauli string containing a negative phase in even dimensions
        if pauli_string.dimension % 2 == 0:
            pauli_string.from_str("-w2(X)(I)(XZ)")
            assert pauli_string.xpow == [1, 0, 1] and pauli_string.zpow == [0, 0, 1] and pauli_string.phase == pauli_string.dimension - 2

        # Test with a Pauli string containing inverse X and Z terms
        pauli_string.from_str("(X!)(Z!)")
        assert pauli_string.xpow == [pauli_string.dimension - 1, 0] and pauli_string.zpow == [0, pauli_string.dimension - 1] and pauli_string.phase == 0

        # Test with a Pauli string containing a Y term in dimension 2
        if pauli_string.dimension == 2:
            pauli_string.from_str("(Y)")
            assert pauli_string.xpow == [1] and pauli_string.zpow == [1] and pauli_string.phase == 1

        # Test with a Pauli string containing X and Z terms without exponent numbers
        pauli_string.from_str("(X)(Z)")
        assert pauli_string.xpow == [1, 0] and pauli_string.zpow == [0, 1] and pauli_string.phase == 0


    def test_get_list_paulis():
        # Test with a Pauli string containing X and Z terms
        pauli_string = "(X2)(X3Z4)"
        dimension = 5
        result = get_list_paulis(pauli_string, dimension)
        assert result == [(('X', '2'), None), (('X', '3'), ('Z', '4'))]

        # Test with a Pauli string containing inverse X and Z terms
        pauli_string = "(X!)(Z)"
        dimension = 5
        result = get_list_paulis(pauli_string, dimension)
        assert result == [(('X', '4'), None), (None, ('Z', '1'))]

        # Test with a Pauli string containing only X terms
        pauli_string = "(X2)(X3)"
        dimension = 5
        result = get_list_paulis(pauli_string, dimension)
        assert result == [(('X', '2'), None), (('X', '3'), None)]

        # Test with a Pauli string containing only Z terms
        pauli_string = "(Z2)(Z3)"
        dimension = 5
        result = get_list_paulis(pauli_string, dimension)
        assert result == [(None, ('Z', '2')), (None, ('Z', '3'))]