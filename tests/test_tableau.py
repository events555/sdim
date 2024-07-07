import pytest
from sdim.paulistring import PauliString
from sdim.tableau import Tableau
from sympy import *

class TestTableau:
    @pytest.fixture
    def tableau_3_qudits(self):
        """
        Fixture for creating a Tableau with 3 qudits.
        """
        return Tableau(num_qudits=3, dimension=4)

    def test_initialization(self, tableau_3_qudits):
        """
        Test that the Tableau initializes correctly.
        """
        assert len(tableau_3_qudits.generator) == 6  # 2*num_qudits generators

    def test_find_coprime(self, tableau_3_qudits):
        """
        Test that the coprime set is computed correctly.
        """
        assert tableau_3_qudits.coprime == {1, 3, 5, 7}
        tableau_3_qudits.generator[0].xpow = [2, 0, 0]
        tableau_3_qudits.generator[0].zpow = [0, 2, 0]
        tableau_3_qudits.generator[1].xpow = [0, 2, 0]
        tableau_3_qudits.generator[1].zpow = [2, 0, 2]
        tableau_3_qudits.generator[2].xpow = [0, 0, 2]
        tableau_3_qudits.generator[2].zpow = [3, 2, 0]
        assert tableau_3_qudits.find_pivot(0, 0, 6) == 2

    def test_row_echelon_form_simple(self, tableau_3_qudits):
        """
        Test the row echelon form on a simple predefined tableau.
        """
        # Manually set a tableau to known values
        tableau_3_qudits.generator[0].xpow = [1, 0, 0]
        tableau_3_qudits.generator[0].zpow = [0, 1, 0]
        tableau_3_qudits.generator[1].xpow = [0, 1, 0]
        tableau_3_qudits.generator[1].zpow = [1, 0, 1]
        tableau_3_qudits.generator[2].xpow = [0, 0, 1]
        tableau_3_qudits.generator[2].zpow = [1, 1, 0]

        tableau_3_qudits.row_echelon_form()        
        
        # Check the resulting generators
        assert tableau_3_qudits.generator[0].xpow == [1, 0, 0]
        assert tableau_3_qudits.generator[0].zpow == [0, 0, 0]
        assert tableau_3_qudits.generator[1].xpow == [0, 1, 0]
        assert tableau_3_qudits.generator[1].zpow == [0, 0, 0]
        assert tableau_3_qudits.generator[2].xpow == [0, 0, 1]
        assert tableau_3_qudits.generator[2].zpow == [0, 0, 0]

    def test_phase_correction_update(self, tableau_3_qudits):
        """
        Test that the phase correction matrix is updated correctly.
        """
        # Manually set phase corrections
        tableau_3_qudits.phase_correction = [
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0]
        ]

        tableau_3_qudits.row_echelon_form()

        expected_phase_correction = [
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0]
        ]
        assert tableau_3_qudits.phase_correction == expected_phase_correction