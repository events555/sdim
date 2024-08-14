import pytest
import numpy as np
from sdim.tableau import Tableau
from sympy import *

class TestTableau:
    @pytest.fixture
    def tableau_3_qudits_even_composite(self):
        """
        Fixture for creating a Tableau with 3 qudits.
        """
        return Tableau(num_qudits=3, dimension=4)
    @pytest.fixture
    def tableau_3_qudits_odd_composite(self):
        """
        Fixture for creating a Tableau with 3 qudits.
        """
        return Tableau(num_qudits=3, dimension=9)
    @pytest.fixture
    def tableau_3_qudits_even_prime(self):
        """
        Fixture for creating a Tableau with 3 qudits.
        """
        return Tableau(num_qudits=3, dimension=2)
    @pytest.fixture
    def tableau_3_qudits_odd_prime(self):
        """
        Fixture for creating a Tableau with 3 qudits.
        """
        return Tableau(num_qudits=3, dimension=3)

    def test_initialization(self, tableau_3_qudits_even_composite):
        """
        Test that the Tableau initializes correctly.
        """
        assert len(tableau_3_qudits_even_composite.generator) == 6  # 2*num_qudits generators

 