import numpy as np
from scipy.integrate import solve_ivp


class Parameters:
    """
    Class to store parameters of an ODE model. 
    
    Each parameter is taken from  a dictionary and stored as an attribute of the class.
    """

    def __init__(self, p_dict):
        """
        Parameters
        ----------
        p_dict : dict
            Dictionary with parameter names as keys and their values as
            dict-values. For example: p_dict = {'p_a': 1.0, 'p_b': 2.0}.
        """

        for p_name, p_val in p_dict.items():
            setattr(self, p_name, p_val)


def Hi(x, K, n):
    """
    Increasing Hill function.

    Parameters
    ----------
    x : array_like
        Input.
    K : float
        Half-maximal value.
    n : float
        Hill exponent.

    Returns
    -------
    y : array_like
    """
    a = np.power(abs(x),n)
    b = np.power(K,n)
    y = np.divide(a,a+b)
    return y


def Hd(x, K, n):
    """
    Decreasing Hill function.

    Parameters
    ----------
    x : array_like
        Input.
    K : float
        Half-maximal value.
    n : float
        Hill exponent.

    Returns
    -------
    y : array_like
    """
    a = np.power(abs(x),n)
    b = np.power(K,n)
    y = np.divide(b,a+b)
    return y


class Guan2008Model:
    """
    Implementation of the ODE model from Guan et al. eLife 2008.
    """

    def __init__(self):
        self.introduction = """The cell cycle is a fundamental biological process that ensures accurate cell divisions. 
                            In a [2018 eLife article](https://elifesciences.org/articles/33549), Guan et al. led a study that established 
                            an experimental and computational platform to study the early embryonic cell cycles of *Xenopus laevis* frogs.
                            Here, we present the mathematical model developed in that article through an interactive application. 
                            This application allows the user to explore the model's behavior by changing parameters and initial conditions. 
                            The model is composed of two species, Total Cyclin B1 and Active CyclinB1:Cdk1 complex, which provides a simple 
                            representation of the cell cycle."""

        self.variables = ['B', 'C']
        self.initial_conditions = [75.0, 35.0]
        self.parameters = Parameters({
            'kS': 1.0, 
            # Ratio of Wee1 to Cdc25 activity
            'rWT': 1.5,
            # Degradation
            'aD': 0.01, 'bD': 0.04, 'KD': 32.0, 'nD': 17.0,
            # Cdc25
            'aT': 0.16, 'bT': 0.8, 'KT': 30.0, 'nT': 11.0,
            # Wee1
            'aW': 0.08, 'bW': 0.4, 'KW': 35.0, 'nW': 3.5,
        })

        self.slider_information = {
            'kS': {
                'initial_value': self.parameters.__getattribute__('kS'), 
                'description': 'Protein synthesis rate', 
                'min': 0.0, 
                'max': 2.0, 
                'step': 0.001, 
                'units': 'nM/min'
            },
            # Ratio of Wee1 to Cdc25 activity
            'rWT': {
                'initial_value': self.parameters.__getattribute__('rWT'),
                'description': 'Ratio of Wee1 to Cdc25 activity',
                'min': 0.00001,
                'max': 5.0,
                'step': 0.001,
                'units': 'a.u.'
            },
            # Degradation
            'aD': {
                'initial_value': self.parameters.__getattribute__('aD'),
                'description': 'Basal protein degradation rate',
                'min': 0.0,
                'max': 0.1,
                'step': 0.00001,
                'units': '1/min'
            },
            'bD': {
                'initial_value': self.parameters.__getattribute__('bD'),
                'description': 'CyclinB1:Cdk1-dependent protein degradation rate',
                'min': 0.000001,
                'max': 5.0,
                'step': 0.001,
                'units': '1/min'
            },
            'KD': {
                'initial_value': self.parameters.__getattribute__('KD'),
                'description': 'Half-maximal protein degradation rate',
                'min': 1.0,
                'max': 120.0,
                'step': 0.1,
                'units': 'nM'
            },
            'nD': {
                'initial_value': self.parameters.__getattribute__('nD'),
                'description': 'Hill exponent for protein degradation',
                'min': 1.0,
                'max': 25.0,
                'step': 0.1,
                'units': 'a.u.'
            },
            # Cdc25 activation
            'aT': {
                'initial_value': self.parameters.__getattribute__('aT'),
                'description': 'Basal Cdc25 activation rate',
                'min': 0.00001,
                'max': 10.0,
                'step': 0.001,
                'units': '1/min'
            },
            'bT': {
                'initial_value': self.parameters.__getattribute__('bT'),
                'description': 'CyclinB1:Cdk1-dependent Cdc25 activation rate',
                'min': 0.0001,
                'max': 10.0,
                'step': 0.001,
                'units': '1/min'
            },
            'KT': {
                'initial_value': self.parameters.__getattribute__('KT'),
                'description': 'Half-maximal Cdc25 activation rate',
                'min': 1.0,
                'max': 120.0,
                'step': 0.1,
                'units': 'nM'
            },
            'nT': {
                'initial_value': self.parameters.__getattribute__('nT'),
                'description': 'Hill exponent for Cdc25 activation',
                'min': 1.0,
                'max': 25.0,
                'step': 0.1,
                'units': 'a.u.'
            },
            # Wee1 inhibition
            'aW': {
                'initial_value': self.parameters.__getattribute__ ('aW'),
                'description': 'Basal Wee1 inhibition rate',
                'min': 0.0001,
                'max': 10.0,
                'step': 0.00001,
                'units': '1/min'
            },
            'bW': {
                'initial_value': self.parameters.__getattribute__('bW'),
                'description': 'CyclinB1:Cdk1-dependent Wee1 inhibition rate',
                'min': 0.0001,
                'max': 10.0,
                'step': 0.0001,
                'units': '1/min'
            },
            'KW': {
                'initial_value': self.parameters.__getattribute__('KW'),
                'description': 'Half-maximal Wee1 inhibition rate',
                'min': 1.0,
                'max': 120.0,
                'step': 0.1,
                'units': 'nM'
            },
            'nW': {
                
                'initial_value': self.parameters.__getattribute__('nW'),
                'description': 'Hill exponent for Wee1 inhibition',
                'min': 1.0,
                'max': 25.0,
                'step': 0.1,
                'units': 'a.u.'
            },
        }

    def set_parameters(self, p_dict):
        """
        Set the parameters of the model.

        Parameters
        ----------
        p_dict : dict
            Dictionary of parameters.
        """
        self.parameters = Parameters(p_dict)

    def set_initial_conditions(self, y0):
        """
        Set the initial conditions of the model.

        Parameters
        ----------
        y0 : array_like
            Initial conditions.
        """
        self.initial_conditions = y0


    def interaction_terms(self, C, p):
        """
        Calculate the Hill interaction terms.

        Parameters
        ----------
        C : array_like
            Concentration of CyclinB1:Cdk1 complex.
        p : Parameters
            Parameters of the model.

        Returns
        -------
        HD : array_like
            Protein degradation rate.
        HT : array_like
            Cdc25 activation rate.
        HW : array_like
            Wee1 inhibition rate.
        """
        HD = p.aD + p.bD * Hi(C, p.KD, p.nD)
        HT = (p.aT + p.bT * Hi(C, p.KT, p.nT)) / np.sqrt(p.rWT)
        HW = (p.aW + p.bW * Hd(C, p.KW, p.nW)) * np.sqrt(p.rWT)

        return HD, HT, HW

    def equations(self, t, y, p):
        """
        Implementation of the ODE model from Guan et al. eLife 2008.

        The model contains 2 differential equations describing the evolution of 
        Total Cyclin B1 (B) and CyclinB1:Cdk1 complex (C). Parameters are 
        
        Parameters
        ----------
        t : array_like
            Time points.
        y : array_like
            State variables.
        p : Parameters
            Parameters of the model.

        Returns
        -------
        dydt : array_like
        """
        B, C = y

        # Hill interaction terms
        HD, HT, HW = self.interaction_terms(C, p)

        # ODEs
        dBdt = p.kS - HD * B
        dCdt = p.kS + HT * (B - C) - HW * C - HD * C
        dydt = [dBdt, dCdt]

        return dydt
        
    
    def nullclines(self, C, p):
        """
        Compute the nullclines of the model.

        Parameters
        ----------
        C : array_like
            Array of C values.

        p : Parameters
            Parameters of the model.

        Returns
        -------
        B_nullcline : function
            Nullcline for B.
        C_nullcline : function
            Nullcline for C.
        """
        HD, HT, HW = self.interaction_terms(C, p)
        T = HT * np.sqrt(p.rWT) 
        W = HW / np.sqrt(p.rWT) 

        # Nullclines
        B_nullcline = p.kS / HD
        C_nullcline = C + np.sqrt(p.rWT) * (HD * C + np.sqrt(p.rWT) * W * C - p.kS) / T

        return B_nullcline, C_nullcline
        


    