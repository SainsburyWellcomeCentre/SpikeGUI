# -*- coding: utf-8 -*-
"""
Created on Fri Jul 18 11:12:20 2014

@author: crousse
"""
from datetime import date
from datetime import time
class Mouse(object):
    def __init__(self):
        """
        Dates are of type datetime.date
        """
        self.id = ""
        self.gender = ""
        self.genotype = ""
        self.birthDate = date(1900,  1,  1)
        self.weight = 0
        self.craniotomy = (0, 0)
        self.surgeryDate = date(1900,  1,  1)
        self.surgeryStart = time(0,  0,  0)
        self.surgeryEnd = time(0,  0,  0)
        self.anaestheticAmount = 0
        self.injectionDate = date(1900,  1,  1)
        self.perfDate = date(1900,  1,  1)
        self.results = "" # TODO: check that appropriate
       
    @property   
    def isTransgenic(self):
        return not(self.genotype.lower() == 'wt')
            
    def age(self, compareDate):
        """
        Returns the age in days
        """
        if compareDate == date(1900,  1,  1):
            raise ValueError("The compare date was not set properly: {}".format(compareDate))
        delta = compareDate - self.birthDate
        return delta.days
        
    def setGender(self, gender):
        genders = ['male', 'female']
        if gender.lower() in genders:
            self.gender = gender.lower()
        else:
            raise KeyError("Gender {} is not supported, supported genders are: {}.".format(gender,  genders))
