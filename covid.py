import os
import simpy
import numpy as np
from numpy import random
from scipy.interpolate import interp1d
import pandas as pd
import matplotlib.pyplot as plt
from enum import Enum
import random as rd

# parameters
N_POPULATION = 488000 # total persons in
SIM_TIME = 180 # in days
MASK =0.4
INITIALLY_INFECTED=12
MASK_TRANSMISSION_REDUCTION = 0.5
MASK_INFECTION_REDUCTION = 0.3
QUARANTINE_EFFECTIVENESS = 0.8
SYMPTOMS_DELAY_SHAPE = 4.0
SYMPTOMS_DELAY_SCALE = 6.0
CONTAGION_DURATION_SHAPE = 2.0
CONTAGION_DURATION_SCALE = 4.0
SEVERE_SCALE = 12.0
SEVERE_SHAPE = 2.0
STREET_EXPOSE_INTERVAL=0.5
CHANCE_OF_DIAGNOSIS_IF_MODERATE = 0.5
DIAGNOSIS_DELAY=2

#age
age_str = [
    '0-9',
    '10-19',
    '20-29',
    '30-39',
    '40-49',
    '50-59',
    '60-69',
    '70-79',
    '80+',
]
age_dist = [
    0.1463559, 
    0.28887671,
    0.4237656,
    0.56420489,
    0.71197656,
    0.8418588,
    0.93131528,
    0.97634216
]

OUTCOME_THRESHOLDS = [
    [0.5, 0.4, 0.065, 0.025, 0.01]
]


# global variables
people = []

# metrics
active = {}
new_infected = {}
deaths = {}
susceptible = {}
in_incubation = {}
contagious = {}
infected = {}
severe = {}

class  Outcome(Enum) :
    NO_INFECTION = 0
    NO_SYMPTOMS = 1
    MILD_to_MODERATE = 2
    SEVERE = 3
    DEATH = 4

def get_outcome(severity):
    p = np.random.random()
    if p > severity[0]:
        return Outcome.NO_INFECTION
    elif p > severity[1]:
        return Outcome.NO_SYMPTOMS
    elif p > severity[2]:
        return Outcome.MILD_to_MODERATE
    elif p > severity[3]:
        return Outcome.SEVERE
    return Outcome.DEATH

# setup Person object
class Person(object):
    def __init__(self, env, i,age):
        self.env = env
        self.process = env.process
        self.timeout = env.timeout
        self.id = i
        self.infected = False
        self.in_incubation = False
        self.contagious = False
        self.susceptible = True
        self.vaccinated = False
        self.dead = False 
        self.masks_usage = MASK
        self.in_quarantine = False
        self.expected_outcome = 0
        self.age=age
        self.severity_group=OUTCOME_THRESHOLDS[0]
        self.transmited =0
        self.diagnosed = False
        self.active=False
        

    def expose_to_virus(self):
        if not self.susceptible:
            return False
        self.susceptible = False
        expected_outcome = get_outcome(self.severity_group)
        if expected_outcome == Outcome.NO_INFECTION:
            return False
        else:
            self.expected_outcome = expected_outcome
        incubation_time = np.random.weibull(SYMPTOMS_DELAY_SHAPE) * SYMPTOMS_DELAY_SCALE
        self.infected = True
        self.active = True
        self.process(self.run_contagion(incubation_time))
        return True

    def run_contagion(self,incubation_time):
        self.in_incubation = True
        yield self.timeout(incubation_time)
        self.in_incubation = False
        self.contagious = True
        self.process(self.run_contagion_street())
        contagion_duration = np.random.weibull(CONTAGION_DURATION_SHAPE) * CONTAGION_DURATION_SCALE
        self.progression()
        yield self.timeout(contagion_duration)
        self.contagious = False
    
    def run_contagion_street(self):
        yield self.timeout(np.random.exponential(STREET_EXPOSE_INTERVAL))
        while self.contagious:
            if self.test_street_transmission():
                contact_on_street=get_not_infected()
                if contact_on_street.test_street_infection():
                    if contact_on_street.expose_to_virus():
                        self.transmited += 1
            yield self.timeout(np.random.exponential(STREET_EXPOSE_INTERVAL))

    def progression(self):
        if self.expected_outcome == Outcome.DEATH:
            self.progression_death()
        elif self.expected_outcome == Outcome.SEVERE:
            self.progression_severe()
        elif self.expected_outcome == Outcome.MILD_to_MODERATE:
            self.progression_mild_to_moderate()
        else:
            self.progression_no_symptoms()

    def progression_death(self):
        time_until_outcome = np.random.weibull(2) * 17
        self.diagnosed = True
        self.in_quarantine = True
        self.process(self.run_death(time_until_outcome))

    def progression_severe(self):
        time_until_outcome = np.random.weibull(SEVERE_SHAPE) * SEVERE_SCALE
        self.wait_for_diagnosis(DIAGNOSIS_DELAY)
        self.process(self.run_cure(time_until_outcome))
    
    def progression_mild_to_moderate(self):
        time_until_outcome = np.random.weibull(2) * 20       
        if CHANCE_OF_DIAGNOSIS_IF_MODERATE > np.random.random():
            self.wait_for_diagnosis(DIAGNOSIS_DELAY)
        self.process(self.run_cure(time_until_outcome))

    def progression_no_symptoms(self):
        time_until_outcome = np.random.weibull(2) * 15   
        self.process(self.run_cure(time_until_outcome))

    def wait_for_diagnosis(self, DIAGNOSIS_DELAY):
        time_for_diagnosis = np.random.weibull(4) * DIAGNOSIS_DELAY
        yield self.timeout(DIAGNOSIS_DELAY)
        self.diagnosed = True
        self.in_quarantine = True

    def run_cure(self, cure):
        yield self.timeout(cure)
        self.in_quarantine = False
        self.active = False
        #self.remove_immunization()

    def run_death(self, time_until_death):
        yield self.timeout(time_until_death)
        self.active = False 
        self.contagious = False
        self.susceptible = False
        self.in_quarantine = False
        self.dead = True

    def test_quarantine(self):
        """
        Test if a person's quarantine can avoid a transmission
        """
        return self.in_quarantine and np.random.random() < QUARANTINE_EFFECTIVENESS

    def test_street_transmission(self):
        """
        Test if a person can transmit to others in the street, given person's containment measures
        """
        return (
            (not self.test_quarantine()) 
            and self.test_mask_transmission()
        )

    def test_street_infection(self):
        """
        Test if a person can be infected in the street, given person's containment measures
        """
        return (
            self.susceptible 
            and (not self.test_quarantine()) 
            and self.test_mask_infection()
        ) 

    def test_mask_transmission(self):
        """
        Test if a person's mask usage can prevent transmission
        """
        if self.masks_usage > np.random.random():
            if MASK_TRANSMISSION_REDUCTION > np.random.random():  # mask was effective
                return 0
        return 1

    def test_mask_infection(self):
        """
        Test if a person's mask usage can avoid infection
        """
        if self.masks_usage > np.random.random():
            if MASK_INFECTION_REDUCTION > np.random.random():  # mask was effective
                return 0
        return 1

                

def collect_metrics(env, people):
    while True:
        active[env.now] = sum([person.active for person in people])
        infected[env.now] = sum([person.diagnosed for person in people])
        if env.now == 0 :
            new_infected[env.now] =infected[env.now]
        else :
            new_infected[env.now] =infected[env.now]-infected[env.now-1]
        deaths[env.now] = sum([person.dead for person in people])
        susceptible[env.now] = N_POPULATION-sum([person.susceptible for person in people])
        in_incubation[env.now] = sum([person.in_incubation for person in people])
        contagious[env.now] = sum([person.contagious for person in people])
        severe[env.now] = sum([((person.expected_outcome==Outcome.SEVERE) and person.active) for person in people])
        
        yield env.timeout(1)

def get_not_infected():
    global people
    num=100
    while num>=0:
        choose =rd.choice(people)
        if not (choose.infected):
            return choose
        num-=1
    return people[choose]

def set_initial_infection():
    success = False
    while not success:
        someone = get_not_infected()
        success = someone.expose_to_virus()

def init_age():
    p = np.random.random()
    if p < age_dist[0]:
        return 0
    elif p < age_dist[1]:
        return 1
    elif p < age_dist[2]:
        return 2
    elif p < age_dist[3]:
        return 3
    elif p < age_dist[4]:
        return 4
    elif p < age_dist[5]:
        return 5
    elif p < age_dist[6]:
        return 6
    elif p < age_dist[7]:
        return 7
    return 8

# main
def main():
    global people
    # setup environment
    env = simpy.Environment()

    # create people
    people = [Person(env, i,init_age()) for i in range(N_POPULATION)]
   
    # initial infection
    for _ in range(INITIALLY_INFECTED):
        set_initial_infection()

    env.process(collect_metrics(env, people)) # collect metrics at the end of each day
    
    # run simulation
    env.run(until=SIM_TIME)

    #for i in range(len(age_str)):
    #    print(age_str[i]+" : "+(str)(sum([x.age==i for x in people])))
    
    # export 
    df = pd.DataFrame({
        'active':pd.Series(active),
        'new_diagnosed':pd.Series(new_infected),
        'deaths':pd.Series(deaths),
        'susceptible':pd.Series(susceptible),
        'in_incubation':pd.Series(in_incubation),
        'contagious':pd.Series(contagious),
        'infected':pd.Series(infected),
        'severe':pd.Series(severe)

    })
    filename = __file__.split('.')[0]
    df.to_csv(f'{filename}.csv', index=True)

if __name__ == "__main__":
    main()
