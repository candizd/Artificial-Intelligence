import random as rnd
import numpy as np
import matplotlib.pyplot as plt


POPULATION_SIZE = 12
MUTATION_RATE = 0.3
TOURNAMENT_SELECTION_SIZE = 6

class lecture:
    def __init__(self, id, instructors, semester=None, elective=None):
        self.id = id
        self.instructors = instructors
        self.semester = semester
        self.elective = elective

    def __str__(self):
        return self.id + "by" + self.instructors

l1 = lecture("MAT103", "IG", "1")
l2 = lecture("INF101", "VG", "1")
l3 = lecture("INF103", "FB", "1")
l4 = lecture("INF107", "FB", "1")
l5 = lecture("DEU121", "DEU", "1")
l6 = lecture("ENG101", "ENG", "1")
l7 = lecture("TUR001", "TUR", "1")
l8 = lecture("INF201", "CY", "3")
l9 = lecture("INF203", "EMY", "3")
l10 = lecture("INF205", "CY", "3")
l11 = lecture("INF209", "FB", "3")
l12 = lecture("INF211", "CY", "3")
l13 = lecture("ENG201", "ENG", "3")
l14 = lecture("AIT001", "AIT", "3")
l15 = lecture("INF303", "OK", "5")
l16 = lecture("ISG001", "ISG", "5")
l17 = lecture("ENG301", "ENG", "5")
l18 = lecture("INF506", "EI", None , True)
l19 = lecture("INF523", "DG", None , True)
l20 = lecture("INF517", "SI", None , True)
l21 = lecture("INF701", "CY", None , True)
l22 = lecture("INF714", "EI", None , True)
l23 = lecture("INF905", "BB", None , True)
l24 = lecture("MEC313","HS","3", True)
l25 = lecture("ETE091", "KC", "3")
l26 = lecture("INF499", "OM", "5")
l27 = lecture("INF401", "OM", "7")
l28 = lecture("INF106","CY","1")
l29 = lecture("INF207","AY","3")
l30 = lecture("INF301","AY","5")

lectures = [l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12, l13, l14, l15, l16, l17, l18, l19, l20, l21, l22, l23, l24, l25, l26, l27, l28, l29, l30]
rooms =  ["R1", "R2", "R3", "R4","R5","R6","R7","R8"] 
slots = ["T1", "T2", "T3", "T4", "T5","T6"]
days = ["M", "TUE", "W", "TH", "F"]

class scheduledLecture:
    def __init__(self, lecture, room, slot, day):
        self.lecture = lecture
        self.room = room
        self.slot = slot
        self.day = day

    def __str__(self):
        return "[" + self.lecture.id + ", " + self.lecture.instructors + ", " + self.room + ", " + self.day + ", " + self.slot + "]"

class schedule:
    def __init__(self, scheduledLectures):
        self.scheduledLectures = scheduledLectures
        self.fitness = calculateFitness(scheduledLectures)

    def __str__(self):
        s = ""
        for ind in self.scheduledLectures:
            s = s + "[" + ind.lecture.id + ", " + ind.lecture.instructors + ", " + ind.room + ", " + ind.day + ", " + ind.slot + "] "
        return s

def createPopulation():
    ##Create a random population
    for i in range(POPULATION_SIZE):
        global population
        scheduled = []
        for k in lectures:
            scheduled.append(scheduledLecture(k, rooms[rnd.randrange(0, len(rooms))], slots[rnd.randrange(0, len(slots))], days[rnd.randrange(0, len(days))]))

        population.append(schedule(scheduled))
    
    population = sorted(population, key=lambda a : a.fitness)


def calculateFitness(individual):
    ##Calculate fittness value for each individual 
    fitness = 0
    for i in individual:
        for k in individual:
            if i == k: continue
            if i.day == k.day and i.slot == k.slot:
                if i.room == k.room:
                    fitness += 1
                if i.lecture.instructors == k.lecture.instructors:
                    fitness += 1
                   
                if (i.lecture.elective and k.lecture.elective) or (i.lecture.elective and k.lecture.semester == "5")  or (i.lecture.semester == "5" and k.lecture.elective):
                    fitness += 1
                    
                if i.lecture.semester == k.lecture.semester:
                    fitness += 1
                    
    return fitness // 2


def selectParent():
    parents = []
    for ind in population:
        #Select random 4 parents and order them according to their fitness values
        i = 0
        while i < TOURNAMENT_SELECTION_SIZE:
            parents.append(population[rnd.randrange(0, POPULATION_SIZE)])
            i = i + 1
        
    parents.sort(key=lambda x: x.fitness, reverse= True)    
    
    return parents

def crossover(parents):
    #Select 2 random crosspoints and generate offsprings
    point1 = rnd.randint(0, len(parents[0].scheduledLectures)- (len(lectures) // 2) + 1 )
    point2 = rnd.randint(point1, len(parents[1].scheduledLectures))

    offspring_1 = schedule(parents[0].scheduledLectures[:point1] + parents[1].scheduledLectures[point1:point2] + parents[0].scheduledLectures[point2:])
    offspring_2 = schedule(parents[1].scheduledLectures[:point1] + parents[0].scheduledLectures[point1:point2] + parents[1].scheduledLectures[point2:])

    return offspring_1, offspring_2

def mutation(offspring):
    #Change a random value if mutation happens
    for ind in offspring.scheduledLectures:
        if rnd.random() < MUTATION_RATE:
            dice = rnd.randrange(1,3)
            if dice == 1:
                ind.room = rooms[rnd.randrange(0, len(rooms))]
            elif dice == 2:
                ind.day = days[rnd.randrange(0, len(days))]
            elif dice == 3:
                ind.slot = slots[rnd.randrange(0, len(slots))]

    return offspring

def selection(offsprings):
    global population
    #make a new generation with the best individuals
    new_gen = population + offsprings
    new_gen = sorted(new_gen, key=lambda ind: ind.fitness)[:POPULATION_SIZE]
   
    population = new_gen


def evolution():

    offsprings = []

    while len(offsprings) != len(population):
        
        parents = selectParent()

        offspring_1, offspring_2 = crossover(parents)

        offspring_1 = mutation(offspring_1)
        offspring_2 = mutation(offspring_2)

        #Check if the offsprings list has a duplicate in it
        control1 = True
        for ind in offsprings:            
            if (ind.__str__() == offspring_1.__str__()) or (ind.__str__() == offspring_2.__str__()):
                control1 = False
                break
        #Check if the offsprings already exists in population
        control2 = True
        for ind in population:
            if (ind.__str__() == offspring_1.__str__()) or (ind.__str__() == offspring_2.__str__()):
                control2 = False
                break
       

        if control1 and control2:
            offsprings.append(offspring_1)
            offsprings.append(offspring_2)

        
    selection(offsprings)


# Start
gens = []
runs = []
for run in range(10):
    population = []

    createPopulation()
    print(f'Run: {run}')
    control = False
    solution = None

    gen = 0
    while gen < 500:
        print("GEN: " + str(gen))

        index = 1
        for i in population:
            
            print("Individual " + str(index) + " with fitness score " + str(i.fitness))
            print(i)

            if i.fitness == 0:
                control = True
                solution = i
            
            index += 1
        gen += 1

        if control: break
        elif control == False:
            evolution()

    if control:
        print("Solution found")
        print("Generation: " + str(gen))
        print(solution)
        gens.append(gen) 
        runs.append(run)
    else:
        print("Solution not found")

print(f'Mean: {np.mean(gens)}')
print(f'St. dev: {np.std(gens)}')
print(f'Min: {min(gens)}')
print(f'Max: {max(gens)}')

x = np.array(runs)
y = np.array(gens)

#create basic scatterplot
plt.plot(x, y, 'o')

#obtain m (slope) and b(intercept) of linear regression line
m, b = np.polyfit(x, y, 1)

#add linear regression line to scatterplot 
plt.plot(x, m*x+b)

plt.xlabel("runs")
plt.ylabel("gens")
plt.show()
