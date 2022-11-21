from array import *
import random
import matplotlib.pyplot as plt
import numpy as np
Carry2 = []
for run in range(100):
	Gen = np.array([])
	Fit = np.array([])
	Gen2 = np.array([])
	##Create Graph
	n = 6
	graph = []
	for i in range(n):
		vertex = []
		for j in range(n):
			vertex.append(random.randint(0, 1))
		graph.append(vertex)
	for i in range(n):
		for j in range(0, i):
			graph[i][j] = graph[j][i]
	for i in range(n):
		graph[i][i] = 0
	for v in graph:
		print(v)


	max_num_colors = 5
	number_of_colors = max_num_colors
	
	##GA
	condition = True
	while(condition and number_of_colors > 0):
		def create_individual():
			individual = []
			for i in range(n):
				individual.append(random.randint(1, number_of_colors))
			return individual
		##Create Population
		population_size = 200
		generation = 0
		population = []
		for i in range(population_size):
			individual = create_individual()
			population.append(individual)

		##Fitness
		def fitness(graph, individual):
			fitness = 0
			for i in range(n):
				for j in range(i, n):
					if(individual[i] == individual[j] and graph[i][j] == 1):
						fitness += 1
			return fitness

		##Crossover
		def crossover(parent1, parent2):
			position = random.randint(2, n-2)
			child1 = []
			child2 = []
			for i in range(position+1):
				child1.append(parent1[i])
				child2.append(parent2[i])
			for i in range(position+1, n):
				child1.append(parent2[i])
				child2.append(parent1[i])
			return child1, child2

		def mutation(individual):
			probability = 0.4
			check = random.uniform(0, 1)
			if(check <= probability):
				position = random.randint(0, n-1)
				individual[position] = random.randint(1, number_of_colors)
			return individual

		##Selection
		def tournament_selection(population):
			new_population = []
			for j in range(2):
				random.shuffle(population)
				for i in range(0, population_size-1, 2):
					if fitness(graph, population[i]) < fitness(graph, population[i+1]):
						new_population.append(population[i])
					else:
						new_population.append(population[i+1])
			return new_population

		
		best_fitness = fitness(graph, population[0])
		fittest_individual = population[0]
		gen = 0

		
		while(best_fitness != 0 and gen != 1000):
			gen += 1
			population = tournament_selection(population)
			new_population = []
			random.shuffle(population)
			for i in range(0, population_size-1, 2):
				child1, child2 = crossover(population[i], population[i+1])
				new_population.append(child1)
				new_population.append(child2)
			for individual in new_population:
				if(gen < 200):
					individual = mutation(individual)
				else:
					individual = mutation(individual)
			population = new_population
			best_fitness = fitness(graph, population[0])
			fittest_individual = population[0]
			for individual in population:
				if(fitness(graph, individual) < best_fitness):
					best_fitness = fitness(graph, individual)
					fittest_individual = individual
			if gen % 10 == 0:
				print("Generation: ", gen, "Best_Fitness: ",
					best_fitness, "Individual: ", fittest_individual)

			Gen2 = np.append(Gen2,gen)
			Gen = np.append(Gen, gen)
			Fit = np.append(Fit, best_fitness)
			
			
		print("Using ", number_of_colors, " colors : ")
		print("Generation: ", gen, "Best_Fitness: ",
			best_fitness, "Individual: ", fittest_individual)
		print("\n\n")
		if(best_fitness != 0):
			condition = False
			print("Graph is ", number_of_colors+1, " colorable")
			Carry2.append(Gen2[-1001])
			print(Carry2)
		else:
			Gen = np.append(Gen, gen)
			Fit = np.append(Fit, best_fitness)
			Gen = []
			Fit = []
			number_of_colors -= 1

    
print(f'Mean: {np.mean(Carry2)}')
print(f'St. dev: {np.std(Carry2)}')
print(f'Min: {min(Carry2)}')
print(f'Max: {max(Carry2)}')
