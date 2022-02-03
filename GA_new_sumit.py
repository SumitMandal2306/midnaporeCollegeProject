import random
import matplotlib.pyplot as plt
import sys


def distinct_checking(populations, index):
    if index > 0:
        temp = index - 1
        while temp >= 0:
            if populations[index] == populations[temp]:
                random.shuffle(populations[index])
                temp = index - 1
            else:
                temp -= 1


# def datamatrix(size):
#     matrix = []
#     temp_list = []
#     for i in range(0, size):
#         for j in range(0, size):
#             temp = random.randint(0, 99)
#             temp_list.append(temp)
#         matrix.append(temp_list.copy())
#         temp_list.clear()
#     return matrix


# def generate_city(size):
#     city_list = []
#     for i in range(0, size):
#         city_list.append(i)
#     return city_list


def readTSPData(fileName):
    sourceFile = open(fileName, "r")
    rawData = sourceFile.read()
    sourceFile.close()
    formattedData = []

    temp = ""
    tempLine = []
    for i in rawData:
        if i != " ":
            temp += str(i)
        if i == " " or i == "\n":
            if temp != "":
                temp = float(temp)
                tempLine.append(temp)
                temp = ""
        if i == "\n":
            formattedData.append(tempLine)
            tempLine = []
    temp = float(temp)
    tempLine.append(temp)
    formattedData.append(tempLine)
    return formattedData


# put data text file name in the readTSPData function to read the tsp data
dataMatrix = readTSPData("data.txt")
cityList = [i for i in range(0, len(dataMatrix))]

# numberOfData = 10
popSize = 50

# dataMatrix = datamatrix(numberOfData)
# cityList = generate_city(numberOfData)

eliteSize = int(popSize / 5.2)
mutationRate = 0.05
crossOverRate = 1
generationNo = 100
chromosomeRollNo = 0


class chromosomes:
    def __init__(self, route, chromosomeRollNo, parents=["NA", "NA"]):
        self.chromosomeRollNo = chromosomeRollNo
        self.route = route
        self.distance = 0.0
        self.fitnessScore = 0.0
        self.parents = parents

    def __repr__(self):
        # return " " + str(self.chromosomeRollNo) + ") " + str(self.route) + " Route Distance = "+ str(self.distance)
        # + " Fitness = " + str(self.fitnessScore) + "\n"
        return " " + str(self.chromosomeRollNo) + ") " + str(self.route) + " Cost = " + str(
            self.distance) + " Parents= " + str(self.parents) + "\n"


def generateInitialPopulation(popSize, initialPopulation, cityList):
    def factorial(n):
        if n == 1:
            return n
        else:
            return n * factorial(n - 1)

    global chromosomeRollNo

    if popSize <= 1:
        print("Please enter a valid population size, it must be greater than 1.")
        sys.exit()

    else:
        if popSize > len(cityList):
            if popSize > factorial(len(cityList)):
                print("Can't generate desired number of population, their can be a duplicate.")
                sys.exit()
            else:
                for i in range(0, popSize):
                    chromosome = chromosomes(random.sample(cityList, len(cityList)), chromosomeRollNo)
                    chromosomeRollNo += 1
                    initialPopulation.append(chromosome)
                    distinct_checking(initialPopulation, i)
        else:
            for i in range(0, popSize):
                chromosome = chromosomes(random.sample(cityList, len(cityList)), chromosomeRollNo)
                chromosomeRollNo += 1
                initialPopulation.append(chromosome)
                distinct_checking(initialPopulation, i)
        # print("initial population :", initialPopulation)


def fitnessOperator(chromosome):
    route = chromosome.route
    totalDistance = 0
    fromCity = 0
    toCity = 0
    for i in range(0, len(route) - 1):
        fromCity = int(route[i])
        toCity = int(route[i + 1])
        totalDistance += dataMatrix[fromCity][toCity]
    fromCity = toCity
    toCity = int(route[0])
    totalDistance += dataMatrix[fromCity][toCity]
    return totalDistance


def assignFitness(population):
    for i in population:
        i.distance = fitnessOperator(i)
        i.fitnessScore = 1 / i.distance
        # print("fitness score : population - ", i, i.fitnessScore)


def elitism(population, eliteChromosomes, eliteSize):
    sortedPopulation = sorted(population, key=lambda x: x.fitnessScore, reverse=True)
    for i in range(0, eliteSize):
        eliteChromosomes.append(sortedPopulation[i])


def rwSelection(population):
    totalFitness = 0
    for i in population:
        totalFitness += i.fitnessScore
    P = random.random()
    N = 0.0

    # print(totalFitness, P)

    for i in population:
        N += i.fitnessScore / totalFitness
        temp = i.fitnessScore / totalFitness

        if N > P:
            print(i, " cumulative fitness value :", temp)
            return i
        temp = 0


def tournament_selection(chromosomes, participents_no=5):
    selected_participents = []

    for i in range(0, participents_no):
        selected_participents.append(random.choice(chromosomes))
    selected_participents.sort(key=lambda x: x.fitnessScore, reverse=True)
    selected_parent = selected_participents[0]
    selected_participents.clear()
    return selected_parent


def selectParents(population, matingPool, numberOfParents):
    temp_population = sorted(population, key=lambda x: x.fitnessScore, reverse=True)
    for i in range(0, numberOfParents):
        selectedParent = rwSelection(population)

        # selectedParent = tournament_selection(population)

        matingPool.append(selectedParent)

    # run only when using tournament selection for other selection comment out this part
    # for i in range(0, eliteSize):
    #     matingPool.append(temp_population[i])
    # random.shuffle(matingPool)


def orderedCrossOver(parent1, parent2):
    child = []
    parent1subset = []
    parent2subset = []
    randomPoint1 = random.randint(0, (len(parent1) - 1))
    randomPoint2 = random.randint(0, (len(parent1) - 1))
    startGene = min(randomPoint1, randomPoint2)
    endGene = max(randomPoint1, randomPoint2)

    for i in range(startGene, endGene):
        parent1subset.append(parent1[i])

    parent2subset = [item for item in parent2 if item not in parent1subset]
    child = parent1subset + parent2subset
    return child


def single_point_crossover(parent1, parent2):
    child1 = []
    child2 = []
    point = random.randint(0, len(parent1) - 1)
    nextPoint = point + 1

    # making the first child
    if point == 0:
        child1.append(parent1[point])
    else:
        for i in range(0, nextPoint):
            child1.append(parent1[i])
    for i in range(nextPoint, len(parent2)):
        child1.append(parent2[i])

    # making the second child
    if point == 0:
        child2.append(parent2[point])
    else:
        for i in range(0, nextPoint):
            child2.append(parent2[i])
    for i in range(nextPoint, len(parent1)):
        child2.append(parent1[i])

    if fitnessOperator(chromosomes(child1, None)) < fitnessOperator(chromosomes(child2, None)):
        return child1
    else:
        return child2


def two_point_crossover(parent1, parent2):
    child1 = []
    child2 = []
    point1 = random.randint(0, len(parent1) - 1)
    point2 = random.randint(0, len(parent1) - 1)
    while point2 == point1:
        point2 = random.randint(0, len(parent1) - 1)
    start = min(point1, point2)
    end = max(point1, point2)

    # making the first child
    if start == 0:
        child1.append(parent1[start])
    else:
        for i in range(0, start + 1):
            child1.append(parent1[i])
    for i in range(start + 1, end + 1):
        child1.append(parent2[i])
    for i in range(end + 1, len(parent1)):
        child1.append(parent1[i])

    # making the second child
    if start == 0:
        child2.append(parent2[start])
    else:
        for i in range(0, start + 1):
            child2.append(parent2[i])
    for i in range(start + 1, end + 1):
        child2.append(parent1[i])
    for i in range(end + 1, len(parent2)):
        child2.append(parent2[i])

    if fitnessOperator(chromosomes(child1, None)) < fitnessOperator(chromosomes(child2, None)):
        return child1
    else:
        return child2


def generateChildren(matingPool, children, eliteSize, crossOverRate):
    global chromosomeRollNo
    count = len(matingPool)
    random.shuffle(matingPool)

    for i in range(0, count - eliteSize):
        parent1 = matingPool[i]
        parent2 = matingPool[i + 1]
        child = []
        childRoute = []

        if random.random() < crossOverRate:
            # childRoute = orderedCrossOver(parent1.route, parent2.route)
            # childRoute = single_point_crossover(parent1.route, parent2.route)
            childRoute = two_point_crossover(parent1.route, parent2.route)
            parents = [parent1.chromosomeRollNo, parent2.chromosomeRollNo]
            child = chromosomes(childRoute, chromosomeRollNo, parents)
            chromosomeRollNo += 1
        else:
            temp = [parent1, parent2]
            child = random.choice(temp)

        children.append(child)


def mutate(chromosome, mutationRate):
    if random.random() < mutationRate:
        route = chromosome.route
        routeLength = len(route) - 1
        position1 = random.randint(0, routeLength)
        position2 = random.randint(0, routeLength)
        temp = route[position1]
        route[position1] = route[position2]
        route[position2] = temp
        chromosome.route = route
    return chromosome


def mutateChildren(children, mutationRate):
    for i in children:
        i = mutate(i, mutationRate)
    return children


def createNextGeneration(eliteChromosomes, children, nextGeneration):
    for i in eliteChromosomes:
        nextGeneration.append(i)
    for i in children:
        nextGeneration.append(i)


def geneticAlgorithm():
    costList = []

    initialPopulation = []
    generateInitialPopulation(popSize, initialPopulation, cityList)
    population = []
    population = initialPopulation

    for i in range(0, generationNo):
        print("\n")
        print("/////////////////////////////////////////////////////////")
        print("/////_____________ GENERATION: ", i + 1, "_______________//")
        print("/////////////////////////////////////////////////////////")
        print("\n")
        assignFitness(population)

        print("/////_____________ POPULATION _______________//")
        print(population)

        eliteChromosomes = []
        elitism(population, eliteChromosomes, eliteSize)

        print("/////_____________ BEST CHROMOSOMES OF THIS GENERATION _______________//")
        print(eliteChromosomes)

        matingPool = []
        selectParents(population, matingPool, popSize)
        print("/////_____________ SELECTED PARENTS FOR CROSSOVER _______________//")
        print(matingPool)

        children = []
        generateChildren(matingPool, children, eliteSize, crossOverRate)
        assignFitness(children)
        print("/////_____________ GENERATED CHILDREN FROM CROSSOVER _______________//")
        print(children)

        mutateChildren(children, mutationRate)
        assignFitness(children)
        print("/////_____________ APPLYING MUTATION ON CHILDREN _______________//")
        print(children)

        nextGeneration = []
        createNextGeneration(eliteChromosomes, children, nextGeneration)

        print(
            "/////_____________ NEXT GENERATION AFTER APPENDING ELITE CHROMOSOMES AND CHILDREN TOGETHER _______________//")
        print(nextGeneration)
        population = nextGeneration

        sortedPopulation = sorted(population, key=lambda x: x.distance)
        print("/////_____________ BEST CHROMOSOME: ", sortedPopulation[0], " _______________//")

        costList.append(sortedPopulation[0].distance)

        # userInput = input("PRESS 1 FOR NEXT GENERATION, 0 FOR STOP: ")
        # if(userInput == "0"):
        #   break

    return costList


costList = geneticAlgorithm()

plt.title("TSP USING GENETIC ALGORITHM\n MIN COST = " + str(costList[-1]))
plt.xlabel("Generations")
plt.ylabel("Cost")
plt.plot(costList, marker="o", mfc="#db513b", mec="#db513b", linestyle='dashed', color="#5fcc3d")
plt.show()
