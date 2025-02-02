import random
import matplotlib.pyplot as plt


#Genetic Algorithm Parameters
popSize = 10
eliteSize = 2
mutationRate = .05
crossOverRate = 1
generationNo = 2
chromosomeRollNo = 0


plotFlag = 1 # Assign 1 for plotting, 0 for not plotting



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


#put data text file name in the readTSPData function to read the tsp data
dataMatrix = readTSPData("data.txt")
cityList = [ i for i in range(0,len(dataMatrix)) ] #City List


class chromosomes:
  def __init__(self,route,chromosomeRollNo, parents=["NA","NA"]):
    self.chromosomeRollNo = chromosomeRollNo
    self.route = route
    self.distance = 0.0
    self.fitnessScore = 0.0
    self.parents = parents
    
  def __repr__(self):
    #return " " + str(self.chromosomeRollNo) + ") " + str(self.route) + " Route Distance = "+ str(self.distance) + " Fitness = " + str(self.fitnessScore) + "\n"
    return " " + str(self.chromosomeRollNo) + ") " + str(self.route) + " Cost = " + str(self.distance) + " Parents= "+ str(self.parents) + "\n"


def generateInitialPopulation(popSize, initialPopulation, cityList):
  global chromosomeRollNo
  count = popSize
  for i in range(0,count):
    chromosome = chromosomes(random.sample(cityList, len(cityList)),chromosomeRollNo)
    if chromosome in initialPopulation:
      while(chromosome in initialPopulation):
        chromosome = chromosomes(random.sample(cityList, len(cityList)),chromosomeRollNo)
    chromosomeRollNo += 1
    initialPopulation.append(chromosome)


def fitnessOperator(chromosome):
  route = chromosome.route
  totalDistance = 0
  fromCity = 0
  toCity = 0
  for i in range(0,len(route)-1):
    fromCity = int(route[i])
    toCity = int(route[i+1])
    totalDistance += dataMatrix[fromCity][toCity]
  fromCity = toCity
  toCity = int(route[0])
  totalDistance += dataMatrix[fromCity][toCity]
  return totalDistance


def assignFitness(population):
  for i in population:
    i.distance = fitnessOperator(i)
    i.fitnessScore = 1/i.distance


def elitism(population,eliteChromosomes,eliteSize):
  sortedPopulation =  sorted(population, key= lambda x : x.fitnessScore, reverse = True)
  for i in range(0,eliteSize):
    eliteChromosomes.append(sortedPopulation[i])


def rwSelection(population):
  totalFitness = 0
  for i in population:
    totalFitness += i.fitnessScore
  P = random.random()
  N = 0.0
  for i in population:
    N += i.fitnessScore/totalFitness
    if (N>P):
      return i


def selectParents(population,matingPool,numberOfParents):
  for i in range(0,numberOfParents):
    selectedParent = rwSelection(population)
    matingPool.append(selectedParent)


def orderedCrossOver(parent1, parent2):
  child = []
  parent1subset = []
  parent2subset = []
  randomPoint1 = random.randint(0,(len(parent1) - 1))
  randomPoint2 = random.randint(0,(len(parent1) - 0))
  startGene = min(randomPoint1, randomPoint2)
  endGene = max(randomPoint1, randomPoint2)

  for i in range(startGene, endGene):
    parent1subset.append(parent1[i])
  
  parent2subset = [item for item in parent2 if item not in parent1subset]
  child = parent1subset + parent2subset
  return child


def generateChildren(matingPool, children, eliteSize, crossOverRate):
  global chromosomeRollNo
  count = len(matingPool)
  random.shuffle(matingPool)

  for i in range(0,count - eliteSize):
    parent1 = matingPool[i]
    parent2 = matingPool[i+1]
    child = []
    childRoute = []

    if(random.random() < crossOverRate):
      childRoute = orderedCrossOver(parent1.route,parent2.route)
      parents = [parent1.chromosomeRollNo,parent2.chromosomeRollNo]
      child = chromosomes(childRoute,chromosomeRollNo,parents)
      chromosomeRollNo += 1
    else:
      temp = [parent1,parent2]
      child = random.choice(temp)

    children.append(child)


def mutate(chromosome,mutationRate):
  if random.random() < mutationRate:
    route = chromosome.route
    routeLength = len(route) - 1
    position1 = random.randint(0,routeLength)
    position2 = random.randint(0,routeLength)
    temp = route[position1]
    route[position1] = route[position2]
    route[position2] = temp
    chromosome.route = route
  return chromosome


def mutateChildren(children,mutationRate):
  for i in children:
    i = mutate(i,mutationRate)
  return children


def createNextGeneration(eliteChromosomes,children,nextGeneration):
  for i in eliteChromosomes:
    nextGeneration.append(i)
  for i in children:
    nextGeneration.append(i)


def geneticAlgorithm():
  costList =[]

  initialPopulation = []
  generateInitialPopulation(popSize,initialPopulation,cityList)
  population = []
  population = initialPopulation

  for i in range(0,generationNo):
    print("\n")
    print("/////////////////////////////////////////////////////////")
    print("/////_____________ GENERATION: ", i+1 ,"_______________//")
    print("/////////////////////////////////////////////////////////")
    print("\n")
    assignFitness(population)

    print("/////_____________ POPULATION _______________//")
    print(population)

    eliteChromosomes = []
    elitism(population,eliteChromosomes,eliteSize)

    print("/////_____________ BEST CHROMOSOMES OF THIS POPULATION _______________//")
    print(eliteChromosomes)

    matingPool = []
    selectParents(population,matingPool,popSize)
    print("/////_____________ SELECTED PARENTS FOR CROSSOVER _______________//")
    print(matingPool)

    children = []
    generateChildren(matingPool,children,eliteSize,crossOverRate)
    assignFitness(children)
    print("/////_____________ GENERATED CHILDREN FROM CROSSOVER _______________//")
    print(children)

    mutateChildren(children,mutationRate)
    assignFitness(children)
    print("/////_____________ APPLYING MUTATION ON CHILDREN _______________//")
    print(children)

    nextGeneration = []
    createNextGeneration(eliteChromosomes,children,nextGeneration)

    print("/////_____________ NEXT GENERATION AFTER APPENDING ELITE CHROMOSOMES AND CHILDREN TOGETHER _______________//")
    print(nextGeneration)
    population = nextGeneration

    sortedPopulation = sorted(population,key = lambda x : x.distance)
    print("/////_____________ BEST CHROMOSOME: ",sortedPopulation[0]," _______________//")
    
    costList.append(sortedPopulation[0].distance)

    # userInput = input("PRESS 1 FOR NEXT GENERATION, 0 FOR STOP: ")
    # if(userInput == "0"):
    #   break


  return costList
  

costList = geneticAlgorithm()


#plotting

if plotFlag == 1:
  plt.title("TSP USING GENETIC ALGORITHM\n MIN COST = " + str(costList[-1]))
  plt.xlabel("Generations")
  plt.ylabel("Cost")
  plt.plot(costList, marker="o", mfc="#db513b", mec="#db513b", linestyle = 'dashed', color="#5fcc3d")
  plt.show()
