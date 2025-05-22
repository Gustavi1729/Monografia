from Listas import *
from pymoo.util.dominator import Dominator
from scipy.spatial.distance import pdist, squareform
import random
import importlib
import matplotlib.pyplot as plt
import pandas as pd
from pymoo.indicators.hv import HV
dom = Dominator()


def fast_non_dominated_sort(popOriginal, population):
    """
    Perform fast non-dominated sorting for NSGA-II.
    
    Args:
        population: A 2D NumPy array where each row represents a solution and each column is an objective.
    
    Returns:
        fronts: A list of Pareto fronts, where each front is a list of indices.
        ranks: A NumPy array of ranks, where rank[i] is the front number of solution i.
    """
    num_solutions = population.shape[0]
    ranks = np.zeros(num_solutions, dtype=int)
    
    # Domination count and dominated solutions list
    domination_count = np.zeros(num_solutions, dtype=int)
    dominated_solutions = [[] for _ in range(num_solutions)]
    
    # Initial front
    front_0 = []

    # Compute dominance relationships
    for i in range(num_solutions):
        for j in range(i + 1, num_solutions):
            # Compare solution i and solution j
            if np.all(population[i] <= population[j]) and np.any(population[i] < population[j]):
                # i dominates j
                domination_count[j] += 1
                dominated_solutions[i].append(j)
            elif np.all(population[j] <= population[i]) and np.any(population[j] < population[i]):
                # j dominates i
                domination_count[i] += 1
                dominated_solutions[j].append(i)
        
        # If no one dominates this solution, it's in the first front
        if domination_count[i] == 0:
            ranks[i] = 0
            front_0.append(i)
    
    # Generate subsequent fronts
    fronts = [front_0]
    current_front = front_0
    rank = 0
    
    while current_front:
        next_front = []
        for i in current_front:
            for j in dominated_solutions[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:  # If j is no longer dominated by any remaining solutions
                    ranks[j] = rank + 1
                    next_front.append(j)
        rank += 1
        current_front = next_front
        if current_front:
            fronts.append(current_front)

    popOriginal = np.array(popOriginal)
    
    return np.hstack((popOriginal.reshape(popOriginal.shape[0],1), ranks.reshape(ranks.shape[0],1)))


'''Reparo'''
def Reparo(instancia):
    
    resposta = instancia.CheckRestr(imprimir=0)
    for quant,nut in enumerate(resposta[1]):
        InfoVetorInstancia = np.minimum(np.sum(instancia.vetorAlimentos,axis=1),1)[:, np.newaxis]*Informacoes[:,:-2] # Pega o vetorAlimentos e cria um vetor com informações nutricionais apenas das coisas que tem
        melhorAlimento = np.argmax(InfoVetorInstancia[:,nut[0]]) #Pega o melhor alimento pra se colocar na lista baseado no que falta 
        #print(f'ADICIONANDO {np.ceil(resposta[2][quant]/InfoVetorInstancia[melhorAlimento, nut[0]])} de {ListaDeAlimentos[melhorAlimento].Nome}')


        instancia.vetorAlimentos[melhorAlimento, nut[1]] += np.ceil(resposta[2][quant]/InfoVetorInstancia[melhorAlimento, nut[0]]) # Calcula quantos disso adicionar baseado no que falta
    instancia.SomaDias()
    return instancia

'''
def Reparo2(instancia):
    
    resposta = instancia.CheckRestr(imprimir=0)
    for quant,nut in enumerate(resposta[1]):
        InfoVetorInstancia = Informacoes[:,:-2] # Pega o vetorAlimentos e cria um vetor com informações nutricionais apenas das coisas que tem
        melhoresAlimento = np.argsort(InfoVetorInstancia[:,nut[0]])[-50:] #Pega o melhor alimento pra se colocar na lista baseado no que falta
        melhorAlimento = np.random.choice(melhoresAlimento) 
        #print(f'ADICIONANDO {np.ceil(resposta[2][quant]/InfoVetorInstancia[melhorAlimento, nut[0]])} de {ListaDeAlimentos[melhorAlimento].Nome}')

        
        instancia.vetorAlimentos[melhorAlimento, nut[1]] += np.ceil(resposta[2][quant]/InfoVetorInstancia[melhorAlimento, nut[0]]) # Calcula quantos disso adicionar baseado no que falta
    instancia.SomaDias()
    return instancia
'''

def geraPopInit(n):
    '''
    Gera a população inicial do algoritmo

    Args:
        n: Tamanho da população
    Returns:
        Array de tamanho n com n objetos do tipo instância

    '''
    PopInit = []
    for i in range(n):
        #instPop = Instancia(np.array([[random.choices(values, weights=weights, k=1)[0] for _ in range(7)] for _ in range(580)]))
        instPop = Instancia(np.random.choice(values, size=(580, 7), p=weights))
        instPop.SomaDias()
        if instPop.CheckRestr()[0] == 1:
            instPop = Reparo(instPop)

        PopInit.append(instPop)
    
    return PopInit

def calc_crowding_distance(F, **kwargs):
    n_points, n_obj = F.shape

    # sort each column and get index
    I = np.argsort(F, axis=0, kind='mergesort')

    # sort the objective space values for the whole matrix
    F = F[I, np.arange(n_obj)]

    # calculate the distance from each point to the last and next
    dist = np.vstack([F, np.full(n_obj, np.inf)]) - np.vstack([np.full(n_obj, -np.inf), F])

    # calculate the norm for each objective - set to NaN if all values are equal
    norm = np.max(F, axis=0) - np.min(F, axis=0)
    norm[norm == 0] = np.nan

    # prepare the distance to last and next vectors
    dist_to_last, dist_to_next = dist, np.copy(dist)
    dist_to_last, dist_to_next = dist_to_last[:-1] / norm, dist_to_next[1:] / norm

    # if we divide by zero because all values in one columns are equal replace by none
    dist_to_last[np.isnan(dist_to_last)] = 0.0
    dist_to_next[np.isnan(dist_to_next)] = 0.0

    # sum up the distance to next and last and norm by objectives - also reorder from sorted list
    J = np.argsort(I, axis=0)
    cd = np.sum(dist_to_last[J, np.arange(n_obj)] + dist_to_next[J, np.arange(n_obj)], axis=1) / n_obj

    return cd

'''
def calculaProfundidades(x):
  profList = []
  iterador = 0
  while (len(x) > 0):
    iterador += 1
    indicesFronteira =  is_pareto_efficient(calculaObjPop(x), return_mask = True)
    fronteira = [x[i] for i in range(len(x)) if indicesFronteira[i] == 1]
    #x = np.delete(x, indicesFronteira, axis=0)
    x = [x[i] for i in range(len(x)) if indicesFronteira[i] == 0]

    profList.extend([[element,iterador] for element in fronteira])
  return np.array(profList)
'''
def calculaMedidasNSGA(CrowdPop):
    '''
    Calcula as medidas utilizadas no algoritmo NSGA-II para escolha da população

    Args:
        CrowdPop: A população ao qual o algoritmo será computado
    Returns:
        Retorna um array do tamanho da população e tres colunas, a primeira com a instância, a segunda com o rank dela, e a terceira com o crowding distance
    '''

    for alimento in CrowdPop: #A factibilidade da instancia é calculada aqui e armazenada no proprio objeto. Ela é utilizada na escolha do torneio binário
        alimento.calculaFact()

    CrowdObj = calculaObjPop(CrowdPop) #Calcula as funções objetivos da população
    rank = fast_non_dominated_sort(CrowdPop,CrowdObj) #Faz i
    rank[:,1] += 1
    tuple_array = np.array([(pop, rank, obj) for pop, rank, obj in zip(CrowdPop, rank[:,1], CrowdObj)], dtype=object)

    CrowdArray = []
    for i in range(max(rank[:,1])):
        x = calc_crowding_distance(np.vstack((tuple_array[tuple_array[:,1] == i+1][:,2])))
        CrowdArray.append(x)

    CrowdArray = np.hstack(CrowdArray)
    final = np.hstack((rank, CrowdArray.reshape(CrowdArray.shape[0],1)))
    return final


def escolhedor(ind1, ind2):

    if ind1[0].Infactivel < ind2[0].Infactivel:
        return ind1
    elif ind1[0].Infactivel > ind2[0].Infactivel:
        return ind2
    if (ind1[0].Infactivel + ind2[0].Infactivel != 0):
        return ind1 if ind1[0].Penalidade < ind2[0].Penalidade else ind2



    if ind1[1] != ind2[1]:
        return ind1 if ind1[1] < ind2[1] else ind2
    return ind1 if ind1[2] > ind2[2] else ind2

def NSGAtop(pop, k):
    '''Escolhe as k melhores instâncias baseado nas métricas do NSGAII 
    
    Args:
        pop: População de indivíduos
        k: Quantidade de indivíduos selecionados
    
    Returns:
        Array de tamanho k que corresponde a população de indivíduos selecionados
    '''
    sorted_arr = pop[np.lexsort((-pop[:, 2], pop[:, 1]))][:k,:] #Ordena por rank de menor a maior, e se igual por crowding distance, do maior ao menor. Escolhemos então os k primeiros
    return sorted_arr




def binary_tournament_selectionNSGA(k, pop):
    selected_individuals = []

    for _ in range(k):
        # Randomly select two individuals
        individual1, individual2 = random.sample(range(k), 2)
        
        # Compare their fitness values
        winner = escolhedor(pop[individual1], pop[individual2])
        
        # Add the winner to the selected individuals
        selected_individuals.append(winner[0]) #Retorna array de individuos
    return selected_individuals



def column_wise_crossover(parent1, parent2): #Faz crossover por coluna
    # Ensure parents have the same shape
    if len(parent1) != len(parent2) or len(parent1[0]) != len(parent2[0]):
        raise ValueError("Parents must have the same shape")
    
    # Transpose the parents to work with columns
    parent1_columns = list(zip(*parent1))  # Convert rows to columns
    parent2_columns = list(zip(*parent2))  # Convert rows to columns
    
    # Create empty offspring columns
    offspring1_columns = []
    offspring2_columns = []
    
    # Perform crossover column by column
    for col1, col2 in zip(parent1_columns, parent2_columns):
        if random.random() < 0.5:  # 50% chance to swap columns
            offspring1_columns.append(col2)
            offspring2_columns.append(col1)
        else:
            offspring1_columns.append(col1)
            offspring2_columns.append(col2)
    
    # Transpose back to convert columns to rows
    offspring1 = list(zip(*offspring1_columns))
    offspring2 = list(zip(*offspring2_columns))
    
    # Convert tuples back to lists (optional, for consistency)
    offspring1 = [list(row) for row in offspring1]
    offspring2 = [list(row) for row in offspring2]
    
    return offspring1, offspring2

def multi_point_column_crossover(parent1, parent2, num_crossover_points=20):
    # Ensure the parents have the same shape
    assert parent1.shape == parent2.shape, "Parents must have the same shape"
    
    # Create an empty array to hold the offspring
    offspring = np.empty_like(parent1)
    
    # Perform column-wise multi-point crossover
    for j in range(parent1.shape[1]):  # Iterate over each column
        # Generate sorted crossover points for this column
        crossover_points = sorted(np.random.choice(range(1, parent1.shape[0]), num_crossover_points, replace=False))
        
        # Alternate between parents for each segment
        start = 0
        for i, crossover_point in enumerate(crossover_points):
            if i % 2 == 0:
                offspring[start:crossover_point, j] = parent1[start:crossover_point, j]
            else:
                offspring[start:crossover_point, j] = parent2[start:crossover_point, j]
            start = crossover_point
        
        # Handle the last segment
        if len(crossover_points) % 2 == 0:
            offspring[start:, j] = parent1[start:, j]
        else:
            offspring[start:, j] = parent2[start:, j]
    
    return offspring




def mutate_2d_array(array, mutation_rate=0.01, prob_1 = 0.1): #Mutação com probabilidade de flipar 1 reduzida
    # Iterate through each element in the 2D array
    for i in range(len(array)):
        for j in range(len(array[i])):
            # Flip the bit with probability equal to the mutation rate
            if random.random() < mutation_rate:
                if array[i][j] == 0 and random.random() < prob_1:
                    array[i][j] += 1  # Flip 0 to 1 with probability prob_1
                elif array[i][j] >= 1:
                    array[i][j] -= 1  # Flip 1 to 0
    return array


def WDH(k, it, mut = 0.05, flip = 1/30,  plotar=True, indepth=True):
    popInit = geraPopInit(k)
    colorwheel = [] #Paleta de cores para o plot
    for c in range(it):
        colorwheel.append((0, 0.6, 1 - c/it))

    for i in range(it):
        if (i % 50 == 0 and indepth==True):
            print(f'{i}/{it}')
        popNova = []
        finals = calculaMedidasNSGA(popInit)
        vencedores = binary_tournament_selectionNSGA(k, finals)
        while len(popNova) < k:
             
             pai, mae = random.sample(vencedores,2 )
             cross1 = multi_point_column_crossover(pai.vetorAlimentos, mae.vetorAlimentos)
             cross2 = multi_point_column_crossover(mae.vetorAlimentos, pai.vetorAlimentos)
             
             filho1 = Instancia(mutate_2d_array(cross1, mutation_rate=mut, prob_1 = flip))
             filho2 = Instancia(mutate_2d_array(cross2, mutation_rate=mut, prob_1 = flip))

             '''
             filho1.SomaDias()
             filho2.SomaDias()

             if filho1.CheckRestr()[0] == 1:
                filho1 = Reparo(filho1)
             if filho2.CheckRestr()[0] == 1:
                filho2 = Reparo(filho2)
             '''   
             popNova.append(filho1)
             popNova.append(filho2)
             
        popInit = list(popNova) + list(popInit)
        popInit = maxMin(popInit, k)
        if plotar == True:
            plotMaxMin(popInit, cor=colorwheel[i-1])
            '''
        if (i % (it/10) == 0):
               for ind in popInit:
                   ind.SomaDias()
                   if ind.CheckRestr()[0] == 1:
                       ind = Reparo2(ind)

    for ind in popInit:
                   ind.SomaDias()
                   if ind.CheckRestr()[0] == 1:
                       ind = Reparo2(ind)
           '''
    return popInit




def NSGAII(k, it, mut = 0.05, flip = 1/30, plotar=True, indepth=True):

    popInit = geraPopInit(k)
    colorwheel = [] #Paleta de cores para o plot
    for c in range(it):
        colorwheel.append((0, 0.6, 1 - c/it))

    for i in range(it):
           if (i % 50 == 0 and indepth==True):
                print(f'{i}/{it}')
           popNova = []
           finals = calculaMedidasNSGA(popInit)
           vencedores = binary_tournament_selectionNSGA(k, finals)

           while len(popNova) < k:
             
             pai, mae = random.sample(vencedores,2 )
             
             cross1 = multi_point_column_crossover(pai.vetorAlimentos, mae.vetorAlimentos)
             cross2 = multi_point_column_crossover(mae.vetorAlimentos, pai.vetorAlimentos)
             
             filho1 = Instancia(mutate_2d_array(cross1, mutation_rate=mut, prob_1 = flip))
             filho2 = Instancia(mutate_2d_array(cross2, mutation_rate=mut, prob_1 = flip))
             '''
             filho1.SomaDias()
             filho2.SomaDias()

             if filho1.CheckRestr()[0] == 1:
                filho1 = Reparo(filho1)
             if filho2.CheckRestr()[0] == 1:
                filho2 = Reparo(filho2)
             '''   
             popNova.append(filho1)
             popNova.append(filho2)
              
           finalsNew = calculaMedidasNSGA(list(popNova) + list(popInit))
           popInit = NSGAtop(finalsNew,k)[:,0]


           if plotar == True:
            plotMaxMin(popInit, cor=colorwheel[i-1])

    return popInit

























'''  FUNÇÕES MULTIOBJETIVO'''


def pontosDecisao(resultAbordagem):
    return np.array([i.vetorAlimentos for i in resultAbordagem])



def hypervolume(S1, S2):
    volume1 = calculaObjPop(S1)
    volume2 = calculaObjPop(S2)
    volume = np.vstack((volume1,volume2))

    F_min = volume.min(axis=0)
    F_max = volume.max(axis=0)

    # Normalize objectives
    volume1_normalized = (volume1 - F_min) / (F_max - F_min)
    volume2_normalized = (volume2 - F_min) / (F_max - F_min)
    ref_point = np.array([1.001, 1.001,1.001])
    ind = HV(ref_point=ref_point)
    return (ind(volume1_normalized), ind(volume2_normalized))
    #print("HV", ind(volume_normalized))


def FO2(cobaia):
    arraypresenca = cobaia.CriaArrayPresenca()
    somall = 0
    for dia in range(7):
      somatot = 0
      for d in range(1, dia+1):
          somainterna = sum([(arraypresenca[g, dia - d] and arraypresenca[g, dia])*Penalidades[g] for g in range(14)])
          somatot += np.any(arraypresenca[:,dia] & arraypresenca[:,dia - d])*Penalidades[14 + d] + somainterna

      somall += somatot
    return somall

def calculaObjPop(pop): 
    
    stackado = np.array([inst.vetorAlimentos for inst in pop])
    
    combined_info = np.stack((InformacoesPreco, InformacoesProteina), axis=0)  # Shape: (2, 580, 7)
    res_combined = np.sum(stackado[:, None, :, :] * combined_info[None, :, :, :], axis=(2, 3))  # Shape: (100, 2)
    entra = np.array([FO2(i) for i in pop])
    res_combined = np.insert(res_combined, 1, entra, axis=1)

    res_combined[:,2] *= -1
        
    return res_combined


def is_pareto_efficient(costs, return_mask = True):

    is_efficient = np.arange(costs.shape[0])
    n_points = costs.shape[0]
    next_point_index = 0  # Next index in the is_efficient array to search for
    while next_point_index<len(costs):
        nondominated_point_mask = np.any(costs<costs[next_point_index], axis=1)
        nondominated_point_mask[next_point_index] = True
        is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
        costs = costs[nondominated_point_mask]
        next_point_index = np.sum(nondominated_point_mask[:next_point_index])+1
    if return_mask:
        is_efficient_mask = np.zeros(n_points, dtype = bool)
        is_efficient_mask[is_efficient] = True
        return is_efficient_mask
    else:
        return is_efficient
    


def strengthMat(pontos, imprimir = True):

  mat_1 = dom.calc_domination_matrix(calculaObjPop(pontos))
  mat_1[mat_1 < 0] = 0
  return mat_1

def c_distHam(arr):


    binars = ([np.sum(dieta.vetorAlimentos, axis=1) for dieta in arr])
    distance_matrix = squareform(pdist(binars, metric="hamming") * 580)

    arrFin = (distance_matrix.astype(int))

    return arrFin


def DWUcdistMat(pontos, strengths, profundidade = False):
  MatX = np.zeros(np.shape(pontos)[0]) #Cria array vazio
  strengthsSum = np.sum(strengths, axis=1) #Cria uma soma das forças para usar em DominaX

  MatX = np.sum(strengths * strengthsSum[:, np.newaxis], axis=0) #Cria uma matriz de d(x)

  dxMat =  np.abs(np.subtract.outer(MatX,MatX)) + 1 #Calcula a parte de baixo da fórmula de wd(x) de cada ponto
  return (c_distHam(pontos))/dxMat #retorna uma matriz dos wd(x) de cada par de pontos

def DWUcdistMemo(p1, p2, Mat, FullP):

    #Nessa função é feito apenas um "recorte" da matriz completa, que inclui apenas os pontos p1 como linhas e os pontos p2 como colunas
    #Isso é feito para não haver a necessidade de recalcular a matriz toda vez que o algoritmo é chamado para dois conjuntos de pontos diferentes

    #Convertendo os arrays numpy em listas pra achar o indíce mais rapidamente
    #List comprehension acelera substancialmente a eficiência do código
    p1_indices = np.array([FullP[x] for x in p1])
    p2_indices = np.array([FullP[x] for x in p2])
    return Mat[p1_indices[:, None], p2_indices]

def maxMin(points, scalar, profundidade = False):

  pontosInit = index_map = {value: idx for idx, value in enumerate(points)}

  Mat = DWUcdistMat(points, strengthMat(points, imprimir=False))
  distances = DWUcdistMemo(points,points,Mat, pontosInit)
  maxdist = np.argmax(distances) #Pega o indice dos pontos mais distantes entre si
  xi,xj =  np.unravel_index(maxdist, distances.shape) #Pega o valor a partir dos indices
  R = np.array([points[xi],points[xj]]) #Cria um conjunto com esses dois pontos inicialmente
  points = np.delete(points,[xi,xj],axis=0) # Retira os pontos da matriz P original

  while (np.shape(R)[0] < scalar):
   
    min = np.min(DWUcdistMemo(points, R, Mat, pontosInit), axis=1) #Calcula a menor distância entre cada ponto até um valor no conjunto R
    index = np.argmax(min) # Da matriz de valores minimos, pega o máximo
    pointToAdd = points[index] # Pega o valor correspondente do indice recuperado
    R = np.append(R,[pointToAdd],axis=0)  # Adiciona o ponto ao R
    points = np.delete(points,[index],axis=0) # Retira de P
  return R


def plotMaxMin(result, cor):
    resMaxMin = calculaObjPop(result)
    resMaxMinPar = resMaxMin[is_pareto_efficient(resMaxMin)]


    plt.xlabel("Custo")
    plt.ylabel("Proteínas")
    plt.scatter(resMaxMin[:,0],resMaxMin[:,2], color = cor)
    plt.xlabel("Custo")
    plt.ylabel("Proteínas")
    plt.scatter(resMaxMinPar[:,0],resMaxMinPar[:,2], color = cor)

def count_dominations_fast(points):
    points = np.expand_dims(points, axis=1)  # Shape (n,1,d)
    
    # Compare all points with each other using broadcasting
    dominates = (points <= points.swapaxes(0, 1)).all(axis=2) & (points < points.swapaxes(0, 1)).any(axis=2)
    
    # Sum along the second axis to count dominations per point
    domination_counts = dominates.sum(axis=1)
    return domination_counts


def pontosFronteiraPareto(pop):
    mask = is_pareto_efficient(calculaObjPop(pop))
    fronteiraParetoDecisao = pontosDecisao(pop[mask])
    fronteiraParetoObj = calculaObjPop(pop[mask])

    return fronteiraParetoDecisao, fronteiraParetoObj
        