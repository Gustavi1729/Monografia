from pymoo.indicators.hv import HV
from FuncoesGenetico import *
from multiprocessing import Process, Queue
import threading

def worker(k, it,mut1, mut2, flip1, flip2, q):
    S2 = NSGAII(k,it, mut2, flip2, plotar=False, indepth=False)
    S1 = WDH(k,it, mut1, flip1, plotar=False, indepth=False)

    pack = hypervolume(S1,S2)

    S1hv = pack[0]
    S2hv = pack[1]
    arrayUniformidade1 = c_distHam(S1)
    arrayUniformidade2 = c_distHam(S2)
    S1min = np.min(arrayUniformidade1[arrayUniformidade1 > 0])
    S2min = np.min(arrayUniformidade2[arrayUniformidade2 > 0])
    S1med = np.mean(arrayUniformidade1[arrayUniformidade1 > 0])
    S2med = np.mean(arrayUniformidade2[arrayUniformidade2 > 0])

    q.put([S1hv, S2hv, S1min, S2min, S1med, S2med])

def Comparador(algos,k,it, mut1, mut2, flip1, flip2):

    processos = []
    final = []

    q = Queue()
    for i in range(algos):
        if i % 5 == 0:
            print(f'{i}/{algos} CICLOS COMPLETOS')
        p = Process(target=worker, args=(k, it, mut1, mut2, flip1, flip2, q))
        processos.append(p)
        p.start()

    for p in processos:
        p.join()

    while not q.empty():
        final.append(q.get())

    labels = ["HiperVolume WDH: ", "HiperVolume NSGA-II: ", "d_min WDH: ", "d_min NSGA-II: ", "d_med WDH: ", "d_med NSGA-II: "]

    finalNP = np.array(final)
    for c in range(6):
        print(labels[c], np.mean(finalNP[:,c]))

    return finalNP



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