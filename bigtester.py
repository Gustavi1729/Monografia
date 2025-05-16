from pymoo.indicators.hv import HV
from FuncoesGenetico import *
from multiprocessing import Process, Queue
import threading

def worker(k, it,mut1, mut2, flip1, flip2, q):
    print("DOIS PONTOS PARENTESES")
    S2 = NSGAII(k,it, mut2, flip2, plotar=False, indepth=False)
    S1 = iteradorTest(k,it, mut1, flip1, plotar=False, indepth=False)

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

def TheBigTester2(algos,k,it, mut1, mut2, flip1, flip2):

    print("Nova Versão 3.3.1")
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

    finalNP = np.array(final)
    for c in range(6):
        print(np.mean(finalNP[:,c]))

    return finalNP




def TheBigTester(algos,k,it):

    print("Nova Versão 2.3")
    meuItHV = []
    nsgaHV = []
    meuItUNI = []
    nsgaUNI = []
    meuItUNIMean = []
    nsgaUNIMean = []
    DadosBrutos1 = []
    DadosBrutos2 = []
    for i in range(algos):
        if i % 5 == 0:
            print(f'{i}/{algos} CICLOS COMPLETOS')
        S1 = iterador(k,it, plotar=False, indepth=False)
        S2 = iteradorTest(k,it, plotar=False, indepth=False)

        pack = hypervolume(S1,S2)
        meuItHV.append(pack[0])
        nsgaHV.append(pack[1])

        arrayUniformidade1 = c_distHam(S1)
        meuItUNI.append(np.min(arrayUniformidade1[arrayUniformidade1 > 0]))
        meuItUNIMean.append((np.mean(arrayUniformidade1[arrayUniformidade1 > 0])))
        arrayUniformidade2 = c_distHam(S2)
        nsgaUNI.append(np.min(arrayUniformidade2[arrayUniformidade2 > 0]))
        nsgaUNIMean.append((np.mean(arrayUniformidade2[arrayUniformidade2 > 0])))
        DadosBrutos1.append(S1)
        DadosBrutos2.append(S2)


    print(np.mean(meuItHV))
    print(np.mean(nsgaHV))
    print(np.mean(meuItUNI))
    print(np.mean(nsgaUNI))
    print(np.mean(meuItUNIMean))
    print(np.mean(nsgaUNIMean))
    return DadosBrutos1, DadosBrutos2


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