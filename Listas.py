import numpy as np
import pandas as pd


class Alimento:
  def __init__(self, row, cat):
        for c, attr in enumerate(nutrientes):
          setattr(self, attr, row.iloc[c])
        self.categoria = cat
        self.preco = np.random.choice(ValoresPreco)

  def __repr__(self):
        return f"Alimento({', '.join(f'{k}={v!r}' for k, v in self.__dict__.items())})"
  

#Características da instância do problema
class Instancia:

  Infactivel = 0 # 0 se não for, 1 se for
  Penalidade = 0 # 





  def __init__(self, vetor = np.zeros((580,7),dtype=np.uint8)):
        self.vetorAlimentos = vetor

  def __repr__(self):
        return f"Alimento({', '.join(f'{k}={v!r}' for k, v in self.__dict__.items())})"

  def __str__(self):
    res = 'Quantidades: \n'

    for dia in range(7):
      res = res + f'\n\n{Dias[dia]}:'
      for c,i in  enumerate(self.vetorAlimentos[:,dia]):
          if i > 0:
            res = res + f"{i*100}g de {ListaDeAlimentos[c].Nome},\\\ Calorias {i*ListaDeAlimentos[c].Energia_kcal:.0f} kcal,Carboidratos: {i*ListaDeAlimentos[c].Carboidratos_g:.1f}g , Lipidios: {i*ListaDeAlimentos[c].Lipideos_g:.1f}g , Proteína: {i*ListaDeAlimentos[c].Proteina_g:.1f}g    \n"
    return res

  def SomaDias(self):
    Info7Dias = np.repeat(Informacoes[np.newaxis, :, :], 7, axis=0) #Informações são repetidas 7x pra fazer a multiplicação
    vetorTransposto = self.vetorAlimentos.T # Transpõe a matriz pra alinhar pra multiplicação
    vetorTransposto3D = vetorTransposto[:, :, np.newaxis] #Expande para as duas serem tri-dimensionais
    total = np.sum(vetorTransposto3D*Info7Dias,axis=1) #Faz a multiplicação das matrizes e soma o total de cada um dos 7 dias
    self.totalDias = total
    return total # Resultado : Matriz Dia X Nutriente

  def SomaInstancia(self):
    vetorTransposto = self.vetorAlimentos.sum(axis=1).reshape(-1, 1) # Soma as colunas (dia)
    total = np.sum(Informacoes*vetorTransposto,axis=0) #Transpõe matriz para alinhamento
    self.TotalSem = total
    return total
  
  def CriaArrayPresenca(self):
     self.arraypresenca = np.zeros((15,7), dtype=int)
     for c in range(7):
        vetor = self.vetorAlimentos[:, c]
        filtro = np.unique(InformacoesCategoria[vetor > 0])
        self.arraypresenca[filtro,c] = 1
     return self.arraypresenca

     

  def FuncaoObj1(self):
    return np.sum(InformacoesPreco*self.vetorAlimentos)

  def FuncaoObj2(self):
    return np.sum(InformacoesProteina*self.vetorAlimentos)

  def CalculaFuncaoObj(self):
     self.funcaoObj = [self.FuncaoObj1(), self.FuncaoObj2()]
     return self.funcaoObj


  def CheckRestr(self, imprimir=0):
    erros = []
    quants = []
    falhou = 0
    for dia in range(7):
      for info in range(16):
        if self.totalDias[dia,info] < VDR[info]:
          if imprimir == 1:
            print(f"Não passou na quantidade de {nutrientes[info+2]} no dia {dia+1} : {self.totalDias[dia,info]} é menor que {VDR[info]}")
          erros.append([info, dia])
          quants.append(VDR[info] - self.totalDias[dia,info])

    return (falhou, erros, quants)


  def calculaFact(self):

    vetorTransposto = self.vetorAlimentos.T # Transpõe a matriz pra alinhar pra multiplicação
    vetorTransposto3D = vetorTransposto[:, :, np.newaxis] #Expande para as duas serem tri-dimensionais
    total = np.sum(vetorTransposto3D*Info7Dias,axis=1) #Faz a multiplicação das matrizes e soma o total de cada um dos 7 dias
    quants = []
    for dia in range(7):
      for info in range(16):
        if total[dia,info] < VDR[info]:
          #print(f"Não passou na quantidade de {nutrientes[info+2]} no dia {dia+1} : {self.totalDias[dia,info]} é menor que {VDR[info]}")
          self.Infactivel = 1
          #erros.append([info, dia])
          quants.append(VDR[info] - total[dia,info])
    self.Penalidade = np.sum(quants)



'''
  def CheckRestr(self, imprimir=0):
    erros = []
    quants = []
    falhou = 0
    for dia in range(7):
      for info in range(16):
        if self.totalDias[dia,info] < VDR[info]:
          if imprimir == 1:
            print(f"Não passou na quantidade de {nutrientes[info+2]} no dia {dia+1} : {self.totalDias[dia,info]} é menor que {VDR[info]}")
          falhou = 1
          erros.append([info, dia])
          quants.append(VDR[info] - self.totalDias[dia,info])
    return (falhou, erros, quants)
'''





values = [0, 1, 2]
weights = [0.95, 0.04, 0.01]

nutrientes = ["ID",
"Nome",
'Energia_kcal',
'Proteina_g',
'Lipideos_g',
'Colesterol_mg',
'Carboidratos_g',
'Fibra_Alimentar_g',
'Calcio_mg',
'Magnesio_mg',
'Manganes_mg',
'Fosforo_mg',
'Ferro_mg',
'Sodio_mg',
'Potassio_mg',
'Cobre_mg',
'Zinco_mg',
'Vitamica_C_mg',]

Categorias = [
  'Cereais', #0
  'Verduras', #1
  'Frutas', #2
  'Gorduras', #3
  'Frutos_do_Mar', #4
  'Carnes', #5
  'Leite', #6
  'Bebidas', #7
  'Ovos', #8
  'Açucarados', #9
  'Miscelanea', #10
  'Industrializados', #11
  'Preparados', #12
  'Leguminosas', #13
  'Nozes' #14

]

Penalidades = [
    0.3,
    0.1,
    0.1,
    0.1,
    1,
    3,
    0.3,
    0.1,
    0.1,
    0.1,
    0.1,
    0.1,
    0.1,
    0.3,
    0.1,

    3,
    2.5,
    1.8,
    1,
    0.2,
    0.1


]

Dias = [
    'Segunda-Feira',
    'Terça-Feira',
    'Quarta-Feira',
    'Quinta-Feira',
    'Sexta-Feira',
    'Sábado',
    'Domingo'

]

VDR = [
    2000,
    50,
    65,
    300,
    300,
    25,
    1000,
    420,
    3,
    700,
    14,
    2000,
    3500,
    0.9,
    11,
    100
]

ListaDeAlimentos = []
Informacoes = []
ValoresPreco = np.arange(0, 10.01, 0.01)
InformacoesPreco = []
InformacoesProteina = []
InformacoesCategoria = []



''' DEPOIS DE CRIADAS AS LISTAS VAZIAS, FAZEMOS OPERAÇÕES PRA LEITURA DOS DADOS E TER TUDO EM MÃOS. COMO OS DADOS SÃO DE LEITURA NÃO PRECISAREMOS FAZER ALTERAÇÕES DEPOIS    '''

file_path = 'Taco-4a-Edicao.xlsx'
df = pd.read_excel(file_path, engine='openpyxl')
colRemove = [2,4, 10, 13, 21, 22, 23, 24, 25, 26, 27] # Remoção de Colunas não utilizadaas
df_dropped = df.drop(columns=df[df.columns[colRemove]])

# Substituição de dados faltantes com 0
df = df_dropped
df = df.replace('Tr', 0)
df = df.replace(' Tr', 0)
df = df.replace(' ', 0)
df = df.fillna(0)

for index, row in df.iloc[1:64].iterrows():
  ListaDeAlimentos.append(Alimento(row,0))

for index, row in df.iloc[65:164].iterrows():
  ListaDeAlimentos.append(Alimento(row,1))

for index, row in df.iloc[165:261].iterrows():
  ListaDeAlimentos.append(Alimento(row,2))

for index, row in df.iloc[262:276].iterrows():
  ListaDeAlimentos.append(Alimento(row,3))

for index, row in df.iloc[277:327].iterrows():
  ListaDeAlimentos.append(Alimento(row,4))

for index, row in df.iloc[328: 451].iterrows():
  ListaDeAlimentos.append(Alimento(row,5))

for index, row in df.iloc[452: 476].iterrows():
  ListaDeAlimentos.append(Alimento(row,6))

for index, row in df.iloc[477: 491].iterrows():
  ListaDeAlimentos.append(Alimento(row,7))

for index, row in df.iloc[492: 499].iterrows():
  ListaDeAlimentos.append(Alimento(row,8))

for index, row in df.iloc[500: 520].iterrows():
  ListaDeAlimentos.append(Alimento(row,9))

for index, row in df.iloc[521: 530].iterrows():
 ListaDeAlimentos.append(Alimento(row,10))

for index, row in df.iloc[531: 536].iterrows():
 ListaDeAlimentos.append(Alimento(row,11))

for index, row in df.iloc[537: 569].iterrows():
 ListaDeAlimentos.append(Alimento(row,12))

for index, row in df.iloc[570: 600].iterrows():
 ListaDeAlimentos.append(Alimento(row,13))

for index, row in df.iloc[601: 612].iterrows():
 ListaDeAlimentos.append(Alimento(row,14))


 #Identificação de linhas com asteriscos
dfAsterisk = df[df.map(lambda x: x == '*').any(axis=1)].iloc[:,:1]
flattened_list = [item for sublist in dfAsterisk.values.tolist() for item in sublist]
flattened_list.pop()

new_list = [item for idx, item in enumerate(ListaDeAlimentos) if (idx+1) not in flattened_list]
ListaDeAlimentos = new_list

Informacoes = np.array([list(x.__dict__.values())[2:] for x in ListaDeAlimentos],dtype=np.float32)

InformacoesPreco = np.copy(Informacoes[:,-1].reshape(-1, 1))
InformacoesProteina = np.copy(Informacoes[:,1].reshape(-1, 1))

InformacoesCategoria = np.zeros((580), dtype=int)

Info7Dias = np.repeat(Informacoes[np.newaxis, :, :], 7, axis=0)

for c,alimento in enumerate(ListaDeAlimentos):
    InformacoesCategoria[c] = alimento.categoria 
InformacoesCategoria = np.repeat(InformacoesCategoria[:,np.newaxis], 7, axis=1)
