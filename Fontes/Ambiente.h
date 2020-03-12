#ifndef __AMBIENTE__
#define __AMBIENTE__

#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <tuple>

#include <thrust/device_vector.h>

using std::cerr;
using std::endl;
using std::fstream;
using std::ifstream;
using std::numeric_limits;
using std::string;
using std::streamsize;

template <class T>
using DVector = thrust::device_vector<T>;

using thrust::raw_pointer_cast;

/*
  Estrutura utilizada para armazenar uma posição do ambiente. 
*/
struct Posicao {

  int x, y, lote, quadra;

};

/*
  Estrutura utilizada para armazenar uma vizinhança de Moore do ambiente. 
*/
struct Vizinhanca {

  int xOrigem, yOrigem, xDestino, yDestino, loteDestino, quadraDestino;

};

/*
  Estrutura utilizada para armazenar um registro lido do arquivo 
  "Ambiente/DistribuicaoHumanos.csv", em que:

  "q": quadra do humano; 
  "l": lote do humano; 
  "x": latitude do humano; 
  "y": longitude do humano; 
  "s": sexo do humano; 
  "fe": faixa etária do humano; 
  "sd": saude do humano; 
  "st": sorotipo atual do humano; 
  "cic": ciclo de entrada do humano na simulação; 
*/
struct Caso {

  int q, l, x, y, s, fe, sd, st, cic;

};

/*
  Classe que armazena todos os dados relacionados ao ambiente de simulação. 
*/
class Ambiente {

  public:

  string entradaMC; streamsize sMax = numeric_limits<streamsize>::max();
  fstream arquivo;

  // Dados em CPU. 
  int nQuadras, *nLotes, sizeNLotes, *indQuadras, sizeIndQuadras;
  int *indViz, sizeIndViz, sizeViz; Vizinhanca *viz;
  int *indPos, sizeIndPos, sizePos; Posicao *pos;
  int sizeFEVac, *fEVac, sizePerVac, *perVac, sizeCicVac, *cicVac;
  int sizeDistHumanos; Caso *distHumanos;
  int sizeComp; double *comp; int sizeQuaren; double *quaren;

  // Dados em GPU. 
  DVector<int> *nLotesDev, *indQuadrasDev, *indVizDev; 
  DVector<Vizinhanca> *vizDev;
  DVector<int> *indPosDev; DVector<Posicao> *posDev;
  DVector<int> *fEVacDev, *perVacDev, *cicVacDev;
  DVector<Caso> *distHumanosDev; DVector<double> *compDev;
  DVector<double> *quarenDev;

  // Ponteiros em CPU para os dados em GPU. 
  int *PnLotesDev, *PindQuadrasDev, *PindVizDev; Posicao *PposDev;
  int *PindPosDev, *PfEVacDev, *PperVacDev, *PcicVacDev; Vizinhanca *PvizDev;
  Caso *PdistHumanosDev; double *PcompDev; double *PquarenDev;

  Ambiente(string entradaMC);
  int getMemoriaGPU();
  ~Ambiente();

  private:

  void toGPU();
  void lerVetoresAmbientais();  
  int *lerVetor(int n);
  void lerQuadrasLotes();
  void lerVizinhancas();
  void lerPosicoes();
  std::tuple<int, int *> lerControle();
  void lerVetoresControles();
  void lerArquivoDistribuicaoHumanos();

};

#endif
