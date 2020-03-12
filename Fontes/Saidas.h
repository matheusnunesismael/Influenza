#ifndef __SAIDAS__
#define __SAIDAS__

#include <fstream>
#include <iostream>
#include <string>

#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>

class Ambiente;
class Parametros;

using std::cerr;
using std::endl;
using std::ofstream;
using std::string;
using std::to_string;

template <class T>
using DVector = thrust::device_vector<T>;

using thrust::copy;
using thrust::fill_n;
using thrust::raw_pointer_cast;

/*
  Classe responsável pelo armazenamento de todas as saídas geradas durante a 
  execução de simulações tipo Monte Carlo e individuais. 
*/
class Saidas {

  public:

  string saidaMC; Ambiente *ambiente; Parametros *parametros;

  // Dados armazenados em CPU. 
  int *popTH, sizePopTH, *indPopQH, sizeIndPopQH, *popQH, sizePopQH;
  int *popNovoTH, sizePopNovoTH, *popNovoQH, sizePopNovoQH;

  int *espacialH, sizeEspacialH;
  int *espacialNovoH, sizeEspacialNovoH;

  // Dados armazenados em GPU. 
  DVector<int> *popTHDev, *indPopQHDev, *popQHDev, *popNovoTHDev, *popNovoQHDev;
  DVector<int> *espacialHDev, *espacialNovoHDev;

  // Ponteiros em CPU para os dados armazenados em GPU. 
  int *PpopTHDev, *PindPopQHDev, *PpopQHDev, *PespacialHDev, *PpopNovoTHDev;
  int *PpopNovoQHDev;
  int *PespacialNovoHDev;

  Saidas(Ambiente *ambiente, Parametros *parametros, string saidaMC);
  ~Saidas();
  void salvarPopulacoes();
  void salvarEspaciais(string saidaSim);
  void toCPU();
  int getMemoriaGPU();
  void limparEspaciais();

  private:

  void toGPU();
  void salvarSaidaEspacial(int *espacial, string saidaSim, string nomeArquivo);
  void salvarPopT(int *popT, int nCols, string prefNomeArquivo);
  void salvarPopQ(int *indPopQ, int *popQ, int nCols, string prefNomeArquivo);
  void calcIndPopQ(int *indPopQ, int nCols);

};

#endif