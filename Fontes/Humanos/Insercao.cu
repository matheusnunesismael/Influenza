#include "Insercao.h"
#include "Fontes/Ambiente.h"
#include "Fontes/Parametros.h"
#include "Fontes/Humanos/Humanos.h"
#include "Fontes/Macros/MacrosHumanos.h"
#include "Fontes/Macros/MacrosGerais.h"

PreInsercaoHumanos::PreInsercaoHumanos(Parametros *parametros, int ciclo, 
                                       Ambiente *ambiente) {
  this->parametros = parametros->PparametrosDev;
  this->ciclo = ciclo;
  this->sizeDistHumanos = ambiente->sizeDistHumanos;
  this->distHumanos = ambiente->PdistHumanosDev;
}

__host__ __device__
int PreInsercaoHumanos::operator()(int id) {
  int nHumanos = 0;

  // São utilizadas as informações presentes no arquivo 
  // "Entradas/MonteCarlo_{1}/Ambiente/DistribuicaoHumanos.csv". 
  for (int i = 0; i < sizeDistHumanos; i++) {
    if (distHumanos[i].cic == ciclo) {
      nHumanos += 1;
    }
  }

  return nHumanos;
}

InsercaoHumanos::InsercaoHumanos(Humanos *humanos, Ambiente *ambiente, 
                                 Parametros *parametros, int ciclo) {
  this->humanos = humanos->PhumanosDev;
  this->indQuadras = ambiente->PindQuadrasDev;
  this->parametros = parametros->PparametrosDev;
  this->ciclo = ciclo;
  this->sizeDistHumanos = ambiente->sizeDistHumanos;
  this->distHumanos = ambiente->PdistHumanosDev;
}

__host__ __device__
void InsercaoHumanos::operator()(int id) {
  int i = 0;
  int q, l, x, y, s, fe, sd;
  
  // São utilizadas as informações presentes no arquivo 
  // "Entradas/MonteCarlo_{1}/Ambiente/DistribuicaoHumanos.csv". 
  for (int j = 0; j < sizeDistHumanos; j++) {
    if (distHumanos[j].cic == ciclo) {
      q = distHumanos[j].q; l = distHumanos[j].l; 
      x = distHumanos[j].x; y = distHumanos[j].y; 
      s = distHumanos[j].s; fe = distHumanos[j].fe; 
      sd = distHumanos[j].sd; // st = distHumanos[j].st; 

      // Inicialização do novo agente. 
      inicializarHumano(i, sd, x, y, l, q, s, fe);
      i += 1;
    }
  }
}

__host__ __device__
void InsercaoHumanos::inicializarHumano(int id, int sd, int x, int y, int l, 
                                        int q, int s, int fe) {
  SET_S_H(id, s);
  SET_FE_H(id, fe);
  SET_C_H(id, 0);
  SET_SD_H(id, sd);
  
  SET_X_H(id, x);
  SET_Q_H(id, q);

  SET_Y_H(id, y);
  SET_L_H(id, l);
}
