#include "Movimentacao.h"
#include "Fontes/Ambiente.h"
#include "Fontes/Parametros.h"
#include "Fontes/Seeds.h"
#include "Fontes/Humanos/Humanos.h"
#include "Fontes/Macros/MacrosHumanos.h"
#include "Fontes/Macros/1_MOV_H.h"
#include "Fontes/Macros/MacrosGerais.h"

MovimentacaoHumanos::MovimentacaoHumanos(Humanos *humanos, Ambiente *ambiente, 
                                         Parametros *parametros, 
                                         Seeds *seeds) {
  this->humanos = humanos->PhumanosDev;
  this->parametros = parametros->PparametrosDev;
  this->indQuadras = ambiente->PindQuadrasDev;
  this->indViz = ambiente->PindVizDev;
  this->viz = ambiente->PvizDev;
  this->pos = ambiente->PposDev;
  this->sizePos = ambiente->sizePos;
  this->seeds = seeds->PseedsDev;
}

__host__ __device__
void MovimentacaoHumanos::operator()(int id) {
  dre& seed = seeds[id];
  urd<double> dist(0.0, 1.0);

  int sd_h = GET_SD_H(id), fe_h = GET_FE_H(id);
  int p;

  // Humanos infectantes podem realizar uma movimentação para uma posição 
  // aleatória do ambiente, mimetizando Redes de Mundo Pequeno. 
  if (sd_h == INFECTANTE and randPerc <= PERC_MIGRACAO) {
    p = randPerc * sizePos;
    SET_X_H(id, pos[p].x);
    SET_Y_H(id, pos[p].y);
    SET_L_H(id, pos[p].lote);
    SET_Q_H(id, pos[p].quadra);
  }

  // A movimentação do agente é condicionada à taxa de mobilidade para sua 
  // faixa etária. 
  if (sd_h != MORTO and randPerc <= TAXA_MOBILIDADE(fe_h)) {
    if (sd_h == QUARENTENA) { 
      // Humanos em quarentena realizam movimentação local. 
      movimentacaoLocal(id, seed, dist);
    } else {
      // Demais agentes realizam movimentação aleatória. 
      movimentacaoAleatoria(id, seed, dist);
    }
  }
}

__host__ __device__
void MovimentacaoHumanos::movimentacaoLocal(int id, dre& seed, 
                                            urd<double>& dist) {
  int q = GET_Q_H(id), l = GET_L_H(id), x = GET_X_H(id), y = GET_Y_H(id);
  int k, n;

  n = nVertVizinhos(x, y, l, q, l, q);
  if (n == 0) return;
  
  k = (int)(randPerc * n);
  k = getVertK(k, x, y, l, q, l, q);
  moveHumano(id, k);
}

__host__ __device__
void MovimentacaoHumanos::movimentacaoAleatoria(int id, dre& seed, 
                                                urd<double>& dist) {
  int q = GET_Q_H(id), l = GET_L_H(id), x = GET_X_H(id), y = GET_Y_H(id);
  int k, n;

  n = nVertVizinhos(x, y, l, q);
  if (n == 0) return;
  
  k = (int)(randPerc * n);
  k = getVertK(k, x, y, l, q);
  moveHumano(id, k);
}

__host__ __device__
int MovimentacaoHumanos::nVertVizinhos(int x, int y, int l, int q) {
  int n = 0;
  for (int i = indViz[indQuadras[2 * q] + l];
        i < indViz[indQuadras[2 * q] + l + 1]; i++) {
    if (viz[i].xOrigem == x and viz[i].yOrigem == y) n++;
  }
  return n;
}

__host__ __device__
int MovimentacaoHumanos::nVertVizinhos(int x, int y, int l, int q, 
                                       int ld, int qd) {
  int n = 0;
  for (int i = indViz[indQuadras[2 * q] + l];
        i < indViz[indQuadras[2 * q] + l + 1]; i++) {
    if (viz[i].xOrigem == x and viz[i].yOrigem == y and
        viz[i].loteDestino == ld and viz[i].quadraDestino == qd) n++;
  }
  return n;
}

__host__ __device__
int MovimentacaoHumanos::getVertK(int k, int x, int y, int l, int q) {
  int j = 0;
  for (int i = indViz[indQuadras[2 * q] + l];
        i < indViz[indQuadras[2 * q] + l + 1]; i++) {
    if (viz[i].xOrigem == x and viz[i].yOrigem == y) {
      if (j == k) return i;
      j++;
    }
  }
  return -1;
}

__host__ __device__
int MovimentacaoHumanos::getVertK(int k, int x, int y, int l, int q, 
                                  int ld, int qd) {
  int j = 0;
  for (int i = indViz[indQuadras[2 * q] + l];
        i < indViz[indQuadras[2 * q] + l + 1]; i++) {
    if (viz[i].xOrigem == x and viz[i].yOrigem == y and
        viz[i].loteDestino == ld and viz[i].quadraDestino == qd) {
      if (j == k) return i;
      j++;
    }
  }
  return -1;
}

__host__ __device__
void MovimentacaoHumanos::moveHumano(int id, int k) {
  if (k != -1) {
    SET_X_H(id, viz[k].xDestino);
    SET_Y_H(id, viz[k].yDestino);
    SET_L_H(id, viz[k].loteDestino);
    SET_Q_H(id, viz[k].quadraDestino);
  }
}
