#include "Contato.h"
#include "Fontes/Ambiente.h"
#include "Fontes/Parametros.h"
#include "Fontes/Seeds.h"
#include "Fontes/Humanos/Humanos.h"
#include "Fontes/Macros/MacrosHumanos.h"
#include "Fontes/Macros/0_INI_H.h"
#include "Fontes/Macros/2_CON_H.h"
#include "Fontes/Macros/MacrosGerais.h"
#include "../ParametrosSim.h"

ContatoHumanos::ContatoHumanos(Humanos *humanos, Ambiente *ambiente, 
                               Parametros *parametros, int ciclo, 
                               Seeds *seeds, ParametrosSim *parametrossim) {
  this->humanos = humanos->PhumanosDev;
  this->indHumanos = humanos->PindHumanosDev;
  this->parametros = parametros->PparametrosDev;
  this->pos = ambiente->PposDev;
  this->comp = ambiente->PcompDev;
  this->ciclo = ciclo;
  this->seeds = seeds->PseedsDev;
  this->parametrossim = parametrossim;
}

__host__ __device__
void ContatoHumanos::operator()(int id) {
  dre& seed = seeds[id];
  urd<double> dist(0.0, 1.0);

  int x = pos[id].x, y = pos[id].y;
  int l = pos[id].lote, q = pos[id].quadra;

  int l_h, x_h, y_h, sd_h, fe_h;

  double taxaInfeccao;

  int nSus = 0, nInf = 0;

  // Conta quantos humanos suscetíveis e infectantes há nesta posição. 
  for (int idHumano = indHumanos[q]; idHumano < indHumanos[q + 1]; ++idHumano) {
    l_h = GET_L_H(idHumano); x_h = GET_X_H(idHumano); 
    y_h = GET_Y_H(idHumano); sd_h = GET_SD_H(idHumano);

    if (sd_h == MORTO or l_h != l or x_h != x or y_h != y) continue;

    switch (sd_h) {
      case SUSCETIVEL: nSus++; break;
      case INFECTANTE: nInf++; break;
    }
  }

  // Se há humanos suscetíveis e infectantes nesta posição, pode ocorrer, 
  // probabilisticamente, a infecção dos agentes suscetíveis. 
  if (nSus > 0 and nInf > 0) {
    for (int idHumano = indHumanos[q]; idHumano < indHumanos[q + 1]; ++idHumano) {
      l_h = GET_L_H(idHumano); x_h = GET_X_H(idHumano); 
      y_h = GET_Y_H(idHumano); sd_h = GET_SD_H(idHumano); 
      fe_h = GET_FE_H(idHumano);

      if (sd_h != SUSCETIVEL or l_h != l or x_h != x or y_h != y) continue;

      taxaInfeccao = TAXA_INFECCAO_HUMANO_SUSCETIVEL(fe_h);
      //parametrossim->taxadeinfeccao += taxaInfeccao;
      //parametrossim->numeroexpostos += 1;

      // Se o agente é infectado ele é passado ao estado exposto. 
      if (randPerc <= (taxaInfeccao * comp[ciclo] * K_COMP)) {
        SET_SD_H(idHumano, EXPOSTO);
      }
    }
  }
}
