#include "Transicao.h"
#include "Fontes/Ambiente.h"
#include "Fontes/Parametros.h"
#include "Fontes/Seeds.h"
#include "Fontes/Humanos/Humanos.h"
#include "Fontes/Macros/MacrosHumanos.h"
#include "Fontes/Macros/3_TRA_H.h"
#include "Fontes/Macros/MacrosGerais.h"

TransicaoEstadosHumanos::TransicaoEstadosHumanos(Humanos *humanos, 
                                                 Parametros *parametros, 
                                                 Seeds *seeds) {
  this->humanos = humanos->PhumanosDev;
  this->parametros = parametros->PparametrosDev;
  this->seeds = seeds->PseedsDev;
}

__host__ __device__
void TransicaoEstadosHumanos::operator()(int id) {
  int idHumano = id;
  dre& seed = seeds[id];
  urd<double> dist(0.0, 1.0);

  int c  = GET_C_H(idHumano);  int sd = GET_SD_H(idHumano);
  int fe = GET_FE_H(idHumano); 

  if (sd == MORTO) return;

  switch (sd) {
    // Se o período de exposição do agente terminou, 
    // ele é passado ao estado infectante. 
    case EXPOSTO: {
      if (c >= PERIODO_EXPOSTO_HUMANO(fe)) {
        SET_SD_H(idHumano, INFECTANTE);
        SET_C_H(idHumano, 0);
      } else {
        SET_C_H(idHumano, c + 1);
      }
    } break;
    // Se o período de infectância do agente terminou, 
    // ele é passado ao estado recuperado. 
    case INFECTANTE: {
      if (c >= PERIODO_INFECTADO_HUMANO(fe)) {
        SET_SD_H(idHumano, RECUPERADO);
        SET_C_H(idHumano, 0);
      } else {
        SET_C_H(idHumano, c + 1);
      }
    } break;
    // Se o período de quarentena do agente terminou, 
    // ele é passado ao estado recuperado. 
    case QUARENTENA: {
      if (c >= PERIODO_QUARENTENA_HUMANO(fe)) {
        SET_SD_H(idHumano, RECUPERADO);
        SET_C_H(idHumano, 0);
      } else {
        SET_C_H(idHumano, c + 1);
      }
    } break;
    // Se o período de imunização do agente terminou, 
    // ele é passado ao estado suscetível. 
    case IMUNIZADO: {
      if (c >= PERIODO_IMUNIZADO_HUMANO(fe)) {
        SET_SD_H(idHumano, SUSCETIVEL);
        SET_C_H(idHumano, 0);
      } else {
        SET_C_H(idHumano, c + 1);
      }
    } break;
    // Se o período de recuperação do agente terminou, 
    // ele é passado ao estado suscetível. 
    case RECUPERADO: {
      if (c >= PERIODO_RECUPERADO_HUMANO(fe)) {
        SET_SD_H(idHumano, SUSCETIVEL);
        SET_C_H(idHumano, 0);
      } else {
        SET_C_H(idHumano, c + 1);
      }
    } break;
  }
}

Vacinacao::Vacinacao(Humanos *humanos, Ambiente *ambiente, 
                     Parametros *parametros, int ciclo, 
                     int sizeFEVac, int sizePerVac, int sizeCicVac, 
                     Seeds *seeds) {
  this->humanos = humanos->PhumanosDev;
  this->indHumanos = humanos->PindHumanosDev;
  this->parametros = parametros->PparametrosDev;
  this->ciclo = ciclo;
  this->sizeFEVac = sizeFEVac;
  this->sizePerVac = sizePerVac;
  this->sizeCicVac = sizeCicVac;
  this->fEVac = ambiente->PfEVacDev;
  this->perVac = ambiente->PperVacDev;
  this->cicVac = ambiente->PcicVacDev;
  this->seeds = seeds->PseedsDev;
}

__host__ __device__ 
void Vacinacao::operator()(int id) {
  if (not periodoVacinacao()) return;

  dre& seed = seeds[id];
  urd<double> dist(0.0, 1.0);

  // A vacinação é aplicada em todas as quadras do ambiente. 
  int qVac = id;

  int fe_h, sd_h;
  int n[N_IDADES] = {0, 0, 0, 0};

  // Contagem, por faixas etárias, dos agentes presentes nesta quadra. 
  for (int idHumano = indHumanos[qVac]; 
       idHumano < indHumanos[qVac + 1]; ++idHumano) {
    fe_h = GET_FE_H(idHumano); sd_h = GET_SD_H(idHumano);

    if (sd_h == SUSCETIVEL) {
      n[fe_h]++;
    }
  }

  // Calcula as quantidades por faixas etárias de agentes que serão vacinados. 
  double percentualVacinacao = 1.0 / perVac[0];
  for (int fe = 0; fe < N_IDADES; ++fe) {
    if (faixaEtariaTeraVacinacao(fe)) {
      n[fe] = lround(n[fe] * percentualVacinacao);
    } else {
      n[fe] = 0;
    }
  }

  // Realiza a vacinação para a quantidade de agentes calculada. 
  for (int idHumano = indHumanos[qVac]; 
       idHumano < indHumanos[qVac + 1]; ++idHumano) {
    fe_h = GET_FE_H(idHumano);
    sd_h = GET_SD_H(idHumano);

    if (sd_h == SUSCETIVEL and n[fe_h] > 0) {
      n[fe_h]--;

      // O agente pode ser passado probabilisticamente para o estado imunizado. 
      if (randPerc <= TAXA_EFICACIA_VACINA) {
        SET_SD_H(idHumano, IMUNIZADO);
      }
    }
  }
}

__host__ __device__
bool Vacinacao::periodoVacinacao() {
  // Verifica se há um período de vacinação ativo. 
  if (perVac[1] < perVac[0]) {
    for (int i = 0; i < sizeCicVac; ++i) {
      if (ciclo >= cicVac[i] and ciclo < (cicVac[i] + perVac[0])) {
        return true;
      }
    }
  }
  return false;
}

__host__ __device__
bool Vacinacao::faixaEtariaTeraVacinacao(int fe) {
  // Verifica se a faixa etária passada como argumento receberá vacinação. 
  for (int i = 0; i < sizeFEVac; ++i) {
    if (fEVac[i] == fe) {
      return true;
    }
  }
  return false;
}

PosVacinacao::PosVacinacao(Ambiente *ambiente, int ciclo, 
                           int sizePerVac, int sizeCicVac) {
  this->ciclo = ciclo;
  this->sizePerVac = sizePerVac;
  this->sizeCicVac = sizeCicVac;
  this->perVac = ambiente->PperVacDev;
  this->cicVac = ambiente->PcicVacDev;
}

__host__ __device__ 
void PosVacinacao::operator()(int id) {
  bool houveVacinacao = false;
  if (perVac[1] < perVac[0]) {
    for (int i = 0; i < sizeCicVac; ++i) {
      if (ciclo >= cicVac[i] and ciclo < (cicVac[i] + perVac[0])) {
        houveVacinacao = true;
        break;
      }
    }
  }
  // Se houve vacinação a campanha avança para o próximo dia. 
  if (houveVacinacao) perVac[1]++;
  else perVac[1] = 0;
}

Quarentena::Quarentena(Humanos *humanos, Ambiente *ambiente, int ciclo, 
                       Seeds *seeds) {
  this->humanos = humanos->PhumanosDev;
  this->indHumanos = humanos->PindHumanosDev;
  this->ciclo = ciclo;
  this->quaren = ambiente->PquarenDev;
  this->seeds = seeds->PseedsDev;
}

__host__ __device__
void Quarentena::operator()(int id) {
  dre& seed = seeds[id];
  urd<double> dist(0.0, 1.0);

  int sd_h = GET_SD_H(id);

  // Os agentes infectantes podem ser passados probabilisticamente ao estado 
  // de quarentena. As probabilidades de passagem ao estado de quarentena são 
  // definidas pela variável "quaren" e são distintas para cada ciclo. 
  if (sd_h == INFECTANTE and randPerc <= quaren[ciclo]) {
    SET_SD_H(id, QUARENTENA);
  }
}
