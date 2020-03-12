#include "Humanos.h"
#include "Fontes/Uteis/RandPerc.h"
#include "Fontes/Macros/MacrosGerais.h"
#include "Fontes/Macros/MacrosSO.h"
#include "Fontes/Macros/0_INI_H.h"
#include "Fontes/Macros/MacrosHumanos.h"
#include "Fontes/Parametros.h"
#include "Fontes/Ambiente.h"

__host__ __device__
Humano::Humano() {
  t1 = 0U;
  t2 = 0U;
  t3 = 0U;
}

__host__ __device__
bool EstaMortoHumano::operator()(Humano humano) {
  return ((humano.t1 & MAH_SD) >> AH_SD) == MORTO;
}

__host__ __device__
bool LessQuadraHumano::operator()(Humano humano1, Humano humano2) {
  int q_h1 = ((humano1.t3 & MAH_Q) >> AH_Q);
  int q_h2 = ((humano2.t3 & MAH_Q) >> AH_Q);
  return q_h1 < q_h2;
}

__host__ __device__
int ToQuadraHumano::operator()(Humano humano) {
  return ((humano.t3 & MAH_Q) >> AH_Q);
}

Humanos::Humanos(Parametros *parametros, Ambiente *ambiente) {
  this->parametros = parametros;
  this->ambiente = ambiente;

  // Criação e inicialização dos agentes humanos. 
  contarHumanos();
  humanos = new Humano[nHumanos]();
  criarHumanos();
  sizeIndHumanos = ambiente->nQuadras + 1;

  t = make_counting_iterator(0);
  v1 = make_constant_iterator(1);

  toGPU();

  atualizacaoIndices();
}

Humanos::~Humanos() {
  delete[](humanos); delete(humanosDev); delete(indHumanosDev); 
}

void Humanos::atualizacaoIndices() {
  DVector<int> k2(ambiente->nQuadras, -1);
  DVector<int> v2(ambiente->nQuadras, -1);

  // Ordenação dos agentes humanos por quadra. 
  sort(
    humanosDev->begin(), humanosDev->end(), 
    LessQuadraHumano()
  );

  // Conversão dos agentes humanos para identificadores de quadra. 
  DVector<int> k1(nHumanos);
  transform(
    humanosDev->begin(), humanosDev->end(), 
    k1.begin(), ToQuadraHumano()
  );

  // São obtidas as quantidades de agentes por quadra. 
  reduce_by_key(
    k1.begin(), k1.end(), v1, k2.begin(), v2.begin()
  );

  // Para as quadras sem agentes são inseridos no vetor de índices elementos 
  // com valor zero. 
  int nQuadrasSemAgentes = count(k2.begin(), k2.end(), -1);
  if (nQuadrasSemAgentes > 0) {
    v2.resize(v2.size() - nQuadrasSemAgentes);
    k2.resize(k2.size() - nQuadrasSemAgentes);
    sort(k2.begin(), k2.end());
    DVector<int> quadrasSemAgentes(nQuadrasSemAgentes);
    set_difference(
      t, t + ambiente->nQuadras, 
      k2.begin(), k2.end(), quadrasSemAgentes.begin()
    );
    for (int&& i : quadrasSemAgentes) {
      v2.insert(v2.begin() + i, 0);
    }
  }

  // Realiza uma soma parcial para obter os índices para as quadras. 
  inclusive_scan(
    v2.begin(), v2.end(), indHumanosDev->begin() + 1
  );
}

int Humanos::getMemoriaGPU() {
  int totMem = 0;
  totMem += (nHumanos * sizeof(Humano));
  totMem += (sizeIndHumanos * sizeof(int));
  return totMem;
}

void Humanos::toGPU() {
  humanosDev = new DVector<Humano>(humanos, humanos + nHumanos);
  indHumanosDev = new DVector<int>(sizeIndHumanos, 0);

  PhumanosDev = raw_pointer_cast(humanosDev->data());
  PindHumanosDev = raw_pointer_cast(indHumanosDev->data());
}

void Humanos::inicializarHumano(int id, int sd, int x, int y, int l, 
                                int q, int s, int fe) {
  SET_S_H(id, s);
  SET_FE_H(id, fe);
  SET_SD_H(id, sd);
  SET_C_H(id, 0);

  SET_X_H(id, x);
  SET_Q_H(id, q);

  SET_Y_H(id, y);
  SET_L_H(id, l);
}

void Humanos::inserirHumanos(int quantidade, int estado, int sexo, 
                             int idade, int& i) {
  int p, x, y, l, q, posicoesLote;
  RandPerc rand;

  for (int j = 0; j < quantidade; ++j) {
    // Escolhe aleatoriamente uma quadra e lote iniciais para o agente. 
    q = (int)(rand() * ambiente->nQuadras);
    l = (int)(rand() * ambiente->nLotes[q]);   
    
    // Inicialmente os agentes não podem estar posicionados nas ruas. 
    if (q == RUA) {
      j--;
      continue;
    }

    // Escolhe aleatoriamente uma posição inicial para o agente. 
    posicoesLote = (ambiente->indPos[ambiente->indQuadras[2 * q] + l + 1] - 
                    ambiente->indPos[ambiente->indQuadras[2 * q] + l]);
    p = ENTRE_FAIXA(0, posicoesLote, rand());
    x = ambiente->pos[ambiente->indPos[ambiente->indQuadras[2 * q] + l] + p].x;
    y = ambiente->pos[ambiente->indPos[ambiente->indQuadras[2 * q] + l] + p].y;

    // Inicializa o novo agente. 
    inicializarHumano(i, estado, x, y, l, q, sexo, idade);
    i += 1;
  }
}

void Humanos::criarHumanos() {
  int i = 0;
  int desl = DESL_0_INI_H;
  for (int sexo = MASCULINO; sexo <= FEMININO; ++sexo) {
    for (int idade = BEBE; idade <= IDOSO; ++idade) {
      for (int estado : {SUSCETIVEL, EXPOSTO, INFECTANTE, RECUPERADO}) {
        // Percorre todos os parâmetros de inserção de humanos em 
        // "Entradas/MonteCarlo_{1}/Humanos/0-INI.csv". 
        inserirHumanos(parametros->parametros[desl], estado, sexo,
                        idade, i);
        // É somado dois pois cada parâmetro tem um valor mínimo e máximo. 
        desl += 2;
      }
    }
  }
}

void Humanos::contarHumanos() {
  nHumanos = 0;
  int desl = DESL_0_INI_H;
  for (int sexo = MASCULINO; sexo <= FEMININO; ++sexo) {
    for (int idade = BEBE; idade <= IDOSO; ++idade) {
      for (int estado : {SUSCETIVEL, EXPOSTO, INFECTANTE, RECUPERADO}) {
        nHumanos += parametros->parametros[desl];
        desl += 2;
      }
    }
  }
}
