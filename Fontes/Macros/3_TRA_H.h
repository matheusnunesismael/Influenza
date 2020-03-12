#ifndef __3_TRA_H__
#define __3_TRA_H__

// Macros para acesso aos parâmetros armazenados no arquivo 
// "Entradas/MonteCarlo_{1}/Humanos/3-TRA.csv": 

// PERIODO_EXPOSTO_HUMANO:    Parâmetros "TRA001" a "TRA006". 
// PERIODO_INFECTADO_HUMANO:  Parâmetros "TRA007" a "TRA012". 
// PERIODO_IMUNIZADO_HUMANO:  Parâmetros "TRA013" a "TRA018". 
// PERIODO_QUARENTENA_HUMANO: Parâmetros "TRA019" a "TRA024". 
// PERIODO_RECUPERADO_HUMANO: Parâmetros "TRA025" a "TRA030". 
// TAXA_EFICACIA_VACINA:      Parâmetro  "TRA031". 

#include "MacrosParametros.h"

#define PERIODO_EXPOSTO_HUMANO(ie) \
(int) (ENTRE_FAIXA( \
parametros[DESL_3_TRA_H + 0 + ie * 2], \
parametros[DESL_3_TRA_H + 1 + ie * 2], \
(randPerc)))
#define PERIODO_INFECTADO_HUMANO(ie) \
(int) (ENTRE_FAIXA( \
parametros[DESL_3_TRA_H + 12 + ie * 2], \
parametros[DESL_3_TRA_H + 13 + ie * 2], \
(randPerc)))
#define PERIODO_IMUNIZADO_HUMANO(ie) \
(int) (ENTRE_FAIXA( \
parametros[DESL_3_TRA_H + 24 + ie * 2], \
parametros[DESL_3_TRA_H + 25 + ie * 2], \
(randPerc)))
#define PERIODO_QUARENTENA_HUMANO(ie) \
(int) (ENTRE_FAIXA( \
parametros[DESL_3_TRA_H + 36 + ie * 2], \
parametros[DESL_3_TRA_H + 37 + ie * 2], \
(randPerc)))
#define PERIODO_RECUPERADO_HUMANO(ie) \
(int) (ENTRE_FAIXA( \
parametros[DESL_3_TRA_H + 48 + ie * 2], \
parametros[DESL_3_TRA_H + 49 + ie * 2], \
(randPerc)))
#define TAXA_EFICACIA_VACINA \
(double) (ENTRE_FAIXA( \
parametros[DESL_3_TRA_H + 60], \
parametros[DESL_3_TRA_H + 61], \
(randPerc)))

#endif
