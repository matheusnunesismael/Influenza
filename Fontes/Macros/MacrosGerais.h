#ifndef __MACROS__
#define __MACROS__

//----------------------------------------------------------------------------//
// Quantidade de colunas das saidas de quantidades dos agentes
// N_COLS_H = (N_SEXOS * N_IDADES * N_ESTADOS_H)
#define N_COLS_H    72
//----------------------------------------------------------------------------//
// Linearizacao de matriz
#define VEC(i, j, nc) (((i) * (nc)) + (j))
//----------------------------------------------------------------------------//
// Lotes pertencentes à quadra com id = 0 são as ruas
#define RUA 0
//----------------------------------------------------------------------------//
// Distancia euclidiana
#define DIST(x1, y1, x2, y2) \
(double)(sqrt(pow((x1) - (x2), 2.0) + pow((y1) - (y2), 2.0)))
//----------------------------------------------------------------------------//
// Estados da Influenza para humanos (SD)
#define N_ESTADOS_H   6

#define VIVO  1
#define MORTO 0

#define SUSCETIVEL 1
#define EXPOSTO    2
#define INFECTANTE 3
#define QUARENTENA 4
#define IMUNIZADO  5
#define RECUPERADO 6
//----------------------------------------------------------------------------//
// Sexos para humanos (S)
#define N_SEXOS 2

#define MASCULINO 0
#define FEMININO  1
//----------------------------------------------------------------------------//
// Faixas etarias dos humanos (FE)
#define N_IDADES 6

#define BEBE        0
#define CRIANCA     1
#define ADOLESCENTE 2
#define JOVEM       3
#define ADULTO      4
#define IDOSO       5
//----------------------------------------------------------------------------//
#endif
