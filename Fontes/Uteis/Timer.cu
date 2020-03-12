#include "Timer.h"

/*
  Construtor da classe Timer. 
*/
Timer::Timer() {
  this->total = 0;
}

/*
  Método responsável por armazenar o tempo inicial da ocorrência de um evento. 
*/
void Timer::start() {
  cudaEventCreate(&begin);
  cudaEventCreate(&end);
  cudaEventRecord(begin);
}

/*
  Método responsável por armazenar o tempo final da ocorrência de um evento. 
  Com os tempos iniciais e finais é possível calcular o tempo dispendido em 
  uma operação. 
*/
void Timer::stop() {
  cudaEventRecord(end);
  cudaEventSynchronize(end);
  cudaEventElapsedTime(&parcial, begin, end);
  total += parcial;
}

/*
  Retorna o tempo calculado em segundos. 
*/
double Timer::getTime() {
  return total / 1000;
}
