#ifndef __TIMER__
#define __TIMER__

/*
  Classe responsável pelo cálculo do tempo de execução entre dois trechos de 
  código. Esta classe pode ser utilizada para mensurar o tempo gasto na 
  execução de métodos em GPU. O método "start" inicia a contagem do tempo, 
  "stop" termina a contagem do tempo e "getTime" retorna o tempo dispendido 
  em segundos. 
*/
struct Timer {
  
  private:
  
  cudaEvent_t begin, end;
  float parcial, total;
  
  public:
  
  Timer();
  void start();
  void stop();
  double getTime();
  
};

#endif
