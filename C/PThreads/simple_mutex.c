/* ================================================================================ 
 * COMPILE WITH gcc -pthread simple_mutex.c                                  
 * 2 threads will do 2 different things, but require access to the same variable,
 * which will be protected by a mutex.
 * ================================================================================ */


#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

void do_one_thing(int *);
void do_another_thing(int *);
void do_wrap_up(int, int);

int r1 = 0, r2 = 0, r3 = 0;
pthread_mutex_t r3_mutex=PTHREAD_MUTEX_INITIALIZER;

extern int
main(int argc, char **argv)
{
  pthread_t       thread1,  thread2;

  if (argc==1){
    printf("I need an integer for r3 as cmdline arg.\n");
    return(1);
  }
  r3 = atoi(argv[1]);

  pthread_create(&thread1,        /* pointer to a buffer for thread "ID"                                         */
           NULL,                  /* pointer to a thread attribute object                                        */
          (void *) do_one_thing,  /* pointer to routine at which new thread will start executing                 */
          (void *) &r1);          /* pointer to parameter to be passed to the routine at which new thread starts */
  /* returns: 0 for success, nonzero for errors */


  pthread_create(&thread2,
          NULL,
          (void *) do_another_thing,
          (void *) &r2);

  pthread_join(thread1,  NULL);
  pthread_join(thread2,  NULL);

  do_wrap_up(r1, r2);
  return 0;
}


void do_one_thing(int *pnum_times)
{
  int i, j, x;

  pthread_mutex_lock(&r3_mutex);
  if (r3 > 0) {
     x  =  r3;
     r3--;
  }else {
     x  =  1;
  }
  printf("thread 1 accessing r3. r3 is now: %d\n", r3);
  pthread_mutex_unlock(&r3_mutex);

  for (i = 0;    i  <  4;  i++) {
    for (j = 0; j < 10000; j++) x = x + i;
    (*pnum_times)++;
  }
}

void do_another_thing(int *pnum_times)
{
  int i, j, x;

  pthread_mutex_lock(&r3_mutex);
  if (r3 > 0) {
     x  =  r3;
     r3 *= 2;
  }else {
     x  =  1;
  }
  printf("thread 2 accessing r3. r3 is now: %d\n", r3);
  pthread_mutex_unlock(&r3_mutex);


  for (i = 0;    i  <  4;  i++) {
    for (j = 0; j < 10000; j++) x = x + i;
    (*pnum_times)++;
  }
}

void do_wrap_up(int one_times, int another_times)
{
 int total;

 total = one_times + another_times;
 printf("wrap up: one thing %d, another %d, total %d\n",

 one_times, another_times, total);
}
