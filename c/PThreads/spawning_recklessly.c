/* ================================================================================ 
 * COMPILE WITH gcc -pthread spawning_recklessly.c
 *
 *  Try to spawn many threads. Let's see how many we can get.
 * ================================================================================ */


#include <stdio.h>
#include <pthread.h>
#include <stdlib.h>

/* define struct to pass args */
typedef struct{
  int id;
} argstruct;


void * do_one_thing(void *args)
{
  int i, j, x, id;

  argstruct *a = (argstruct *) args;

  id = a->id;

  printf("thread %d doing one thing \n", id);

  /* comment out following three lines to finish working fast */
  for (i = 0;    i  <  1000;  i++) {
    for (j = 0; j < 10000000; j++) x = x + i;
  }

  printf("thread %d finished doing one thing.\n", id);

  return NULL;
}

extern int
main(void)
{
  

  printf("This program is meant to be overdoing things and take a while so you can top in the meantime and check out what happens.\n");

  int N = 20;

  pthread_t * threadlist = malloc(N*sizeof(pthread_t));
  argstruct ** arglist = malloc(N*sizeof(argstruct*));

  for(int t = 0; t<N; t++){
    argstruct *a = malloc(sizeof(argstruct*));
    a->id = t;
    arglist[t] = a; 
    pthread_create(&threadlist[t],  /* pointer to a buffer for thread "ID"                                         */
            NULL,                   /* pointer to a thread attribute object                                        */
            do_one_thing,           /* pointer to routine at which new thread will start executing                 */
            (void *) arglist[t]);   /* pointer to parameter to be passed to the routine at which new thread starts */
  }



  for(int t = 0; t<N; t++){
    pthread_join(threadlist[t],  NULL);
  }

  return 0;
}


