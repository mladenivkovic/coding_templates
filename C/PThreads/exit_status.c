/* ================================================================================ 
 * COMPILE WITH gcc -pthread exit_status.c
 * Depending on cmd line arg, throw different exit number
 * ================================================================================ */
#include <stdio.h>
#include <pthread.h>
#include <stdlib.h>


pthread_t thread;
static const int real_bad_error = -12;
static const int normal_error = -10;
static const int success = 1;

void * routine_x(int * arg_in)
{
  if ( *arg_in==1) {
    pthread_exit((void *) &real_bad_error);
  }else if ( *arg_in==2 ) {
    return ((void *) &normal_error);
  }else {
    return ((void *) &success);
  }
}


extern int
main(int argc, char **argv)
{
  pthread_t thread;
  void *statusp;

  int arg_in = atoi(argv[1]);

  printf("(Need cmd line arg. 1: real bad error; 2: failure; else: success)\n");
  pthread_create(&thread, NULL, (void *) routine_x, &arg_in);
  pthread_join(thread, &statusp);
  if ((void*)statusp == PTHREAD_CANCELED) {
    printf("Thread was canceled.\n");
  }else {
    printf("Thread completed and exit status is %d.\n", *(int *)statusp);
    printf("If -12: Something really bad happened.\nIf -10: Some error happened.\nIf 1: success!\n");
  }
return 0;
}
