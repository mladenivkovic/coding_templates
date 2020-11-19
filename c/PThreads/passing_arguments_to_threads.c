/* =======================================================================
 * COMPILE WITH gcc -pthread simple_multithreads.c
 *
 *  Pass multiple arguments to dynamically generated threads.
 * =======================================================================
 */

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

/* define struct to pass args */
typedef struct {
  int id;
  int val1;
  float val2;
  char *string;
} argstruct;

void *do_one_thing(void *args) {
  int id, val1;
  float val2;
  char *string;

  argstruct *a = (argstruct *)args;

  id = a->id;
  val1 = a->val1;
  val2 = a->val2;
  string = a->string;

  printf("thread %d got %d, %f and '%s'\n", id, val1, val2, string);

  return NULL;
}

extern int main(void) {

  int N = 4;

  pthread_t *threadlist = malloc(N * sizeof(pthread_t));
  argstruct **arglist = malloc(N * sizeof(argstruct *));

  for (int t = 0; t < N; t++) {
    argstruct *a = malloc(sizeof(argstruct *));
    /* fill struct with junk args to show off*/
    a->id = t;
    a->val1 = (t + 1) * 2;
    a->val2 = (float)(10 * (t + 1)) / 7.3;
    a->string = "Hello World";
    arglist[t] = a;
    pthread_create(
        &threadlist[t], /* pointer to a buffer for thread "ID" */
        NULL,           /* pointer to a thread attribute object           */
        do_one_thing,   /* pointer to routine at which new thread will start
                           executing                 */
        (void *)arglist[t]); /* pointer to parameter to be passed to the
                                routine at which new thread starts */
  }

  for (int t = 0; t < N; t++) {
    pthread_join(threadlist[t], NULL);
  }

  return 0;
}
