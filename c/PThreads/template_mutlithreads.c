/* ================================================================================
 * COMPILE WITH gcc -pthread simple_multithreads.c
 *
 *
 * ================================================================================
 */

#include <pthread.h>
#include <stdio.h>

/* pthreads demands the functions that will be executed initially are of type
 * (void *) the arguments as well! So define them this way, otherwise the
 * compiler will hand out warnings:
 *
 * ISO C forbids conversion of function pointer
 * to object pointer type [-Wpedantic] (void *) do_another_thing
 *
 * You can just type cast the arguments inside the function, or place them in a
 * struct and pass a void* pointer to the struct in pthread_create. Just have a
 * look at the appropriate file in this direction on how to do that.
 */

void *do_one_thing(void *);
void *do_another_thing(void *);
void do_wrap_up(int, int);

int r1 = 0, r2 = 0;

extern int main(void) {
  pthread_t thread1, thread2;

  pthread_create(&thread1, /* pointer to a buffer for thread "ID"     */
                 NULL,     /* pointer to a thread attribute object         */
                 do_one_thing, /* pointer to routine at which new thread will
                                  start executing                 */
                 (void *)&r1); /* pointer to parameter to be passed to the
                                  routine at which new thread starts */
  /* returns: 0 for success, nonzero for errors */

  pthread_create(&thread2, NULL, do_another_thing, (void *)&r2);

  pthread_join(thread1, NULL);
  pthread_join(thread2, NULL);

  do_wrap_up(r1, r2);
  return 0;
}

void *do_one_thing(void *times) {
  int i, j, x;
  int *pnum_times = (int *)times;

  for (i = 0; i < 4; i++) {
    printf("doing one thing counter %d\n", i);
    for (j = 0; j < 10000; j++) x = x + i;
    (*pnum_times)++;
  }

  return NULL;
}

void *do_another_thing(void *times) {
  int i, j, x;
  int *pnum_times = (int *)times;

  for (i = 0; i < 4; i++) {
    printf("doing another counter %d\n", i);
    for (j = 0; j < 10000; j++) x = x + i;
    (*pnum_times)++;
  }

  return NULL;
}

void do_wrap_up(int one_times, int another_times) {
  int total;

  total = one_times + another_times;
  printf("wrap up: one thing %d, another %d, total %d\n",

         one_times, another_times, total);
}
