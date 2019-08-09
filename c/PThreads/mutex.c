/* ================================================================================
 * COMPILE WITH gcc -pthread mutex.c
 * 2 threads will do 2 different things, but require access to the same
 * variable, which will be protected by a mutex.
 * ================================================================================
 */

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

void *do_one_thing();
void *do_another_thing();
void *lock_something();
void *try_if_locked();

int r1 = 0, r2 = 0, r3 = 0;
pthread_mutex_t r3_mutex = PTHREAD_MUTEX_INITIALIZER;

extern int main(int argc, char **argv) {
  pthread_t thread1, thread2;

  if (argc == 1) {
    printf("I need an integer for r3 as cmdline arg.\n");
    return (1);
  }
  r3 = atoi(argv[1]);

  printf("Simple mutex\n");

  pthread_create(&thread1, /* pointer to a buffer for thread "ID"     */
                 NULL,     /* pointer to a thread attribute object         */
                 do_one_thing, /* pointer to routine at which new thread will
                                  start executing                 */
                 NULL); /* pointer to parameter to be passed to the routine at
                           which new thread starts */
  /* returns: 0 for success, nonzero for errors */

  pthread_create(&thread2, NULL, do_another_thing, NULL);

  pthread_join(thread1, NULL);
  pthread_join(thread2, NULL);

  pthread_create(&thread1, NULL, lock_something, NULL);

  pthread_create(&thread2, NULL, try_if_locked, NULL);

  pthread_join(thread1, NULL);
  pthread_join(thread2, NULL);

  return 0;
}

void *do_one_thing() {
  pthread_mutex_lock(&r3_mutex);
  printf("thread 1 accessing r3. r3 is now: %d\n", r3);
  if (r3 > 0) {
    r3--;
  } else {
    r3 = 100;
  }
  pthread_mutex_unlock(&r3_mutex);

  return NULL;
}

void *do_another_thing() {
  pthread_mutex_lock(&r3_mutex);
  printf("thread 2 accessing r3. r3 is now: %d\n", r3);
  if (r3 > 0) {
    r3 += 2;
  } else {
    r3 = 1;
  }
  pthread_mutex_unlock(&r3_mutex);

  return NULL;
}

void *lock_something() {
  pthread_mutex_lock(&r3_mutex);
  printf("thread 1 locked r3. Now wasting some time.\n");
  for (int i = 0; i < 10000000; i++) {
    r3 += 1;
  }
  printf("thread 1 done. Unlocking r3.\n");
  pthread_mutex_unlock(&r3_mutex);

  return NULL;
}

void *try_if_locked() {

  /* pthread_mutex_trylock doesn't block if variable is locked. */

  int test;
  int someint;
  int i;
  /* waste some time so thread 1 can lock variable, otherwise you'll deadlock
   * or get stuff you don't want to */
  for (i = 1; i < 1000000; i++) {
    someint += 1;
  }

  while ((test = pthread_mutex_trylock(&r3_mutex))) {
    printf("thread 2 trying lock: %d\n", test);
    /* now waste some time */
    someint = 0;
    for (i = 1; i < 1000000; i++) {
      someint += 1;
    }
  }
  printf("thread 2 got lock: %d\n", test);

  /* !!!!!!!!!!!!!!!!!IMPORTANT!!!!!!!!!!!!
   * pthread_mutex_trylock locks the variable! you need to unlock it! */
  pthread_mutex_unlock(&r3_mutex);

  return NULL;
}
