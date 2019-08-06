/* ================================================================================ 
 * COMPILE WITH gcc -pthread simple_multithreads.c                                  
 *
 * get thread IDs.
 * ================================================================================ */


#include <stdio.h>
#include <pthread.h>

void *check_id();

int r1 = 0, r2 = 0;
pthread_t thread1, thread2, thread3, thread4, thread5;

extern int
main(void)
{

  pthread_create(&thread1,        /* pointer to a buffer for thread "ID"                                         */
           NULL,                  /* pointer to a thread attribute object                                        */
           check_id,              /* pointer to routine at which new thread will start executing                 */
           NULL);                 /* pointer to parameter to be passed to the routine at which new thread starts */
  /* returns: 0 for success, nonzero for errors */


  pthread_create(&thread2, NULL, check_id, NULL);
  pthread_create(&thread3, NULL, check_id, NULL);
  pthread_create(&thread4, NULL, check_id, NULL);
  pthread_create(&thread5, NULL, check_id, NULL);

  pthread_join(thread1, NULL);
  pthread_join(thread2, NULL);
  pthread_join(thread3, NULL);
  pthread_join(thread4, NULL);
  pthread_join(thread5, NULL);

  return 0;
}


void * check_id()
{

  pthread_t this_thread;
  this_thread = pthread_self();
  if (pthread_equal(thread1, this_thread)){
    printf("Hello from thread 1!\n");
  } else {
    printf("This is not thread 1.\n");
  };

  return NULL;
}


