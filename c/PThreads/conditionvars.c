/* The main routine creates two threads. Each of these threads runs  */
/* the inc_count routine. The inc_count routine locks count_mutex,  */
/* increments count, reads count in a printf statement, and tests  */
/* for the threshold value. If count has reached its threshold value,  */
/* inc_count calls pthread_cond_signal to notify the thread that’s  */
/* waiting for this particular event. Before exiting, inc_count  */
/* releases the mutex. We’ll create a third thread to run the  */
/* watch_count task. The watch_count routine waits for inc_count to  */
/* signal our count_threshold_cv condition variable. */



#include <stdio.h>
#include <pthread.h>
#define TCOUNT 10
#define WATCH_COUNT 12

int count = 0;
pthread_mutex_t count_mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t count_threshold_cv = PTHREAD_COND_INITIALIZER;
int  thread_ids[3]  =  {0,1,2};

void watch_count(int *);
void inc_count(int *);

extern int
main(void)
{
  int      i;
  pthread_t threads[3];
  pthread_create(&threads[0],NULL, (void *) watch_count, &thread_ids[2]);
  pthread_create(&threads[1],NULL, (void *) inc_count, &thread_ids[1]);
  pthread_create(&threads[2],NULL, (void *) inc_count, &thread_ids[0]);
  for (i = 0; i < 3; i++) {
           pthread_join(threads[i], NULL);
  }
  return 0;
}

void watch_count(int *idp)
{
  pthread_mutex_lock(&count_mutex);
  while (count <= WATCH_COUNT) {
    pthread_cond_wait(&count_threshold_cv, &count_mutex);
    /* releases the count mutex while the thread is waiting so other 
     * threads have the opportunity to modify count.*/
    printf("watch_count(): Thread %d,Count is %d\n", *idp, count);
  }
  pthread_mutex_unlock(&count_mutex);
}

void inc_count(int *idp)
{
  for (int i =0; i < TCOUNT; i++) {
    pthread_mutex_lock(&count_mutex);
    count++;
    printf("inc_count(): Thread %d, old count %d,\
      new count %d\n", *idp, count - 1, count );
    if (count == WATCH_COUNT)
      pthread_cond_signal(&count_threshold_cv);
    pthread_mutex_unlock(&count_mutex);
  }
}
