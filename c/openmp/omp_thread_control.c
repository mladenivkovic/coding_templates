/*
 * Write some comments in here.
 * compile with -fopenmp flag (gcc)
 */

#include <omp.h>   /* openMP library     */
#include <stdio.h> /* input, output    */

void thread_numbers(void);
void nested_threads(void);
void mutex(void);
void barrier(void);
void master_single(void);
void locks(void);
void force_update(void);

int main(void) {

  /*manipulate thread numbers*/
  thread_numbers();

  /*nested threads*/
  /*it's still 2 threads as set before*/
  nested_threads();

  /*mutual exclusion*/
  mutex();

  /*synching by barrier*/
  barrier();

  /*let only master or any other thread do stuff*/
  master_single();

  /* lock data */
  locks();

  /* force update of a variable */
  force_update();

  return (0);
}

/*======================================================*/
/*======================================================*/
/*======================================================*/

void thread_numbers(void) {

  printf("=============================\n");
  printf("Thread number manipulation\n");
  printf("=============================\n");

  /*=============*/
  /*Contains:    */
  /*=============*/
  /*- #pragma omp parallel num_threads(x)*/
  /*- omp_set_num_threads(x);*/

  printf("Num_threads=2\n");

#pragma omp parallel num_threads(2)
  {
    int tid = omp_get_thread_num();
    printf("Hello from proc %d\n", tid);
  }

  printf("\nNum_threads=3\n");

#pragma omp parallel num_threads(3)
  {
    int tid = omp_get_thread_num();
    printf("Hello from proc %d\n", tid);
  }

  /*ALTERNATE VERSION*/
  printf("\nNum_threads=2 again\n");

  /* !!!! use before #pragma omp parallel !!!!!! */
  omp_set_num_threads(2);
#pragma omp parallel
  {
    int tid = omp_get_thread_num();
    printf("Hello from proc %d\n", tid);
  }

  printf("\n\n");
}

/*======================================================*/
/*======================================================*/
/*======================================================*/

void nested_threads(void) {

  printf("=============================\n");
  printf("Nested parallel regions \n");
  printf("=============================\n");

  int id, id2;

#pragma omp parallel private(id, id2)
  {
    id = omp_get_thread_num();
/*start nested parallel region*/
#pragma omp parallel num_threads(2) private(id2)
    {
      id2 = omp_get_thread_num();
      printf("Hey from thread %d.%d!\n", id, id2);
    }
  }
}

/*======================================================*/
/*======================================================*/
/*======================================================*/

void mutex(void) {

  printf("\n\n=============================\n");
  printf("Mutual exclusion\n");
  printf("=============================\n");

  /*=============*/
  /*Contains:    */
  /*=============*/
  /*- no mutual exclusion (repeatedly execute to see that sometimes the result
   * is wrong)*/
  /*- critical*/
  /*- atomic*/

  /*========*/
  /* set up */
  /*========*/

  int len = 60;
  int somearray[len];
  for (int i = 0; i < len; i++) {
    /*do random operation*/
    somearray[i] = 17 * i - 23 * i / 5;
  }

  int sum = 0, j, result = 0;

  /*================================*/
  /* No mutual exclusion                        */
  /*================================*/

#pragma omp parallel private(j), firstprivate(sum)
  {
    int id = omp_get_thread_num();
    int nthreads = omp_get_num_threads();

    /*split up work*/
    for (j = id * len / nthreads; j < (id + 1) * len / nthreads; j++) {
      sum += somearray[j];
    }

    /*sum up private sums into shared result*/
    /* without mutual exclusion, errors may occur "randomly". */
    printf("Thread %d accessing result variable\n", id);
    result += sum;
  }

  printf("No mutex result is: %d\n", result);

  /*================================*/
  /* Now mutual exclusion things up.            */
  /* use CRITICAL                   */
  /*================================*/

  sum = 0;
  result = 0;

#pragma omp parallel private(j), firstprivate(sum)
  {
    int id = omp_get_thread_num();
    int nthreads = omp_get_num_threads();

    /*split up work*/
    for (j = id * len / nthreads; j < (id + 1) * len / nthreads; j++) {
      sum += somearray[j];
    }

      /*sum up private sums into shared result*/
      /* "critical" only allows one thread to access this part.
       * you may even call other functions in this part. */
#pragma omp critical
    {
      printf("Thread %d accessing result variable\n", id);
      result += sum;
    }
  }

  printf("Mutex result is: %d\n", result);

  sum = 0;
  result = 0;

  /*================================*/
  /* use ATOMIC instead of CRITICAL */
  /*================================*/

#pragma omp parallel private(j), firstprivate(sum)
  {
    int id = omp_get_thread_num();
    int nthreads = omp_get_num_threads();

    /*split up work*/
    for (j = id * len / nthreads; j < (id + 1) * len / nthreads; j++) {
      sum += somearray[j];
    }

      /*sum up private sums into shared result*/
      /* "atomic" is only valid for the first line after the instruction.
       * it is used to update shared variables, just like "result" in this case.
       */
#pragma omp atomic
    result += sum;
  }

  printf("Atomic mutex result is: %d\n", result);
}

/*======================================================*/
/*======================================================*/
/*======================================================*/

void barrier(void) {

  printf("\n\n=============================\n");
  printf("Barrier\n");
  printf("=============================\n");

  omp_set_num_threads(4);
#pragma omp parallel
  {
    int id = omp_get_thread_num();

    /* do some stuff that takes different threads different amount of time */
    int i;
    long long pow = 1;

    /*pow = 1000^id*/
    for (i = 0; i < id; i++) {
      pow *= 100;
    }

    long long useless_sum = 0;
    for (int i = 0; i < pow; i++) {
      useless_sum += 1;
    }

    printf("Barriered: thread %d finished sum:     %10Ld\n", id, useless_sum);
#pragma omp barrier
    printf("id %d lock n load\n", id);
#pragma omp barrier

    useless_sum = 0;
    for (int i = 0; i < pow; i++) {
      useless_sum += 1;
    }

    printf("Not barriered: thread %d finished sum: %10Ld\n", id, useless_sum);

  } /* end parallel region */
}

/*======================================================*/
/*======================================================*/
/*======================================================*/

void master_single(void) {

  printf("\n\n=============================\n");
  printf("Master and single\n");
  printf("=============================\n");

#pragma omp parallel
  {
    int id = omp_get_thread_num();
    printf("id %d is online.\n", id);

#pragma omp master
    {
      /* more than one line possible */
      printf("id %d is master.\n", id);
    }

    /* give threads except 2 some work to "force" it to do the single
     * instruction*/
    int useless = 0;
    if (id != 2) {
      for (int i = 0; i == 100; i++) {
        useless += 1;
      }
    }

#pragma omp single
    { printf("id %d is doing the single instruction.\n", id); }
  } /* end parallel region */
}

/*======================================================*/
/*======================================================*/
/*======================================================*/

void locks(void) {

  /*===========================================================================*/
  /* A lock is somewhat similar to a critical section: it guarantees that some
   */
  /* instructions can only be performed by one process at a time. However, a */
  /* critical section is indeed about code; a lock is about data. */
  /* With a lock you make sure that some data elements can only be touched */
  /* by one process at a time. */
  /*===========================================================================*/

  printf("\n\n=============================\n");
  printf("Locks\n");
  printf("=============================\n");

  int numbers[20] = {5, 3, 6, 7, 3, 0, 4, 2, 6, 7,
                     1, 1, 0, 5, 3, 2, 6, 5, 3, 9};
  int ones[20] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  int counting[10];
  int totel = 0;

  for (int i = 0; i < 10; i++) {
    counting[i] = 0;
  }

  omp_lock_t writelock; /* name your lock. You can have multiple locks with
                           various names. */
  omp_lock_t printlock;
  omp_init_lock(&writelock);
  omp_init_lock(&printlock);

#pragma omp parallel
  {
  /*===================================================*/
  /* count how many times a number is in array numbers */
  /* store that information in array counting          */
  /*===================================================*/

#pragma omp for
    for (int i = 0; i < 20; i++) {
      omp_set_lock(&writelock);
      {
        /* set lock at address of where you want to write*/
        counting[numbers[i]] += 1;
      }
      omp_unset_lock(&writelock);
    }

#pragma omp for
    for (int i = 0; i < 10; i++) {
      omp_set_lock(&writelock);
      {
        totel += counting[i];
        counting[i] = 0; /* reset counting */
      }
      omp_unset_lock(&writelock);
    }

#pragma omp master
    { printf("Total elements counted lock:    %d\n", totel); }

    omp_destroy_lock(&writelock);

    /*============================*/
    /* do the same without a lock */
    /*============================*/

#pragma omp for
    for (int i = 0; i < 20; i++) {
      /* set lock at address of where you want to write*/
      counting[ones[i]] += 1;
    }

    totel = 0;
#pragma omp for
    for (int i = 0; i < 10; i++) {
      totel += counting[i];
    }

#pragma omp master
    { printf("Total elements counted no lock: %d\n", totel); }

    /*=====================*/
    /* printing with locks */
    /*=====================*/

    int id = omp_get_thread_num();

    omp_set_lock(&printlock);
    { printf("Ordered printing with locks: hello from id %d\n", id); }
    omp_unset_lock(&printlock);

    omp_destroy_lock(&printlock);

  } /*end parallel region */
}

/*======================================================*/
/*======================================================*/
/*======================================================*/

void force_update(void) {

  printf("\n\n=============================\n");
  printf("Force update\n");
  printf("=============================\n");

  int some_shared_var = 0;

#pragma omp parallel
  {

    int id = omp_get_thread_num();

#pragma omp single
    some_shared_var += 10;

/* I recommend using a barrier here just to make sure        */
/* the flush operation does not actually synchronize         */
/* different threads. It just ensures that a thread’s values */
/* are made consistent with main memory.                     */
#pragma omp flush(some_shared_var)

    printf("Thread %d now has some shared var = %d\n", id, some_shared_var);

  } /* end parallel region */
}

/*======================================================*/
/*======================================================*/
/*======================================================*/
