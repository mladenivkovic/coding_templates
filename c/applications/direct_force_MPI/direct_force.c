/* ==============================================
 * Compute gravity forces using direct force
 * computations and MPI.
 * Compile with mpicc
 *
 * !!!!!! DESIGNED TO WORK WITH AN ODD NUMBER
 * !!!!!! OF PROCESSORS ONLY!!!!!!!!!!!!!!!!!
 * (it also doesn't work with only 1 proc)
 */

#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

double G = 1;
int npart = 1790;

/* =============================================== */
int main(int argc, char* argv[]) {
  /* =============================================== */

  /* misc declarations */
  int i, j;
  int rank, size;

  /* Global Arrays
   * IMPORTANT: These are ALL particle data, intended to simulate
   * what would be read in from a single file */
  double x[npart];
  double y[npart];
  double z[npart];
  double m[npart];

  /* initialize MPI */
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  /* invent some data so the code runs*/
  /* fill it up with rank+1 so you can check that communications work */
  for (i = 0; i < npart; i++) {
    x[i] = (rank + 1);
    y[i] = (rank + 1);
    z[i] = (rank + 1);
    m[i] = 1;
  }

  /* distribute particles amongst processors */
  int nguess = npart / size;
  int npart_local[size]; /* number of particles local to each processor */
  int npart_local_start
      [size]; /* cumulative sum of local number of particles, i.e. starting
                 index of particles for each rank */

  npart_local[0] = npart - (size - 1) * nguess;
  npart_local_start[0] = 0;

  for (i = 1; i < size; i++) {
    npart_local[i] = nguess;
    npart_local_start[i] = npart_local_start[i - 1] + npart_local[i - 1];
  }

  /* if (rank==0){ */
  /*   for (int i=0; i<size; i++){ */
  /*     printf("%d %d\n", npart_local[i], npart_local_start[i]); */
  /*   } */
  /* } */

  /* LOCAL arrays, unique to each processor */
  double x_loc[npart_local[rank]];
  double y_loc[npart_local[rank]];
  double z_loc[npart_local[rank]];
  double m_loc[npart_local[rank]];
  double fx_loc[npart_local[rank]];
  double fy_loc[npart_local[rank]];
  double fz_loc[npart_local[rank]];

  /* buffer arrays */
  double* x_buf;
  double* y_buf;
  double* z_buf;
  double* m_buf;
  double* fx_buf;
  double* fy_buf;
  double* fz_buf;
  double* fx_rbuf;
  double* fy_rbuf;
  double* fz_rbuf;

  for (i = 0; i < npart_local[rank]; i++) {
    /* transfer data to local arrays */
    x_loc[i] = x[npart_local_start[rank] + i];
    y_loc[i] = y[npart_local_start[rank] + i];
    z_loc[i] = z[npart_local_start[rank] + i];
    m_loc[i] = m[npart_local_start[rank] + i];
    /* initialize forces to zero */
    fx_loc[i] = 0.0;
    fy_loc[i] = 0.0;
    fz_loc[i] = 0.0;
  }

  /* FORCE COMPUTATION LOOP */

  int nloops = size / 2;

  for (int loop = 0; loop < nloops; loop++) {
    if (rank == 0) printf("Starting loop %d\n", loop);

    /* find who to send to */
    int send_to = rank + loop + 1;
    if (send_to >= size) send_to -= size;
    /* find who you receive from */
    int recv_from = rank - loop - 1;
    if (recv_from < 0) recv_from += size;

    /* for (int r=0; r<size; r++){ */
    /*   if (r==rank){ */
    /*     printf("Stepsize %d: rank %d will send to %d and receive from %d\n",
     * loop+1, rank, send_to, recv_from); */
    /*     fflush(stdout); */
    /*   } */
    /*   MPI_Barrier(MPI_COMM_WORLD); */
    /* } */

    /* allocate receive buffer */
    x_buf = malloc(npart_local[recv_from] * sizeof(double));
    y_buf = malloc(npart_local[recv_from] * sizeof(double));
    z_buf = malloc(npart_local[recv_from] * sizeof(double));
    m_buf = malloc(npart_local[recv_from] * sizeof(double));

    for (int r = 0; r < size; r++) {
      if (r == rank) {
        /* printf("rank %d sending to %d\n", rank, send_to); */
        MPI_Send(x_loc, npart_local[rank], MPI_DOUBLE, send_to, 661,
                 MPI_COMM_WORLD);
        MPI_Send(y_loc, npart_local[rank], MPI_DOUBLE, send_to, 662,
                 MPI_COMM_WORLD);
        MPI_Send(z_loc, npart_local[rank], MPI_DOUBLE, send_to, 663,
                 MPI_COMM_WORLD);
        MPI_Send(m_loc, npart_local[rank], MPI_DOUBLE, send_to, 664,
                 MPI_COMM_WORLD);
      }
      /* IMPORTANT!!!! else if, not else! otherwise, you wait for size-1 times
         to receive something ! */
      else if (r == recv_from) {
        /* printf("rank %d receiving from %d\n", rank, recv_from); */
        MPI_Recv(x_buf, npart_local[recv_from], MPI_DOUBLE, recv_from, 661,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(y_buf, npart_local[recv_from], MPI_DOUBLE, recv_from, 662,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(z_buf, npart_local[recv_from], MPI_DOUBLE, recv_from, 663,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(m_buf, npart_local[recv_from], MPI_DOUBLE, recv_from, 664,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      }
    }

    /* printf("rank %d received %.1lf %.1lf ... %.1lf from rank %d\n", */
    /*     rank, x_buf[0], x_buf[1], x_buf[npart_local[recv_from]-1],
     * recv_from); */

    /* allocate forces buffer for the particles you just received */
    fx_buf = malloc(npart_local[recv_from] * sizeof(double));
    fy_buf = malloc(npart_local[recv_from] * sizeof(double));
    fz_buf = malloc(npart_local[recv_from] * sizeof(double));

    /* initialize forces to zero */
    for (i = 0; i < npart_local[recv_from]; i++) {
      fx_buf[i] = 0;
      fy_buf[i] = 0;
      fz_buf[i] = 0;
    }

    /* now compute direct forces */
    for (i = 0; i < npart_local[rank]; i++) {
      for (j = 0; j < npart_local[recv_from]; j++) {
        double dx = x_loc[i] - x_buf[j];
        double dy = y_loc[i] - y_buf[j];
        double dz = z_loc[i] - z_buf[j];
        double r3half = sqrt(dx * dx + dy * dy + dz * dz); /* r**1/2 */
        r3half = r3half * r3half * r3half;                 /* r**3/2 */

        double fct = -m_loc[i] * m_buf[j] * G / r3half;
        fx_loc[i] += fct * dx;
        fy_loc[i] += fct * dy;
        fz_loc[i] += fct * dz;
        fx_buf[j] -= fct * dx;
        fy_buf[j] -= fct * dy;
        fz_buf[j] -= fct * dz;
      }
    }

    /* allocate forces receive buffer so you can receive what happened on other
     * tasks */
    fx_rbuf = malloc(npart_local[rank] * sizeof(double));
    fy_rbuf = malloc(npart_local[rank] * sizeof(double));
    fz_rbuf = malloc(npart_local[rank] * sizeof(double));

    /* send forces back */
    /* what once was the receiver will now be the sender */
    for (int r = 0; r < size; r++) {
      if (r == recv_from) {
        /* printf("rank %d wants to send %d to %d\n", rank,
         * npart_local[recv_from], recv_from); */
        MPI_Send(fx_buf, npart_local[recv_from], MPI_DOUBLE, recv_from, 111,
                 MPI_COMM_WORLD);
        MPI_Send(fy_buf, npart_local[recv_from], MPI_DOUBLE, recv_from, 112,
                 MPI_COMM_WORLD);
        MPI_Send(fz_buf, npart_local[recv_from], MPI_DOUBLE, recv_from, 113,
                 MPI_COMM_WORLD);
      }
      if (r == rank) {
        /* printf("rank %d wants to receive %d from %d\n", rank,
         * npart_local[rank], send_to); */
        MPI_Recv(fx_rbuf, npart_local[rank], MPI_DOUBLE, send_to, 111,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(fy_rbuf, npart_local[rank], MPI_DOUBLE, send_to, 112,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(fz_rbuf, npart_local[rank], MPI_DOUBLE, send_to, 113,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      }
    }

    /* add up the forces you got */
    for (i = 0; i < npart_local[rank]; i++) {
      fx_loc[i] += fx_rbuf[i];
      fy_loc[i] += fy_rbuf[i];
      fz_loc[i] += fz_rbuf[i];
    }

    /* deallocate temporary buffer arrays so they can be reallocated in the next
     * loop */
    free(x_buf);
    free(y_buf);
    free(z_buf);
    free(m_buf);
    free(fx_buf);
    free(fy_buf);
    free(fz_buf);
    free(fx_rbuf);
    free(fy_rbuf);
    free(fz_rbuf);
  }

  /* no more MPI routines after this point */
  MPI_Finalize();
  if (rank == 0) printf("Done. Bye\n");

  return (0);
}
