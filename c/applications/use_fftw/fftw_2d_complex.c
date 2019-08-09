//=========================================================================================================
// Simple example of how to use fftw3.
// if there is a global fftw3 lib installed:
//    gcc use_fftw.c -o use_fftw.exe -lfftw3 -lm
// if you have a local lib installed:
//    gcc use_fftw.c -o use_fftw.exe -L/home/mivkov/.local/lib
//    -I/home/mivkov/.local/include -lfftw3 -lm
//=========================================================================================================

// If you have a C compiler, such as gcc, that supports the C99 standard, and
// you #include <complex.h> before <fftw3.h>, then fftw_complex is the native
// double-precision complex type and you can manipulate it with ordinary
// arithmetic. Otherwise, FFTW defines its own complex type, which is
// bit-compatible with the C99 complex type.
//
// To use single or long-double precision versions of FFTW, replace the fftw_
// prefix by fftwf_ or fftwl_ and link with -lfftw3f or -lfftw3l, but use the
// same <fftw3.h> header file.

#include <fftw3.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int main(void) {
  int Nx = 400;
  int Ny = 200;  // dimension of input and output
  int nsamples = 200;
  fftw_complex *in, *out;  // pointers fo type fftw_complex; will contain input
                           // and output of the FFT IMPORTANT: allocate memory
                           // with fftw_malloc instead of malloc! complex type:
                           // typedef double fftw_complex[2]; [0] is real part,
                           // [1] is imaginary part
  fftw_plan
      my_plan;  // plan that will store the type of FFT that will be performed

  double pi = 3.1415926;
  double physical_length_x = 40;
  double physical_length_y = 20;
  double lambda1 = 0.5;
  double lambda2 = 0.7;
  double dx = physical_length_x / Nx;
  double dy = physical_length_y / Ny;
  double dkx = 1 / physical_length_x;
  double dky = 1 / physical_length_y;
  int n[2];
  n[0] = Nx;
  n[1] = Ny;

  int i, j, ind, ix, iy, ik;

  //-------------------------------------
  // allocate arrays for input/output
  //-------------------------------------
  in = fftw_malloc(sizeof(fftw_complex) * Nx * Ny);
  out = fftw_malloc(sizeof(fftw_complex) * Nx * Ny);

  //-------------------------------------
  // Create Plan
  //-------------------------------------
  my_plan = fftw_plan_dft(2, n, in, out, FFTW_FORWARD, FFTW_ESTIMATE);

  // get plan:
  // rank (dimension) 2
  // n : array containing array lengths per dimension
  // in/out: input/output arrays
  // FFTW_FORWARD: Do Forward transform. for backwards, use FFTW_BACKWARDS.
  // FFTW_ESTIMATE: how well fft must be optimized. i
  //    FFTW_ESTIMATE does barely any optimisation, but creating plan is fast.
  //    Use if you don't want to repeat transform often.
  //    Otherwise, use FFTW_MEASURE.
  // IMPORTANT: CREATE PLAN BEFORE INITIALISING ARRAYS. FFTW_MEASURE overwrites
  // in/out arrays. Once the plan has been created, you can use it as many times
  // as you like for transforms on the specified in/out arrays, computing the
  // actual transforms via fftw_execute(plan)

  //---------------------------------
  // fill up array with a wave
  //---------------------------------
  for (i = 0; i < Nx; i++) {
    for (j = 0; j < Ny; j++) {
      ind = i * Ny + j;
      in[ind][0] =
          cos(2.0 * pi / lambda1 * i * dx) + sin(2.0 * pi / lambda2 * j * dy);
    }
  }

  //-----------------------
  // execute fft
  // -----------------------
  fftw_execute(my_plan); /* repeat as needed */

  //---------------------------
  // Calculate power spectrum
  //---------------------------
  double *Pk, *distances_k, d, kmax;
  double *Pk_field;

  Pk_field = malloc(sizeof(double) * Nx * Ny);
  for (i = 0; i < Nx; i++) {
    for (j = 0; j < Ny; j++) {
      ind = i * Ny + j;
      Pk_field[ind] = out[ind][0] * out[ind][0] + out[ind][1] * out[ind][1];
    }
  }

  Pk = calloc(Nx * Ny, sizeof(double));
  distances_k = calloc(Nx * Ny, sizeof(double));
  kmax = sqrt(pow((Nx / 2 + 1) * dkx, 2) + pow((Ny / 2 + 1) * dky, 2));
  for (i = 0; i < nsamples; i++) {
    distances_k[i] = 1.0001 * i / nsamples *
                     kmax;  // add a little more to make sure kmax will fit
  }

  //------------------------
  // histogrammize P(k)
  //------------------------
  for (i = 0; i < Nx; i++) {

    if (i < Nx / 2 + 1) {
      ix = i;
    } else {
      ix = -Nx + i;
    }

    for (j = 0; j < Ny; j++) {

      if (j < Ny / 2 + 1) {
        iy = j;
      } else {
        iy = -Ny + j;
      }

      ind = i * Ny + j;
      d = sqrt(pow(ix * dkx, 2) + pow(iy * dky, 2));

      for (ik = 0; ik < nsamples; ik++) {
        if (d <= distances_k[ik] || ik == nsamples) {
          break;
        }
      }

      Pk[ik] += Pk_field[ind];
    }
  }

  //-----------------------------------
  // write arrays to file.
  // can plot them with plot_fftw.py
  //-----------------------------------
  FILE *filep;
  filep = fopen("./fftw_output_2d_complex.txt", "w");
  for (i = 0; i < nsamples; i++) {
    fprintf(filep, "%f\t%f\n", distances_k[i] * 2 * pi, Pk[i]);
  }
  fclose(filep);

  printf("Finished! Written results to ./fftw_output_2d_complex.txt\n");

  //----------------------
  // deallocate arrays
  //----------------------
  fftw_destroy_plan(my_plan);
  fftw_free(in);
  fftw_free(out);
  free(Pk);
  free(Pk_field);
  free(distances_k);

  return 0;
}
