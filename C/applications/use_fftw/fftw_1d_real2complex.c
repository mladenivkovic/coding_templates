//=========================================================================================================
// Simple example of how to use fftw3.
// if there is a global fftw3 lib installed:
//    gcc use_fftw.c -o use_fftw.exe -lfftw3 -lm
// if you have a local lib installed:
//    gcc use_fftw.c -o use_fftw.exe -L/home/mivkov/.local/lib -I/home/mivkov/.local/include -lfftw3 -lm
//=========================================================================================================

// If you have a C compiler, such as gcc, that supports the C99 standard, and you 
// #include <complex.h> before <fftw3.h>, then fftw_complex is the native double-precision 
// complex type and you can manipulate it with ordinary arithmetic. Otherwise, FFTW defines 
// its own complex type, which is bit-compatible with the C99 complex type.
//
// To use single or long-double precision versions of FFTW, replace the fftw_ prefix by 
// fftwf_ or fftwl_ and link with -lfftw3f or -lfftw3l, but use the same <fftw3.h> header file. 

#include <fftw3.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int main(void)
{

    int N=1000;                     // dimension of input and output
    fftw_complex *out;              // pointers fo type fftw_complex; will contain input and output of the FFT
                                    // IMPORTANT: allocate memory with fftw_malloc instead of malloc!
                                    // complex type: typedef double fftw_complex[2]; [0] is real part, [1] is imaginary part
    double *in;                     // HERE, INPUT IS PURELY REAL.
    fftw_plan my_plan;              // plan that will store the type of FFT that will be performed
 
    double pi = 3.1415926;
    double physical_length = 100.0;
    double lambda1 = 0.5;
    double lambda2 = 0.7;
    double dx = physical_length/N;
    double dk = 1.0/physical_length;

    int i;


    //-------------------------------------
    // allocate arrays for input/output
    //-------------------------------------
    // IMPORTANT: input/output arrays don't have same sizes!
    // FFTW makes use of the fact that when input f(x) is real, then F(-k) = F*(k)
    // in = (double *) fftw_malloc(sizeof(double) * N);
    // out = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * N/2+1);
  in  = fftw_malloc(sizeof(double) * N);
  out = fftw_alloc_complex(N/2+1);
  // out = fftw_malloc(sizeof(fftw_complex) * N/2+1); // doesn't work for some reason?! not even when type casting to (fftw_complex *) fftw_malloc(..)

    for (i=0; i<N; i++){
      in[i] = cos(2.0*pi/lambda1*i*dx)+sin(2.0*pi/lambda2*i*dx);
    }

    //-------------------------------------
    // Create Plan
    //-------------------------------------
    my_plan = fftw_plan_dft_r2c_1d(N, in, out, FFTW_ESTIMATE);
    // my_plan = fftw_plan_dft_c2d_1d(N, out, in, FFTW_ESTIMATE); // backwards transform
    // IMPORTANT: the inverse transform (complex to real) has the side-effect of overwriting its input array

    // get plan: 
    // N elements in in and out
    // in/out: input/output arrays
    // NOTE: no FFTW_FORWARD/BACKWARD here!
    // FFTW_ESTIMATE: how well fft must be optimized. i
    //    FFTW_ESTIMATE does barely any optimisation, but creating plan is fast.
    //    Use if you don't want to repeat transform often.
    //    Otherwise, use FFTW_MEASURE.
    // IMPORTANT: CREATE PLAN BEFORE INITIALISING ARRAYS. FFTW_MEASURE overwrites in/out arrays.
    // Once the plan has been created, you can use it as many times as you like 
    // for transforms on the specified in/out arrays, computing the actual transforms 
    // via fftw_execute(plan) 
    

    //---------------------------------
    // fill up array with a wave
    //---------------------------------
    for (i=0; i<N; i++){
      in[i] = cos(2.0*pi/lambda1*i*dx)+sin(2.0*pi/lambda2*i*dx);
    }


    //-----------------------
    // execute fft
    //-----------------------
    fftw_execute(my_plan); /* repeat as needed */


    //---------------------------
    // Calculate power spectrum
    //---------------------------
    double *Pk;
    Pk = malloc(sizeof(double)*N);
    for (i=0; i<N/2+1; i++){
      Pk[i] = out[i][0]*out[i][0] + out[i][1]*out[i][1];
    }


    //-----------------------------------
    // write arrays to file.
    // can plot them with plot_fftw.py
    //-----------------------------------
    FILE *filep;
    filep = fopen("./fftw_output_1d_real.txt", "w");
    for (i=0; i<N/2+1; i++){
      fprintf(filep, "%f\t%f\n", i*dk*2*pi, Pk[i]);
    }
    fclose(filep);

    printf("Finished! Written results to ./fftw_output_1d_real.txt\n");


    //----------------------
    // deallocate arrays
    //----------------------
    fftw_destroy_plan(my_plan);
    fftw_free(in); fftw_free(out);
    free(Pk);

    return 0;
}
