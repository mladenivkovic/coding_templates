/* ------------------------------------------------
 * Use GSL integration
 * ------------------------------------------------ */

#include <gsl/gsl_integration.h>
#include <math.h>
#include <stdio.h>

/* A function to be integrated.
 * params is necessary, even if you don't use them!
 * The __attribute__((unused)) is so gcc won't complain with -Werror */
double function1(double x, __attribute__((unused)) void* params) {
  return x * x * x + 2 * x * x - 7;
}

/* The analytical solution of the integral of the function */
double integral_function1(double x) {
  return 0.25 * x * x * x * x + 2. * x * x * x / 3. - 7. * x;
}

struct function2params {
  double alpha;
  double beta;
};

/* A function to be integrated.  */
double function2(double x, void* params) {
  struct function2params* p = (struct function2params*)params;
  double alpha = p->alpha;
  double beta = p->beta;
  return pow(x, alpha) - beta * x;
}

/* The analytical solution of the integral of the function */
double integral_function2(double x, struct function2params params) {
  double alpha = params.alpha;
  double beta = params.beta;
  return pow(x, alpha + 1.) / (alpha + 1) - beta * x * x * 0.5;
}

int main(void) {

  gsl_function F;
  gsl_integration_workspace* w = gsl_integration_workspace_alloc(1000);
  double a = 5;
  double b = 13;
  double result, error;

  F.function = &function1;
  F.params = NULL;
  /* You can define either absolute error (epsabs) or relative error (epsrel) */
  gsl_integration_qags(&F, a, b, /*espabs=*/0., /*epsrel=*/1e-7, 1000, w,
                       &result, &error);
  double expected1 = integral_function1(b) - integral_function1(a);

  printf("Function 1:\n");
  printf("   result          = % .18f\n", result);
  printf("   exact result    = % .18f\n", expected1);
  printf("   estimated error = % .18f\n", error);
  printf("   actual error    = % .18f\n", result - expected1);
  printf("   intervals       = %zu\n", w->size);

  struct function2params params = {0.3, 8.};
  F.function = &function2;
  F.params = &params;
  gsl_integration_qags(&F, a, b, /*espabs=*/0., /*epsrel=*/1e-7, 1000, w,
                       &result, &error);
  double expected2 =
      integral_function2(b, params) - integral_function2(a, params);

  printf("Function 1:\n");
  printf("   result          = % .18f\n", result);
  printf("   exact result    = % .18f\n", expected2);
  printf("   estimated error = % .18f\n", error);
  printf("   actual error    = % .18f\n", result - expected2);
  printf("   intervals       = %zu\n", w->size);
}
