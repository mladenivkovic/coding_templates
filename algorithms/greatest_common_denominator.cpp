#include <iostream>
#include <string>

using namespace std;

bool verbose = false;

void drawline(){
  cout << "-----------------------------------------------------------------\n";
}


/**
 * Find greatest common denominator using the Euclidean algorithm.
 *
 * The idea is the following: To find the greatest common denominator
 * between integers m and n, we make use of the fact that you can write
 * them as 
 *
 *    m = m1 * g;   n = n1 * g;
 *
 * where g is the greatest common denominator. Let m >= n.
 * If m % n == 0, then n is the greatest common denominator.
 * Otherwise, we can write
 *
 *    m - n = (m1 - n1) * g
 *
 * which obviously still has the same greatest common denominator g.
 * Furthermore, we don't have to subtract n from m only once. We can
 * do it as many times as we want, as long as the result remains >= 0.
 * Let d1 = m // n, where '//' denotes an integer division. Then
 *
 *    m - d1 * n = (m1 - d1 * n1) * g
 *
 * still contains the same greatest denominator g. This operation above
 * is the same as taking the remainder:
 *
 *    m - d1 * n = m % n = (m1 - d1 * n1) * g
 *
 * By definition of the remainder, m % n will always be smaller than n.
 * We can now re-name
 *
 *    m <- n
 *    n <- m % n
 *
 * and repeat the algorithm using increasingly smaller numbers for m and n, 
 * which makes it converge as we approach n = 1.
 **/
void greatest_common_denominator(int m, int n){

  int m_init = m;
  int n_init = n;
  int r;
  int i = 0;

  while (n != 0) {
    int m_start = m;
    int n_start = n;

    r = m % n;
    m = n;
    n = r;
    if (verbose) cout << "-- i=" << i;
    if (verbose) cout << " m_start=" << m_start;
    if (verbose) cout << " n_start=" << n_start;
    if (verbose) cout << " m % n=" << r;
    if (verbose) cout << " m<-" << m;
    if (verbose) cout << " n<-" << n << endl;
    i++;
  }

  cout << " m_init = " << m_init;
  cout << " n_init = " << n_init;
  cout << " greatest common denominator = " << m << endl;
  if (verbose) drawline();
  return;
}

int main() {

  greatest_common_denominator(2, 4);
  greatest_common_denominator(12, 4);
  greatest_common_denominator(24, 20);
  greatest_common_denominator(124, 132);
  greatest_common_denominator(125, 132);

  return 0;
}
