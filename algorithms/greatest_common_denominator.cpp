#include <iostream>
#include <string>

using namespace std;

bool verbose = false;

void drawline(){
  cout << "-----------------------------------------------------------------\n";
}


/**
 * Find greatest common denominator
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
