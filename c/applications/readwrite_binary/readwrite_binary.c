/**
 * Write out a binary dump and read it back in.
 */


#include "data_struct.h"

#include <stdio.h>

int main(void) {

  char outfname[] = "output.dat";
  FILE* out_fp = fopen(outfname, "wb");

  const int N = 10;

  for (int i = 0; i < N; i++){

    struct output_data out = { i, i*2, 'a'+i, i + 2.f, i + 3. };

    fwrite(&out, sizeof(struct output_data), 1, out_fp);

  }

  fclose(out_fp);

  printf("Written file %s\n.", outfname);

  FILE* in_fp = fopen(outfname, "rb");

  for (int i = 0; i < N; i++){

    struct output_data in;

    fread(&in, sizeof(struct output_data), 1, in_fp);

    printf("i=%d, in.index=%d, in.ind=%d, int.char=%c, int.fl=%f, int.d=%lf\n",
        i, in.index, in.some_int, in.some_char, in.some_float, in.some_double);

  }

  fclose(in_fp);

  printf("Done.\n");

  return 0;

}
