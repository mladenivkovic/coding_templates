/* =================================================
 * Generate unique cell IDs for swift.
 * Constraints:
 *  - must be thread safe
 *  - must be reproducible
 *
 *  We leave the first 15 bits of an unsigned long
 *  long for the top level cells, allowing for
 *  32^3 unique top level cells. For every level of
 *  prodigy cells, we add three bits at the leftmost
 *  available position based on the prodigy cell's
 *  position within the parent cell. Finally, add a
 *  leading 1 to mark that the cell ID is taken at
 *  that level and to make it unique.
 *
 *  Simplified Example:
 *   - using 3 rightmostbits instead of 15 for top
 *   level cells
 *   - dots every three digits are included for
 *   improved readability
 *
 *  Depth     Cell ID binary
 *  -------------------------
 *   0        000.000.000.001
 *   1        000.001.000.001
 *   1        000.001.001.001
 *   1        000.001.010.001
 *   1        000.001.011.001
 *   1        000.001.100.001
 *   1        000.001.101.001
 *   1        000.001.110.001
 *   1        000.001.111.001
 *   2        001.000.000.001
 *   2        001.001.000.001
 *   2        001.010.000.001
 *   2        001.011.000.001
 *   2        001.100.000.001
 *   2        001.101.000.001
 *   2        001.110.000.001
 *   2        001.111.000.001
 *      .
 *      .
 *      .
 *   2        001.111.110.001
 *   2        001.111.111.001
 *
 *  etc
 * ================================================= */

#include <math.h>
#include <stdio.h> /* input, output    */
#include <stdlib.h>

/* number of top level cells in each dimension */
#define NTOPLEVEL 8
/* depth of tree */
#define MAXDEPTH 6
/* WARNING: the output written is formatted. Per cell, this takes ~ 30 byte.
 * You will be creating NTOPLEVEL^3 * 8^MAXDEPTH cells, so be careful not to
 * fill up your hard drive within minutes. E.g. NTOPLEVEL=12 and MAXDEPTH=6
 * creates a 13 GB file.*/

/* uncomment if you want to print results to screen */
/* #define PRINT_STDOUT */

/* uncomment if you want to print IDs to file */
#define PRINT_FILE

#ifdef PRINT_FILE
FILE* outfilep;
#endif

unsigned long long ullpow(unsigned long long a, int b) {
  /* computes a^b for unsigned long long a and integer b */
  unsigned long long res = 1;
  for (int i = 0; i < b; i++) res *= a;
  return res;
}

/* returns binary representation of integer a as a string,
 * ready to be printed */
char* binary(unsigned long long a) {

  char* binary_string;

  /* get binary string */

  /* int bits = sizeof(unsigned long long) * 8 - 1; */
  /*  */
  /* binary_string = malloc(sizeof(char) * bits); */
  /*  */
  /* for (int i = bits - 1; i >= 0; i--) { */
  /*   long long pow2 = ullpow(2, i); */
  /*   long long div = a / pow2; */
  /*   if (div > 0) { */
  /*     binary_string[bits - i - 1] = '1'; */
  /*     a -= div * pow2; */
  /*   } else { */
  /*     binary_string[bits - i - 1] = '0'; */
  /*   } */
  /* } */
  /*  */

  /* get binary string with a dot after 15 digits from the left*/

  int bits = sizeof(unsigned long long) * 8;
  binary_string = malloc(sizeof(char) * bits);

  for (int i = bits; i >= 0; i--) {
    if (i == 15) {
      binary_string[bits - i - 1] = '.';
      continue;
    }
    unsigned long long pow2 = ullpow(2, i - 1);
    unsigned long long div = a / pow2;
    if (div > 0ULL) {
      binary_string[bits - i - 1] = '1';
      a -= div * pow2;
    } else {
      binary_string[bits - i - 1] = '0';
    }
  }

  return (binary_string);
}

struct cell {
  unsigned long long cellID;
  unsigned long long parentID;
  unsigned long long pos_in_parent;
  int depth;
};

void assign_cellID(struct cell* c) {
  /* Assign the cellID to a non-top level cell */

  unsigned long long newID = c->parentID;

  /* if parent isn't top level cell, we have to
   * zero out the marker of the previous depth first */
  if (c->depth > 1) newID &= ~(1ULL << ((c->depth - 1) * 3 + 15));

  /* add marker for this depth */
  unsigned long long marker = 1 << (c->depth * 3 + 15);
  /* add marker for position in parent cell */
  marker |= c->pos_in_parent << ((c->depth - 1) * 3 + 15);

  /* finish up */
  newID |= marker;
  c->cellID = newID;

#ifdef PRINT_STDOUT
  printf("Depth %3d parent %s (%lld)\n", c->depth, binary(parentID), parentID);
  printf("      %3s     ID %s (%lld)\n", " ", binary(newID), newID);
#endif
}

void split_cell_recursively(struct cell* c) {
  /* split a cell into 8 children recursively
   * until you reach MAXDEPTH */

  int newdepth = c->depth + 1;
  if (newdepth > MAXDEPTH) return;

  for (unsigned long long i = 0; i < 8; i++) {
    struct cell child;
    child.pos_in_parent = i;
    child.depth = newdepth;
    child.parentID = c->cellID;
    assign_cellID(&child);
#ifdef PRINT_FILE
    fprintf(outfilep, "%lld, %lld, %d\n", child.cellID, child.parentID,
            newdepth);
#endif
    split_cell_recursively(&child);
  }
}

int main() {

  /* safety checks */
  if (NTOPLEVEL > 32) {
    printf("ERROR: Can't work with > 32^3 top level cells\n");
    return 1;
  }
  if (MAXDEPTH > 16) {
    printf("ERROR: Can't work with MAXDEPTH > 16\n");
    return 1;
  }

    /* open file for writing? */
#ifdef PRINT_FILE
  outfilep = fopen("output_unique_cellIDs.txt", "w");
  fprintf(outfilep, "# cell, parent, depth\n");
#endif

  /* create top level cells */

  int ntopcells = NTOPLEVEL * NTOPLEVEL * NTOPLEVEL;
  struct cell* topcells = malloc(ntopcells * sizeof(struct cell));

  for (int c = 0; c < ntopcells; c++) {
    topcells[c].cellID = (unsigned long long)(c + 1);
    topcells[c].parentID = 0;
    topcells[c].pos_in_parent = 0;
    topcells[c].depth = 0;
  }

  for (int c = 0; c < ntopcells; c++) {
#ifdef PRINT_FILE
    struct cell tc = topcells[c];
    fprintf(outfilep, "%lld, %lld, %d\n", tc.cellID, tc.parentID, tc.depth);
#endif
    split_cell_recursively(&topcells[c]);
  }

#ifdef PRINT_FILE
  fclose(outfilep);
#endif
  return 0;
}
