/*--------------------------------------------------
 * Printing bits of variables following
 * https://jameshfisher.com/2017/02/23/printing-bits/
 *------------------------------------------------ */

#ifndef BINARY_REPRESENTATION_H
#define BINARY_REPRESENTATION_H

#include <errno.h>
#include <math.h>
#include <stdio.h>

#include <float.h>
#include <limits.h>
#include <stdint.h>

#include "binary_representation.h"

/* When returning strings: Use 149 spaces:
 * 16 bytes of 8 digits, + 1 space after each byte
 * 2 extra spaces for start and end brackets
 * 1 extra space for Null char */
#define STRLEN 149

/**
 * print a single byte as bits.  To be used repeatedly
 * while printing a single variable.
 **/
void print_byte_as_bits(char val) {
  for (int i = 7; 0 <= i; i--) {
    printf("%c", (val & (1 << i)) ? '1' : '0');
  }
}

/**
 * @brief print the bits of a type 'type' with value 'val'
 **/
void print_bits(char* type, char* val, unsigned char* bytes, size_t num_bytes) {
  printf("(%*s) %*s = [ ", 15, type, 16, val);
  for (size_t i = 0; i < num_bytes; i++) {
    print_byte_as_bits(bytes[i]);
    printf(" ");
  }
  printf("]\n");
}

/*! The actual function to be used. Prints output directly to stdout.*/
#define SHOW(T, V)                                     \
  do {                                                 \
    T x = V;                                           \
    print_bits(#T, #V, (unsigned char*)&x, sizeof(x)); \
  } while (0)

/**
 * @brief Write a single byte into a given string.
 **/
void write_byte_as_bits(char val, char* output) {
  for (int i = 7; i >= 0; i--) {
    /* if i-th bit of val = 1, store char '1' */
    output[7 - i] = (val & (1 << i)) ? '1' : '0';
  }
}

/**
 * Write bits into string.
 **/
void write_bits(unsigned char* bytes, size_t num_bytes, char output[STRLEN]) {

  /* number of extra characters at start and end of string*/
  size_t offset = 1;
  output[0] = '[';
  /* 8 + 1: add space after each byte */
  output[9 * num_bytes + 2 * offset - 1] = ']';
  output[9 * num_bytes + 2 * offset] = '\0';

  for (size_t i = num_bytes; i > 0; i--) {

    /* write the first byte in last position */
    write_byte_as_bits(bytes[num_bytes - i], &output[(i - 1) * 9 + offset]);

    /* Add space after byte is written */
    output[i * 9 - 1 + offset] = ' ';
  }
}

/*! The actual function to be used. Writes result into string. */
#define GET_BINARY_STRING(Type, Value, Output)         \
  {                                                    \
    Type x = Value;                                    \
    write_bits((unsigned char*)&x, sizeof(x), Output); \
  }

#endif
