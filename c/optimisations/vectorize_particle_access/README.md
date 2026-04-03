Vectorize Particle Access
=========================

Playing around with vectorizing memcpys of particle data, where the particle
data is stored in several split structs.

Also with obtaining and looking through optreports, as well as MAQAO.

Compile `full_test.c` to run test with all available versions/realisations of
copy loop.
Compile `main.c` to run single experiment run.



Findings so far
---------------

Problems with passing particles as a struct to getters and setters:

- Compiler doesn't know that `cell_part_data` will be a constant/identical
  pointer throughout a loop. It can't figure out we'll have sequential access.

- Compiler doesn't know that `part->index` will be sequential during a loop.

