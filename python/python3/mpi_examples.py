# /usr/bin/env python3

# ==================================================
# Simple tutorial on how to use mpi4py
# run with mpirun -n 4 python3 mpi_examples.py
# ==================================================


from mpi4py import MPI


# ---------------------------------------
# Initialize
# ---------------------------------------

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
ntasks = comm.Get_size()

print("my rank is", rank, "out of", ntasks)


def rprint(*obj):
    print("Rank", rank, ":", *obj)


# ---------------------------------------
# Do nonblocking sends and receives
# ---------------------------------------
data = {"a": rank, "b": rank * 2}

# determine who sends to whom
if rank == ntasks - 1:
    dest = 0
else:
    dest = rank + 1

if rank == 0:
    src = ntasks - 1
else:
    src = rank - 1


reqs = comm.isend(data, dest=dest, tag=rank)
reqr = comm.irecv(source=src, tag=src)

reqs.wait()
newdata = reqr.wait()

rprint("got data:", newdata)


# ---------------------------------------
# Broadcasts
# ---------------------------------------

if rank == 0:
    bdat = "text to be broadcast"
else:
    bdat = None

bdat = comm.bcast(bdat, root=0)
rprint("Broadcast data:", bdat)
