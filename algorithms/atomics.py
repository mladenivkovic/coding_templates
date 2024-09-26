#!/usr/bin/env python3

# -------------------------------------------------
# Some simple principles behind atomic operations
# Note: These are not true atomic operations,
# but just demonstrations on how to do them
# -------------------------------------------------

from lockable_variables import lockable_var
import random


def atomic_cas(v: lockable_var, old, new):
    """
    Atomic compare and swap.
    if `v.val == old`, set to `new`

    returns:
        True if successfully changed var,
        False otherwise
    """

    # get a random key
    lockkey = random.randint(0, 4294967295)

    # make sure the value can't be changed
    # during the compare and swap:
    v.lock(lockkey)

    if v.val == old:
        v.unlock(lockkey)
        v.val = new
        return True

    #  else:
    v.unlock(lockkey)
    return False


def atomic_add(v: lockable_var, add):
    """
    Atomic addition of a lockable_var.

    returns: updated lockable_var
    """

    done = False

    while not done:
        value = v.val
        # make sure that the value that we are adding to
        # is the same as it currently is in memory, and
        # update it only then; Then the atomic_add will
        # be done correctly.
        done = atomic_cas(v, value, value + add)

    return v


if __name__ == "__main__":

    a = lockable_var(7)
    if atomic_cas(a, 7, 8):
        print("Success: updated a=", a)
    else:
        print("Failure:", a)
    if atomic_cas(a, 7, 8):
        print("Success: updated a=", a)
    else:
        print("Failure:", a)

    print("Addition:", atomic_add(a, 10))
