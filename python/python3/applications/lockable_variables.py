#!/usr/bin/env python3

#-------------------------------------------------
# A class that allows certain attributes 
# (in this case: "val") lockable, i.e. 
# non-changeable until a lock is lifted by 
# the correct lock key.
#-------------------------------------------------

class LockError(Exception):
    pass

class LockWarning(Warning):
    pass

class lockable_var(object):

    def __init__(self, val):
        self.locked = False
        self.lockKey = None
        self.val = val
        return

    def __str__(self):
        return str(self.val) + "; lockable"

    def __repr__(self):
        return "lockable_var"

    def __setattr__(self, name, value):
        """
        Modify how we access/change certain
        attributes.
        """
        if name == "val":
            if self.locked:
                raise LockError("Trying to overwrite locked value")
            else:
                super().__setattr__(name, value)
        else:
            super().__setattr__(name, value)
        return

    def lock(self, lockKey = None):
        """
        Lock the variable.
        returns  True on success, False otherwise
            """
        if self.locked:
            raise LockError("trying to lock already locked variable")
            return False
        self.locked = True
        self.lockKey = lockKey
        return True

    def unlock(self, lockKey = None):
        """
        Lock the variable.
        returns  True on success, False otherwise
        """
        if not self.locked:
            raise LockWarning("trying to unlock already unlocked variable")
            return False
        if self.lockKey == lockKey:
            self.locked = False
            self.lockKye = None
            return True
        else:
            raise LockWarning("Can't unlock something somebody else locked. lockID =", lockKey, "var was locked by", self.lockKey)
            return False



if __name__ == "__main__":

    test = lockable_var(3)

    # test printing override
    print(test) 

    test.xyz = 421 # we can still treat it like a standard object and add attrs
    print(test.xyz)

    # test the locks
    test.lock()
    #  test.val = 7 # fails as intended
    test.unlock()
    test.val = 7

    # test locker ID
    test.lock(2)
    #  test.val = 8 # fails as intended
    #  test.unlock(3) # fails as intended
    test.unlock(2)
    test.val = 9
    print(test)
