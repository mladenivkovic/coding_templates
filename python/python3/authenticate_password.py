#!/usr/bin/env python3

# ==============================================
# Authenticate passwords.
#
# https://www.askpython.com/python/examples/storing-retrieving-passwords-securely
# ==============================================


import os
import hashlib
from getpass import getpass

PEPPER = "X"
HASH_ALGO = "sha256"
ENCODING = "utf-8"
ITERATIONS = 100_000


def create_password_hash(
    password,
    hash_algo=HASH_ALGO,
    encoding=ENCODING,
    salt=None,
    iterations=ITERATIONS,
    pepper=PEPPER,
):
    """
    generate a secure salted hash.
    """
    if salt is None:
        salt = os.urandom(16)
    hash_value = hashlib.pbkdf2_hmac(
        hash_algo, password.encode(encoding), salt, iterations
    )
    password_hash = salt + hash_value + pepper.encode(encoding)

    pwddata = {
        "salt": salt,
        "pepper": pepper,
        "hash_algo": hash_algo,
        "encoding": encoding,
        "iterations": iterations,
        "password_hash": password_hash,
    }

    return pwddata


def password_correct(passworddata1, passworddata2):

    return passworddata1["password_hash"] == passworddata2["password_hash"]


if __name__ == "__main__":

    password = "my_password"

    # First, generate the secure hash.
    pwddata = create_password_hash(password)

    attempts = 10
    a = 0
    while a < attempts:
        a += 1
        entry = getpass(f"[Attempt {a}/{attempts}] Password:")
        print(entry)

        entrydata = create_password_hash(
            entry,
            hash_algo=pwddata["hash_algo"],
            encoding=pwddata["encoding"],
            salt=pwddata["salt"],
            iterations=pwddata["iterations"],
            pepper=pwddata["pepper"],
        )

        if password_correct(pwddata, entrydata):
            print("Correct.")
            quit()
        else:
            print("Invalid password.")

        if a == attempts:
            print("Too many attempts.")
            quit()
