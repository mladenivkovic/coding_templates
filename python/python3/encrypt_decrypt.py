#!/usr/bin/env python3

# ==========================================
# Encrypt and decrypt data.
# ==========================================

import os
from base64 import b64encode
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
import Crypto.Protocol.KDF


ENCODING = "utf-8"
ITERATIONS = 100000
SALT = os.urandom(16)

data = b"secret"
key = b"this is my key with a very long length."


# The key needs to multiple of 16 in length.
# So we need a Key Derivation Function.


def derive_key(key, salt=SALT, iterations=ITERATIONS):

    key_derived = Crypto.Protocol.KDF.PBKDF2(
        key, salt, dkLen=16, count=iterations, prf=None, hmac_hash_module=None
    )

    return key_derived


key_derived = derive_key(key)


# -----------------
# Encode data
# -----------------

cipher_encrypt = AES.new(key_derived, AES.MODE_CFB, iv=b"vector of size16")
encrypted_data = cipher_encrypt.encrypt(data)

# encode it to be written down.
iv = b64encode(cipher_encrypt.iv).decode("utf-8")
ed = b64encode(encrypted_data).decode("utf-8")

result = {"iv": iv, "ciphertext": ed}

print("Encrypted:", result)


# -----------------
# Decoding again
# -----------------

try:
    # decode 'read-in' data
    iv_decode = b64decode(iv)
    ct_decode = b64decode(ed)
    cipher = AES.new(key_derived, AES.MODE_CFB, iv=iv_decode)
    pt = cipher.decrypt(ct_decode)
    print("The message was: ", pt)

except (ValueError, KeyError):
    print("Incorrect decryption")
