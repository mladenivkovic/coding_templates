#!/usr/bin/env python3


# =======================
# Formatting
# =======================


print(
    "{0:17} | {1:12} {2:12}".format(
        "strings", "String1", "this_string_is_too_long_and_will_just_keep_going"
    )
)
print("{0:17} | {1:12} {2:12d}".format("ints", "1234".zfill(7), 123))
print("{0:17} | {1:12} {2:12}".format("bools", True, str(False)))
print("{0:17} | {1:12.4f} {2:12.1f}".format("flts", 12.9, 18.7))
print("{0:17} | {1:12.4E} {2:12.3E}".format("flts, scientific", 1223420.9, 18.7998234))
