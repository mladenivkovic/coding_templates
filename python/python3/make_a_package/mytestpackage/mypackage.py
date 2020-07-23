#!/usr/bin/env python3


def greetings():

    print("Hello there! You successfully called the greetings funciton.")

    return


def test_other_file():

    from .someotherfile import hi_from_other_file

    hi_from_other_file()

    return


if __name__ == "__main__":
    greetings()
    test_other_file()
