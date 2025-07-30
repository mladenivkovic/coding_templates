#!/usr/bin/env python3

import subprocess


# Simple case where everything goes right
cmd = "echo 'hello world!'"
echo = subprocess.run(
                cmd,
                shell=True,
                check=True,
                capture_output=True
            )
stdout = echo.stdout
if isinstance(stdout, bytes):
    stdout = stdout.decode("utf8")
stdout = stdout.strip() # comes with newline, so let's get rid of that

stderr = echo.stderr
if isinstance(stderr, bytes):
    stderr = stderr.decode("utf8")
stderr = stderr.strip() # comes with newline, so let's get rid of that

print("Example 1:")
print("stdout was:", stdout)
print("stderr was:", stderr)


# Capture stderror too
cmd = "echo 'hello world!'; >&2 echo 'this is an error' "
echo_err = subprocess.run(
                cmd,
                shell=True,
                check=True,
                capture_output=True
            )
stdout = echo_err.stdout
if isinstance(stdout, bytes):
    stdout = stdout.decode("utf8")
stdout = stdout.strip() # comes with newline, so let's get rid of that

stderr = echo_err.stderr
if isinstance(stderr, bytes):
    stderr = stderr.decode("utf8")
stderr = stderr.strip() # comes with newline, so let's get rid of that

print("Example 2:")
print("stdout was:", stdout)
print("stderr was:", stderr)


# Handle non-zero exit
cmd = "echo 'hello world!'; exit 1 "
try:
    echo_err = subprocess.run(
                cmd,
                shell=True,
                check=True,
                capture_output=True
            )
except subprocess.CalledProcessError:
    print("Example 3:")
    print("Non-zero exit with code:", echo_err.returncode)


