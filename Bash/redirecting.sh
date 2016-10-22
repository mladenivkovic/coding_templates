#!/bin/bash


f() {
    echo "hi"
    asd
}


f 2>stderrlog | tee stdoutlog
