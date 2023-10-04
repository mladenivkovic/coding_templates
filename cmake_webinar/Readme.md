# CMake webinar

This repository contains the source material for a Archer2 EPCC webinar on CMake.

## Build instructions

You will need to install mkdocs-material. This can be installed as a python package

```bash
pip install mkdocs-material
```

The documentation can be built on the fly by running.

```bash
cd tutorial
mkdocs serve
```

This will start a local server on localhost and display the address at which the documentation is served. By default this will be on localhost. Copy the address and paste it in the address bar of your favorite browser in order to browse the lessons.

## Outline of live-coding

The webinar will mostly be presented as a live-coding session on Cirrus.


## Presentation

- Go trough the introduction in the Introduction pages.
- Go to the Hello World example and demonstrate the configure/build/install process
- Go to WindTunnel example and try to compile. Show how to see the list of variables and helper strings.
- Try to recompile `windtunnel` using the Intel compilers and Intel mpi. Demonstrate how to change the compilers and MPI.
- Try to recompile using the unwrapped MPI wrapper compiler and OpenMPI as the external compiler. The goal is to teach usage of `-DPACKAGE_NAME_ROOT` . Also show issues that can arise from the appropriate policy not being set.


### Tentative

- Go to the MPI Hello World example. The main goal is to give the concept of a target and target properties.

## Using in Docker

You should serve the container on url 0.0.0.0 ( instead of localhost )

```bash
mkdocs serve -a 0.0.0.0:8887  
```
