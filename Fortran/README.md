Last update: 04.07.2016


# THIS DIRECTORY CONTAINS: #


### arrays.f90 ###
    initiating and printing arrays.
    - initiate arraym, allocatable array
    - dimensions
    - allocating arrays
    - accessing and printing multidimensional arrays
    - filling array with input from a file


### complex.f90 ###
    complex numbers itrinistics in fortran.
    - arithmetics
    - real part, imaginary part, conjugate, abs


### CONTENTS.TXT
    this file.


### functions.f90 ###
    defining a function in fortran
    defining a recursive function


### if-else-case.f90 ###
    How to use if, else, and case statements


### inputfiles/*
    Contains files that the programs use.


### linkedlist_in_parts.f90 ###
    This program shows how to create a linked list that is appended to with
    "breaks" in between: Append your "object" to whatever list
    it's supposed to be appended, while the order you "read them in" is
    unknown. (But you know where they should be sorted into.)


### linkedlist_reals.f90 ###
    reads in an unknown number of reals from a file.


### linkedlist.f90 ###
    how to implement a singly linked list.


### logicaloperators.f90 ###
    logical operators: AND, OR, AND.NOT, EQV, NEQV, NOT


### loops.f90 ###
    DO, GENERAL DO, DO WHILE (for implied do, see arrays.f90)


### make/ ###
    How to compile a (bigger) program using a makefile.
    Contains the same files as modules/

    Makefile
        - the actual makefile.

    modules.f90 ###
        - a program to demonstrate how modules work.

    physical_constants.f90 ###
        - module containing physical constants

    precision_specification.f90 ###
        - module specifying the precision of reals
   
    simple_math_module.f90 ###
        - module containing simple math subroutines for demonstration.

### makedir_and_file.f90 ###
    Create a directory and a file within that directory.


### modules/
    modules.f90 ###
        - a program to demonstrate how modules work.

    physical_constants.f90 ###
        - module containing physical constants

    precision_specification.f90 ###
        - module specifying the precision of reals
   
    simple_math_module.f90 ###
        - module containing simple math subroutines for demonstration.


### MPI/
    helloworld_mpi.f90 ###
        - Initialise MPI; each processor says hello.

    mapdomain2d-2.f90 ###
        - Create a "processor map": Split a domain in squares; Then assign
        each processor its neighbours.
        Lowest rank is bottom left, rank increases to the right and upwards.

    mapdomain2d.f90 ###
        - Create a "processor map": Split a domain in squares; Then assign
        each processor its neighbours.
        Lowest rank is upper left, rank increases to the right and downwards.

    mpi_sendarraysforhydrocode.f90 ###
        Send the last two columns of an array to the right neighbour
        and the first two columns the left neighbour.
        receive first two domain columns (first two rows/columns are 
        ghost cells) from left neighbour and receive last two domain
        columns from right neighbour.

    optimal_processor_distribution.f90 ###
        calculate the optimal rectangular processor distribution
        by minimising communication time.

    reduce.f90 ###
        Demonstration of MPI_REDUCE.
        Calculates pi with the MPI reduce operation.
        This is done by integrating  INTEGRAL 4 * 1 / (1 + x^2) FROM 0 TO 1 = 4 * atan(1) = pi.
        It also times the process.

    sendpartofmultidimarray.f90 ###
        sending parts of multidimensional arrays using derived MPI types.

    sendreceive.f90 ###
        Demonstrates MPI_SENDRCV()

    title.f90 ###
        A subroutine which converts an integer to a character
        of precisely len=5 and fills it with zeros if necessary.
        
    write_mpi.f90 ###
        - stdout only from 1 processor
        - writing everything in exaclty 1 file in the correct order


### namelists/

    How to create and read in from namelists.
    Contains a big and small example.

    big_example/
        input.nml
        module_datastuff.f90 ###
        module_numeric.f90 ###
        namelist.f90 ###

    small_example/
        input.nml
        smallnamelistexample.f90 ###


### notes_and_docs/

    Contains various documentations and notes.

    Fortran-Notizen.txt
    gfortran.pdf
    links.txt


### NumericInquiryFunctions/
    NumericInquiryFunctions-Integers.f90 ###
    NumericInquiryFunctions-Integers-Output.txt
    NumericInquiryFunctions-Reals.f90 ###
    NumericInquiryFunctions-Reals-Output.txt


### pointers.f90 ###
    - What pointers are
    - associate and disassociate pointers
    - place in memory


### preprocessingoptions.f90 ###
    preprocessing and preprocessing options.


### printing.f90 ###
    - using the print method
    - format statements


### random.f90 ###
    - generating random reals

    
### readfromfile.f90 ###
    how to read in from files.


### subroutines.f90 ###
    - Fortran subroutines
    - intent
    - passing arrays
    - optional keywords


### timing.f90 ###
    time cpu-time


### timing_module.f90 ###
    a module that implements time measurement in subroutines.
    (is used by MPI/reduce.f90)


### types.f90 ###
    creating and using your own types.


### write.f90 ###
    - using the write(*,*) method
    - formatted output with editor descriptors


### writingtofile.f90 ###
    how to write to a file instead of stdout
