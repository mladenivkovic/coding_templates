#!/usr/bin/env python3


# ======================
class global_params:
    # ======================
    """
    An object to store all global parameters in, so you can only
    pass 1 object to functions that need it.

    """

    # ======================
    def __init__(self):
        # ======================
        """
        Initialises object. 
        """
        self.halo = 0  # the root halo for which to plot for

        self.workdir = ""  # current working directory
        self.numfiles = 0  # number of files in this directory

        self.verbose = (
            False  # whether to print more details of what this program is doing
        )
        self.plotparticles = False  # whether to also create plots with particles

        # dictionnary of accepted keyword command line arguments
        self.accepted_args = {
            "-v": self.set_verbose,
            "-plotparticles": self.set_plotparticles,
            "-pp": self.set_plotparticles,
        }
        return

    # =============================
    def read_cmdlineargs(self):
        # =============================
        """
        Reads in the command line arguments and stores them in the
        global_params object.
        """
        from sys import argv

        nargs = len(argv)
        i = 1  # first cmdlinearg is filename of this file, so skip it

        while i < nargs:
            arg = argv[i]
            arg = arg.strip()
            if arg in self.accepted_args.keys():
                self.accepted_args[arg]()
            else:
                try:
                    self.halo = int(arg)
                except ValueError:
                    print("I didn't recognize the argument '", arg, "'")
                    quit()

            i += 1

        # defensive programming
        if self.halo <= 0:
            print("No or wrong halo given. Halo ID must be > 0")
            quit()

        return

    # ==========================
    def get_output_info(self):
        # ==========================
        """
        Read in the output info based on the files in the current
        working directory.
        Reads in last directory, ncpu, noutputs. 
        """

        from os import getcwd
        from os import listdir

        self.workdir = getcwd()
        filelist = listdir(self.workdir)

        self.numfiles = len(filelist)

        return

    # ========================
    # Setter methods
    # ========================

    def set_plotparticles(self):
        self.plotparticles = True
        return

    def set_verbose(self):
        self.verbose = True
        return


# ===============================
if __name__ == "__main__":
    # ===============================

    # -----------------------
    # Set up
    # -----------------------

    params = global_params()
    params.read_cmdlineargs()
    params.get_output_info()

    print("===============================================")
    print("Working parameters are:")
    print("halo:", params.halo)
    print("number of files:", params.numfiles)
    print("Verbose?", params.verbose)
    print("Particles will be plotted?", params.plotparticles)
    print("===============================================")
