# ~/.bashrc: executed by bash(1) for non-login shells.
# see /usr/share/doc/bash/examples/startup-files (in the package bash-doc)
# for examples
# there is more in the ~/.profile file.


#===================
# GREETING MESSAGE
#===================


    # Colors for echo output.
    # Use with echo -e "$COL_NAME" Your text here "$COL_RESET"
    ESC_SEQ="\x1b["
    COL_RESET=$ESC_SEQ"39;49;00m"
    COL_RED=$ESC_SEQ"31;01m"
    COL_GREEN=$ESC_SEQ"32;01m"
    COL_YELLOW=$ESC_SEQ"33;01m"
    COL_BLUE=$ESC_SEQ"34;01m"
    COL_MAGENTA=$ESC_SEQ"35;01m"
    COL_CYAN=$ESC_SEQ"36;01m"
    COL_CYAN_ITALIC=$ESC_SEQ"36;03m"
    COL_NORMAL_ITALIC=$ESC_SEQ"39;03m"
    COL_DARK_ITALIC=$ESC_SEQ"2;03m"


    # Greetings
    case $- in *i*) # check if interactive, otherwise scp will crash
        echo -e "$COL_CYAN_ITALIC""  Good luck and have fun!" "$COL_RESET"
        # echo -e "$COL_DARK_ITALIC""  type 'showvars', 'showaliases' and 'showfuncs' to see variables,\n  aliases and functions defined in the ~/.bashrc file." "$COL_RESET"
        ;;
    esac



#===========================
# PATH VARIABLE
#===========================

    # Add executables from /home/mivkov/scripts/execs_for_path to $PATH variable
    PATH="$HOME"/"scripts/execs_for_path:"$PATH

    # Add intel (fortran) compiler to path var
    #PATH="/opt/intel/bin:"$PATH

    #add local bin
    PATH="$HOME""/local/bin:"$PATH

    export APPDIR="$HOME/local/"

    #add downloaded programs
    PATH="$APPDIR:"$PATH
    PATH="$APPDIR/briss-0.9":"$PATH"
    PATH="$APPDIR/music":"$PATH"
    PATH="$APPDIR/ramses/bin":"$PATH"
    PATH="$APPDIR/dice/bin":"$PATH"
    PATH="$APPDIR/Gadget-2.0.7/Gadget2":"$PATH"
    PATH="$APPDIR/GadgetConverter/build/rundir":"$PATH"
    PATH="$APPDIR/Gear/src":"$PATH"


    export SPACK_ROOT="$APPDIR/spack/"
    PATH="$SPACK_ROOT/bin":"$PATH"

    export PATH


    MANPATH=/usr/share/man


#===========================
# ENVIRONMENT VARIABLES
#===========================

    export CXX=/usr/bin/gcc-8
    export F77=/usr/bin/gfortran-8
    export F90=/usr/bin/gfortran-8
    export F95=/usr/bin/gfortran-8




#=======================
# LMOD STUFF
#=======================

	# NEEDS TO BE DONE AFTER PATH ADDITIONS AND BEFORE EVERYTHING ELSE
    if [ -f ~/.bashrc_modules ]; then
        . ~/.bashrc_modules
    fi







#================
# HISTORY
#=================


    # don't put duplicate lines or lines starting with space in the history.
    # See bash(1) for more options
    HISTCONTROL=ignoreboth

    # append to the history file, don't overwrite it
    shopt -s histappend

    # for setting history length see HISTSIZE and HISTFILESIZE in bash(1)
    HISTSIZE=2000
    HISTFILESIZE=10000






#============================
# PROMPT
#============================


    # Set light or dark theme
    # export MYPROMPTCOLOR='light'
    # export MYPROMPTCOLOR='dark'
    export MYPROMPTCOLOR='ubuntu18_default_dark'

    parse_git_branch() {
        # get current git branch, if any
        git branch 2> /dev/null | sed -e '/^[^*]/d' -e 's/* \(.*\)/ (\1)/'
    }

    prompt() {
        # This function prepares the prompt properly
        # depending on the MYPROMPTCOLOR variable.

        col_reset="\[\033[39;49;0m\]"

        # set colors
        if [[ ${MYPROMPTCOLOR} == 'light' ]]; then
            col_left="\[\033[37;104;1m\]"
            col_right="\[\033[31;1m\]" # 36 for green
            col_nline="\[\033[94;1m\]"
            compensate=31
        elif [[ ${MYPROMPTCOLOR} == 'dark' ]]; then
            col_left="\[\033[33m\]"
            col_right="\[\033[32;1m\]"
            col_nline="\[\033[32;1m\]"
            compensate=31
        elif [[ ${MYPROMPTCOLOR} == 'ubuntu18_default_dark' ]]; then
            col_left="\[\033[32;1m\]"
            col_right="\[\033[31;1m\]"
            col_nline="\[\033[32;1m\]"
            compensate=31
        else
            # DEFAULTS OF SOME KIND
            col_left="\[\033[1m\]"
            col_right="\[\033[1m\]" # 36 for green
            col_nline="\[\033[1m\]"
            compensate=31

        fi

        # GET TEXT
        left="${col_left}\A [\u@\h] - \w${col_reset}"
        right="${col_right}"$(parse_git_branch)"${col_reset}"
        nextline="${col_nline}"\$"${col_reset}"
        linelen="$(($(tput cols) + compensate))"

        PS1=$(printf "%*s\r%s\n %s %s" "${linelen}" "${right}" "${left}" "${nextline}" )
    }
    PROMPT_COMMAND=prompt







#==================
# VARIABLES
#==================

    export TERM=xterm-256color
    export OR="$APPDIR/ramses/"



#============================
# ALIASES
#============================

    #----------
    # SSH
    #----------

    alias linux.physik='ssh -2Y mivkov@linux.physik.uzh.ch'
    alias phy='ssh -2Y mivkov@linux.physik.uzh.ch'
    alias malin1='ssh -2Y mivkov@malin1.physik.uzh.ch'
    alias ssch='ssh -Yt stud59@ela.cscs.ch ssh -Y daint'





    #---------------------
    # directory shortcuts
    #---------------------


    # GLOBALS
    alias ..='cd ..'
    alias ~='cd ~'

    
    #UNI
    alias papers='cd ~/UZH/Papers'
    alias or="cd $OR"





    #---------
    # Others
    #---------

    alias e='exit'
    alias python=python3
    alias refresh='source ~/.bashrc'
    alias fucking=sudo
    alias vimh="gedit $HOME/Coding/coding_templates/notes/vim-notizen.txt"
    alias jn='jupyter-notebook'
    alias v='firefox build/html/index.html'

    alias pmake='python setup.py build && python setup.py install --user'
    alias pcomp='python -m compileall'






    #-----------------------------------
    # COLOR SUPPORT FOR INBUILT PROGRAMS
    #-----------------------------------

    # enable color support of ls and also add handy aliases
    if [ -x /usr/bin/dircolors ]; then
        test -r ~/.dircolors && eval "$(dircolors -b ~/.dircolors)" || eval "$(dircolors -b)"
        alias ls='ls --color=auto'
        #alias dir='dir --color=auto'
        #alias vdir='vdir --color=auto'

        alias grep='grep --color=auto'
        alias fgrep='fgrep --color=auto'
        alias egrep='egrep --color=auto'
    fi

    # some more ls aliases
    alias ll='ls -alh'
    alias la='ls -a'
    alias l='ls -CF'







#===========================
# FUNCTIONS
#===========================

    #-------------------------------------------------------
    # functions to show specified content of ~/.bashrc file
    #-------------------------------------------------------

    #---------------------
    function showaliases { 
    #---------------------
        #function to print all my self defined aliases in the ~/.SHELLrc file.
        shell_in_use=$0
        echo " "
        echo "Printing all aliases defined in the ".$shell_in_use"rc file:"
        echo " "
        sed -e 's/^\s*//' -e '/^$/d' $HOME/."$shell_in_use"rc | egrep '^[[:blank:]]*alias'
        echo " "
    }



    #---------------------
    function showvars { 
    #---------------------
        #function to print all my self defined and exported variables in the ~/.SHELLrc file.
        shell_in_use=$0
        echo " "
        echo "Printing all exported variables defined in the ".$shell_in_use"rc file:"
        echo " "
        sed -e 's/^\s*//' -e '/^$/d' $HOME/."$shell_in_use"rc | egrep '^[[:blank:]]*export'
        echo " "
    }




    #---------------------
    function showfuncs { 
    #---------------------
        #function to print all my self defined functions in the ~/.bashrc file.
        shell_in_use=$0
        echo " "
        echo "Printing all functions defined in the ".$shell_in_use"rc file:"
        echo " "
        sed -e 's/^\s*//' -e '/^$/d' $HOME/."$shell_in_use"rc | egrep '^[[:blank:]]*function'
        echo " "
    }





    #------------------------
    # Conglomerate functions
    #------------------------


    #----------------
    function space { 
    #----------------
        # Shows workdir size 
        echo "Working directory: " $PWD
        echo "usage of working directory" `du -h 2>>/dev/null | tail -n1`
        echo "use du -h for further details."

        echo ""
        echo "Data on main partition:"
        df -h | head -n1
        df -h | grep nvme0n1p5 --colour=never
        echo "use df -h for further details."
    }

    alias disk=space



    function gitup {
        # a function to check out the master branch, pull,
        # then go back to the original branch and rebase
        mybranch=`git rev-parse --abbrev-ref HEAD`
        git checkout master && git pull && git checkout $mybranch && git rebase master
        return 0
    }





#=====================
# PYTHON
#=====================

    # PYTHONPATH=${PYTHONPATH}"/home/mivkov/Coding/projekte/my_python_modules/physics:"
    # PYTHONPATH=${PYTHONPATH}"/usr/local/lib/python2.7/site-packages:/usr/lib/python2.7/site-packages:"
    # # export PYTHONPATH=${PYTHONPATH}"/home/mivkov/applications/ParaView/lib/python2.7/site-packages:"
    #
    # # PYTHONPATH=${PYTHONPATH}"$APPDIR/pNbody/Doc/sphinxext/lib/python3.6/site-packages:"
    # PYTHONPATH=${PYTHONPATH}"$APPDIR/pNbody/Doc/sphinxext/lib/python2.7/site-packages:"
    # PYTHONPATH=${PYTHONPATH}"$APPDIR/python_libs/lib/python2.7/site-packages:"
    # PYTHONPATH=${PYTHONPATH}"$APPDIR/python_libs/lib/python3.6/site-packages:"
    #
    # export PYTHONPATH
    # export PYTHONLIBS="$APPDIR/python_libs/"









#=====================
# Other RC files
#=====================


    #-------------------------
    # Library Configuration
    #-------------------------

    #openMP stuff
    if [ -f ~/.bashrc_openmp ]; then
        . ~/.bashrc_openmp
    fi

    #CUDA stuff
    # if [ -f ~/.bashrc_cuda ]; then
    #     . ~/.bashrc_cuda
    # fi




    #-------------------------
    # Specific Projects
    #-------------------------


    #include ramses shortcuts
    #if [ -f ~/.bashrc_bachelor ]; then
    #    . ~/.bashrc_bachelor
    #fi

    #Computational Astrophysics
    # if [ -f ~/.bashrc_compast ]; then
    #     . ~/.bashrc_compast
    # fi

    #My small coding projects
    if [ -f ~/.bashrc_coding ]; then
        . ~/.bashrc_coding
    fi

    #masterarbeit stuff
    if [ -f ~/.bashrc_master ]; then
        . ~/.bashrc_master
    fi

    #masterarbeit stuff
    if [ -f ~/.bashrc_phd ]; then
        . ~/.bashrc_phd
    fi


    #SWIFT stuff
    if [ -f ~/.bashrc_swift ]; then
        . ~/.bashrc_swift
    fi





#=================================
# Small Misc Stuff
#=================================

    shopt -s direxpand # expand variables fully. Needed to prevent fuckery like expanding $HOME to \$HOME
