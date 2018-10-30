# ~/.bashrc: executed by bash(1) for non-login shells.
# see /usr/share/doc/bash/examples/startup-files (in the package bash-doc)
# for examples
# there is more in the ~/.profile file.




#=================
# COLORS
#=================

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


    #Blue = 34
    #Green = 32
    #Light Green = 1;32
    #Cyan = 36
    #Red = 31
    #Purple = 35
    #Brown = 33
    #Yellow = 1;33
    #white = 1;37
    #Light Grey = 0;37
    #Black = 30
    #Dark Grey= 1;30


    #0   = default colour
    #1   = bold
    #4   = underlined
    #5   = flashing text
    #7   = reverse field
    #40  = black background
    #41  = red background
    #42  = green background
    #43  = orange background
    #44  = blue background
    #45  = purple background
    #46  = cyan background
    #47  = grey background
    #100 = dark grey background
    #101 = light red background
    #102 = light green background
    #103 = yellow background
    #104 = light blue background
    #105 = light purple background
    #106 = turquoise background
#






#=================
# GREETING MESSAGE
#=================

    # Greetings
    case $- in *i*) # check if interactive, otherwise scp will crash
        echo -e "$COL_CYAN_ITALIC""  Good luck and have fun!" "$COL_RESET"
        echo -e "$COL_DARK_ITALIC""  type 'showvars', 'showaliases' and 'showfuncs' to see variables,\n  aliases and functions defined in the ~/.bashrc file." "$COL_RESET"
        ;;
    esac







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
# VARIABLES
#============================

    #---------------------
    # PROMPT
    #---------------------

    export NEWLINE=$'\n'

    #FOR DARK THEME
    #PS1="\[\e[40;34m\]\A [\u@\h] - \w  \[\e[m\]${NEWLINE}"
    # PS1="\[\e[33m\]\A @ \w  \[\e[m\]${NEWLINE}"
    # export PS1="$PS1"'  ' # space around newline did something weird with colors.

    #FOR LIGHT THEME
    PS1="\[\e[37;104;1m\]\A [\u@\h] - \w \[\e[0m\] ${NEWLINE}"
    export PS1="$PS1"'  ' # space around newline did something weird with colors.






    #----------------
    # PATH VARIABLE
    #----------------

    # Add executables from /home/mivkov/Skripte/execs_for_path to $PATH variable
    PATH="$HOME"/"scripts/execs_for_path:"$PATH

    # Add intel (fortran) compiler to path var
    #PATH="/opt/intel/bin:"$PATH

    #add local bin
    PATH="$HOME""/local/bin:"$PATH

    #add downloaded programs
    PATH="$HOME""/Programme:"$PATH
    PATH="$HOME""/Programme/briss-0.9":"$PATH"

    export PATH



    #------------
    # Others
    #------------

    export TERM=xterm-256color
    export OR='/home/mivkov/UZH/ramses/'



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

    # MY CODING AND PROJECTS
    alias pj='cd ~/Coding/projekte'                                 # go to projects
    alias templ='cd ~/Coding/coding_templates'                      # go to templates
    alias t=templ
    alias ft='cd ~/Coding/coding_templates/Fortran'                 # go to fortran templates
    alias pt='cd ~/Coding/coding_templates/Python/python3'          # go to python templates
    alias ct='cd ~/Coding/coding_templates/C'                       # go to C templates
    alias cppt='cd ~/Coding/coding_templates/cpp'                   # go to C++ templates
    alias ppt='cd ~/Coding/coding_templates/Python/python3/plots'   # go to python plot templates
    alias bt='cd ~/Coding/coding_templates/Bash'                    # go to bash templates
    alias lt='cd ~/Coding/coding_templates/LaTeX'                   # go to LaTeX templates
    alias rcf='cd ~/Coding/coding_templates/rc-files'               # go to LaTeX templates
    alias plg='cd ~/Coding/Playground'
    alias f='cd ~/Coding/projekte/formelsammlung'                   # go to formelsammlung
    alias g='cd ~/Coding/projekte/glossar'                          # go to formelsammlung


    #UNI
    alias papers='cd ~/UZH/Papers'
    alias or='cd $OR'
    # alias fw='cd ~/Public/Fortran_Workshop/Exercises'
    #alias d='cd ~/UZH/Introduction_to_Data_Science/esc403'
    alias sw='cd ~/EPFL/swiftsim'





    #---------
    # Others
    #---------

    alias e='exit'
    alias python=python3
    alias refresh='source ~/.bashrc'
    alias fucking=sudo









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









#=====================
# PYTHON
#=====================

    export PYTHONPATH=${PYTHONPATH}"/home/mivkov/Coding/projekte/my_python_modules/physics:"
    export PYTHONPATH=${PYTHONPATH}"/usr/local/lib/python2.7/site-packages:/usr/lib/python2.7/site-packages:"
    # export PYTHONPATH=${PYTHONPATH}"/home/mivkov/Programme/ParaView/lib/python2.7/site-packages:"











#=====================
# Other RC files
#=====================

    #include ramses shortcuts
    #if [ -f ~/.bashrc_ramses ]; then
    #    . ~/.bashrc_ramses
    #fi

    #openMP stuff
    if [ -f ~/.bashrc_openmp ]; then
        . ~/.bashrc_openmp
    fi

    #CUDA stuff
    if [ -f ~/.bashrc_cuda ]; then
        . ~/.bashrc_cuda
    fi


    #masterarbeit stuff
    if [ -f ~/.bashrc_master ]; then
        . ~/.bashrc_master
    fi


    #SWIFT stuff
    if [ -f ~/.bashrc_swift ]; then
        . ~/.bashrc_swift
    fi


    #Computational Astrophysics
    # if [ -f ~/.bashrc_compast ]; then
    #     . ~/.bashrc_compast
    # fi




#====================
# MISC
#====================

    #ls coloring
    #di = directory
    #fi = file
    #ln = symbolic link
    #pi = fifo file
    #so = socket file
    #bd = block (buffered) special file
    #cd = character (unbuffered) special file
    #or = symbolic link pointing to a non-existent file (orphan)
    #mi = non-existent file pointed to by a symbolic link (visible when you type ls -l)
    #ex = file which is executable (ie. has 'x' set in permissions).
    #*.rpm = files with the ending .rpm
    #LS_COLORS=$LS_COLORS:'di=1;34:' ; export LS_COLORS


    #disable vim C-s freezeout (solve with C-q)
    #stty -ixon


# Add an "alert" alias for long running commands.  Use like so:
#   sleep 10; alert
#alias alert='notify-send --urgency=low -i "$([ $? = 0 ] && echo terminal || echo error)" "$(history|tail -n1|sed -e '\''s/^\s*[0-9]\+\s*//;s/[;&|]\s*alert$//'\'')"'

# Alias definitions.
# You may want to put all your additions into a separate file like
# ~/.bash_aliases, instead of adding them here directly.
# See /usr/share/doc/bash-doc/examples in the bash-doc package.

#if [ -f ~/.bash_aliases ]; then
#    . ~/.bash_aliases
#fi

# enable programmable completion features (you don't need to enable
# this, if it's already enabled in /etc/bash.bashrc and /etc/profile
# sources /etc/bash.bashrc).
#if ! shopt -oq posix; then
#  if [ -f /usr/share/bash-completion/bash_completion ]; then
#    . /usr/share/bash-completion/bash_completion
#  elif [ -f /etc/bash_completion ]; then
#    . /etc/bash_completion
#  fi
#fi



