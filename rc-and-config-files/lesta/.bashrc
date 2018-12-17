# Some applications read the EDITOR variable to determine your favourite text
# editor. So uncomment the line below and enter the editor of your choice :-)
export EDITOR=/usr/bin/vim
#export EDITOR=/usr/bin/mcedit

test -s ~/.alias && . ~/.alias || true
# ~/.bashrc: executed by bash(1) for non-login shells.
# see /usr/share/doc/bash/examples/startup-files (in the package bash-doc)
# for examples
# there is more in the ~/.profile file.



export TERM=xterm-256color


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






#=================
# GREETING MESSAGE
#=================

    # Greetings

    if [[ "$HOSTNAME" = "lesta"* ]]; then
        case $- in *i*)
            echo -e "$COL_RED""  You're on lesta" "$COL_RESET"
            ;;
        esac
    fi

    if [[ "$HOSTNAME" = "unige"* ]]; then
        case $- in *i*)
            echo -e "$COL_RED""  You're on unige" "$COL_RESET"
            ;;
        esac
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
HISTSIZE=1000
HISTFILESIZE=2000







#===================================
# COLOR SUPPORT FOR INBUILT PROGRAMS
#===================================

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
    alias la='ls -A'
    alias l='ls -CF'



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
    LS_COLORS=$LS_COLORS:'di=1;34:' ; export LS_COLORS

#============================
# VARIABLES
#============================


    # Set light or dark theme
    # export MYPROMPTCOLOR='light'
    # export MYPROMPTCOLOR='dark'
    # export MYPROMPTCOLOR='ubuntu18_default_dark'
    export MYPROMPTCOLOR='ssh'

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
        elif [[ ${MYPROMPTCOLOR} == 'ssh' ]]; then
            col_left="\[\033[33;1m\]"
            col_right="\[\033[31;1m\]"
            col_nline="\[\033[33;1m\]"
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







#============================
# ALIASES
#============================

    #SSH
    alias lesta='ssh  lesta'


    #directory shortcuts
    alias ..='cd ..'
    alias ~='cd ~'


    #Others
    alias e='exit'
    alias refresh='source ~/.bashrc'



#===========================
# FUNCTIONS
#===========================


    #functions to show specified content of ~/.bashrc file

    function showaliases { #function to print all my self defined aliases in the ~/.SHELLrc file.
        shell_in_use=$0
        echo " "
        echo "Printing all aliases defined in the ".$shell_in_use"rc file:"
        echo " "
        sed -e 's/^\s*//' -e '/^$/d' $HOME/."$shell_in_use"rc | egrep '^[[:blank:]]*alias'
        echo " "
    }

    function showvars { #function to print all my self defined and exported variables in the ~/.SHELLrc file.
        shell_in_use=$0
        echo " "
        echo "Printing all exported variables defined in the ".$shell_in_use"rc file:"
        echo " "
        sed -e 's/^\s*//' -e '/^$/d' $HOME/."$shell_in_use"rc | egrep '^[[:blank:]]*export'
        echo " "
    }

    function showfuncs { #function to print all my self defined functions in the ~/.bashrc file.
        shell_in_use=$0
        echo " "
        echo "Printing all functions defined in the ".$shell_in_use"rc file:"
        echo " "
        sed -e 's/^\s*//' -e '/^$/d' $HOME/."$shell_in_use"rc | egrep '^[[:blank:]]*function'
        echo " "
    }

    #===========================


    function space { # Zeigt Grösse des Arbeitsverzeichnisses
        echo "Arbeitsverzeichnis: " $PWD
        echo "Grösse des Arbeitsverzeichnisses " `du -h 2>>/dev/null | tail -n1`
        echo "Benutze du -h für weitere Informationen."

        echo ""
        echo "Informationen zur Hauptpartition:"
        df -h | head -n1
        df -h | grep nvme0n1p5 --colour=never
        echo "Benutze df -h für weitere Informationen."
    }

    alias disk=space





#=====================
# PYTHON
#=====================


#=====================
# Other RC files
#=====================


    #masterarbeit stuff
    # if [ -f ~/.bashrc_master ]; then
    #     . ~/.bashrc_master
    # fi




    case $- in *i*) # check if interactive, otherwise scp will crash

        if [[ "$HOSTNAME" = "lesta"* ]]; then
            if [ -f ~/.bashrc_lesta ]; then
                . ~/.bashrc_lesta
            fi
        fi
        ;;

    esac
