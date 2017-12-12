# Created by newuser for 5.1.1

#=================
# COLORS
#=================

ESC_SEQ="\x1b["
COL_RESET=$ESC_SEQ"39;49;00m"
COL_RED=$ESC_SEQ"31;05m"
COL_GREEN=$ESC_SEQ"32;055555m"
COL_GREEN_ITALIC=$ESC_SEQ"32;03m"
COL_YELLOW=$ESC_SEQ"33;05m"
COL_BLUE=$ESC_SEQ"34;05m"
COL_MAGENTA=$ESC_SEQ"35;05m"
COL_CYAN=$ESC_SEQ"36;05m"
COL_NORMAL_ITALIC=$ESC_SEQ"39;03m"

echo -e "$COL_GREEN_ITALIC"" Good luck and have fun!" "$COL_RESET"
echo -e "$COL_BLUE""    You are using zsh." "$COL_RESET" 


# prompt
export NEWLINE=$'\n'
export PROMPT="%F{blue}%T [%n@%m] - %d %f ${NEWLINE}  "
#export RPROMPT="%F{blue}%T %f"


