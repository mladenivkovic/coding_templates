"================
" PLUGINS
"================
    "must be first.
    source ~/.vimrc_plugins






"=================
" VIM GENERAL
"=================

    " Enable mouse support in console
    set mouse=a

    " set backup
    " Don't forget to create these directories first if you're on a new
    " machine!
    set backup
    set backupdir=~/.vim/backup
    set directory=~/.vim/tmp

    " Split screen navigation key mapping change
    nnoremap <C-J> <C-W><C-J>
    nnoremap <C-K> <C-W><C-K>
    nnoremap <C-L> <C-W><C-L>
    nnoremap <C-H> <C-W><C-H>

    " enable 'intelligent' pasting
    "set paste " conflicts with YouCompleteMe

    " enable copypasting between sessions
    " if this doesn't work, check whether you
    " installed the right vim: vim --version | grep clipboard
    " must contain +clipboard or +xterm_clipboard.
    " (@ Ubuntu 16.04, you need vim-gnome)
    set clipboard=unnamedplus

    " make backspace work 'normally'
    set backspace=indent,eol,start 

    " tab completion in command line
    set wildmode=longest,list ",full
    set wildmenu

    "Line Numbers
    set number
    set relativenumber

    " this enables 'visual' wrapping
    set wrap

    " line wrap and other stuff
    set linebreak
    set nolist

    " this turns off physical line wrapping (ie: automatic insertion of newlines)
    set textwidth=0 
    set wrapmargin=0

    " line wrap with indent kept
    set breakindent  

    " moving along visual lines, not physical lines
    noremap  <buffer> <silent> k gk
    noremap  <buffer> <silent> j gj
    noremap  <buffer> <silent> 0 g0
    noremap  <buffer> <silent> $ g$

    " Allow to store file specific options as comments in file
    " WARNING: Potential security risk
    " set modeline
    " set modelines=5

    " Who doesn't like autoindent?
    set autoindent








"=================
" SEARCHING
"=================

    " Ignoring case
    set ignorecase

    " Incremental searching
    set incsearch

    " Highlight things that we find with the search
    set hlsearch

    set grepprg=grep\ -nH\ $*







"===================
" FOLDING
"===================

    " Enable folding
    set foldmethod=indent
    set foldlevel=1
    set foldminlines=5
    set foldnestmax=3
    set foldenable

    " Set fold methods for every file type only! 
    " let g:vimsyn_folding='af'
    " let g:tex_fold_enabled=1
    " let g:python_folding = 1
    " let python_folding = 1
    " let g:fortran_folding = 1
    " let g:bash_folding = 1
    " let g:c_folding = 1
    "let g:xml_syntax_folding = 1
    "let g:php_folding = 1
    "let g:perl_fold = 1



    " Enable folding with the spacebar
    nnoremap <space> za

    " Save/autoload folding when writing/loading file
    augroup remember_folds
      autocmd!
      autocmd BufWinLeave *.* mkview
      autocmd BufWinEnter *.* loadview
    augroup END










"=====================
" SYNTAX AND FILETYPE
"=====================

    " Needed for Syntax Highlighting and stuff
    filetype on 
    filetype plugin on "needed for plugins: nerdcommenter and to autoload .vim/after/ftplugin
    filetype indent off "needed by plugin slim

    " Fortran highlighting
    let fortran_free_source=1
    let fortran_have_tabs=0
    let fortran_more_precise=0
    set redrawtime=10000    " sometimes syntax highlight breaks for large fortran files

    " needs to be after fortran definitions
    syntax on
    set re=1










"==========================
" COLORS AND HIGHLIGHTING
"==========================

    set t_Co=16


    " Set highlight colors for brackets/braces/parenthesis matching
    hi MatchParen cterm=bold ctermbg=none ctermfg=red
    hi Fold cterm=bold ctermbg=6 ctermfg=grey


   
    "-----------------------------
    "Light Color Scheme
    "-----------------------------
    " "statusline
    " hi StatusLine ctermbg=Black ctermfg=Blue
    " "vertical split line
    " hi VertSplit ctermbg=Black ctermfg=DarkGrey
    " "horizontal split line
    " hi StatusLineNC ctermbg=Black ctermfg=DarkGrey
    " " Line numbers
    " hi LineNr ctermfg=Black ctermbg=Blue
    " hi CursorLineNr ctermfg=Black ctermbg=Green
    " hi Visual term=reverse cterm=reverse guibg=Grey
    "
    " " Tab line highlighting
    " hi TabLineFill ctermbg=Blue ctermfg=Blue
    " hi TabLine ctermfg=Yellow ctermbg=Blue cterm=bold
    " hi TabLineSel ctermfg=DarkGrey ctermbg=Green cterm=bold

    


    "-----------------------------
    "Dark Color Scheme
    "-----------------------------
    " "statusline
    " hi StatusLine ctermbg=Black ctermfg=Grey
    " "vertical split line
    " hi VertSplit ctermbg=Black ctermfg=DarkGrey
    " "horizontal split line
    " hi StatusLineNC ctermbg=Black ctermfg=DarkGrey
    "
    " " Tab line highlighting
    " hi TabLineFill ctermbg=Yellow ctermfg=DarkGrey
    " hi TabLine ctermfg=LightGrey ctermbg=DarkGrey cterm=bold
    " hi TabLineSel ctermfg=DarkGrey ctermbg=yellow cterm=bold


    "--------------------------------------
    "Dark Ubuntu Defaul Color Scheme
    "--------------------------------------
    "statusline
    hi StatusLine ctermbg=Black ctermfg=Grey
    "vertical split line
    hi VertSplit ctermbg=Black ctermfg=DarkGrey
    "horizontal split line
    hi StatusLineNC ctermbg=Black ctermfg=DarkGrey

    " Tab line highlighting
    hi TabLineFill ctermbg=Yellow ctermfg=DarkGrey
    hi TabLine ctermfg=LightGrey ctermbg=DarkGrey cterm=bold
    hi TabLineSel ctermfg=DarkGrey ctermbg=yellow cterm=bold






"====================
" STATUS LINE
"====================


    " Now add what you want

    set laststatus=2                "status line permanently on

    set statusline=%*               "add personal highlighting
                                    "specified in hi User1 ctermbg...
    "set statusline+=%<\            " cut at start
    "set statusline+=%t              "tail of the filename
    "set statusline+=[%{strlen(&fenc)?&fenc:'none'}, "file encoding
    "set statusline+=%{&ff}]        "file format
    "set statusline+=%h             "help file flag
    set statusline+=%F                    " path
    set statusline+=%m              "modified flag
    set statusline+=%r              "read only flag
    let emptyspace='    '
    set statusline+=%{emptyspace}
    set statusline+=@%{hostname()}
    "set statusline+=%y             "filetype
    set statusline+=%=              "left/right separator
    set statusline+=Col\ %c,\       "cursor column
    set statusline+=Lin\ %l/%L  "cursor line/total lines
    set statusline+=\ %P        "percent through file 




"=============================
" INDENTATION
"=============================

    " set defaults
    setlocal ts=4 sts=4 sw=4 expandtab smarttab tw=0

    " set filetype specifics
    augroup filetypespecs 
        autocmd!
        autocmd FileType cpp        setlocal ts=2 sts=2 sw=2 expandtab
        autocmd FileType c          setlocal ts=2 sts=2 sw=2 expandtab
        autocmd FileType fortran    setlocal ts=2 sts=2 sw=2 expandtab
        autocmd FileType python     setlocal ts=4 sts=4 sw=4 expandtab
        autocmd FileType rst        setlocal ts=4 sts=4 sw=4 expandtab
        autocmd FileType sh         setlocal ts=4 sts=4 sw=4 expandtab
        autocmd FileType text       setlocal ts=4 sts=4 sw=4 expandtab
        autocmd FileType txt        setlocal ts=4 sts=4 sw=4 expandtab
        autocmd FileType vim        setlocal ts=4 sts=4 sw=4 expandtab
    augroup end

