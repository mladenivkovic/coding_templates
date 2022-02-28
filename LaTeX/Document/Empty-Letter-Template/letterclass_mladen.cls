\author{Mladen Ivkovic}
\RequirePackage[dvipsnames]{xcolor} % moved here I guess article class causes option clash with xcolor if used after loading the class
\LoadClass[12pt]{article}
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{letterclass_mladen}[2021/02/28 Mladen's Letter Class]
\RequirePackage[left=0.75in,top=0.75in,right=0.75in,bottom=0.75in]{geometry} % Document margins
\RequirePackage{hyperref}
\RequirePackage{fontawesome}
\RequirePackage{ifthen}
\RequirePackage{lipsum}
\RequirePackage{array}
\pagestyle{empty}




%--------------------------
% Setup
%--------------------------

\definecolor{LinkColour}{HTML}{3f64a1}

\hypersetup{
    colorlinks=true,
    urlcolor=LinkColour,
}

\newcommand*{\setname}[2]{\def\@firstname{#1}\def\@lastname{#2}}
\newcommand*{\setstreet}[1]{\def\@street{#1}}
\newcommand*{\setcity}[1]{\def\@city{#1}}
\newcommand*{\setcountry}[1]{\def\@country{#1}}
\newcommand*{\setmobile}[1]{\def\@mobile{#1}}
\newcommand*{\setmail}[1]{\def\@mail{#1}}
\newcommand*{\setlinkedinaccount}[1]{\def\@linkedinaccount{#1}}
\newcommand*{\setgithubaccount}[1]{\def\@githubaccount{#1}}
\newcommand*{\sethomepage}[1]{\def\@homepage{#1}}
\newcommand*{\setposition}[1]{\def\@position{#1}}
\newcommand*{\setcompanyname}[1]{\def\@companyname{#1}}

\newcommand*{\setaddressee}[1]{\def\@addressee{#1}}
\newcommand*{\setdate}[1]{\def\@date{#1}}





%--------------------------
% Header
%--------------------------

\newcommand{\letterheader}{
    \hfill
    \begin{minipage}{0.45\linewidth}
        \flushright
        \name \\[0.25cm] 
        \position \\[0.25cm]
        \contact \\[0.25cm]
    \end{minipage}
}

%~~~~~~~~~~~~~~~~~~~~~~~~~~
% contact subviews 
%~~~~~~~~~~~~~~~~~~~~~~~~~~

\newcommand{\homepageview}
    {\href{\@homepage}{\faHome\ \footnotesize homepage}}

\newcommand{\githubview}
    {\href{\@linkedinaccount}{\faLinkedinSquare\ \footnotesize linkedin}}

\newcommand{\linkedinview}
    {\href{\@githubaccount}{\faGithubSquare\ \footnotesize github}}



\newcommand{\name}{%
    {\large \@firstname \ \@lastname}
}

\newcommand{\contact}{%
        {\small \@street \\ \@city \\ \@country \\[.125cm] \@mail}
}

\newcommand{\contactlinks}{%
        {\small \homepageview \cps \linkedinview \cps \githubview}
}

\newcommand{\position}{%
        {\small \@position \\ \@companyname} %
}

\newcommand{\writedate}{%
        \begin{flushright}
            {\it \@date}
        \end{flushright}
}

\newcommand{\signature}{%
    \vspace{10ex}
    \noindent Sincerely, \\[2em]
    \@firstname \ \@lastname
}

\newcommand{\address}{
    \vspace{15ex}
    \writedate \vspace{6ex}
    \textbf{\noindent Dear \@addressee, \\[1em]}
}

\newcommand\letterbody[1]{ 
  #1
}


