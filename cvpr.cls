% ---------------------------------------------------------------
% CVPR document class
% Based on the CVPR style templates
% ---------------------------------------------------------------

\ProvidesClass{cvpr}[2021/01/01 v1.0 CVPR Paper]
\NeedsTeXFormat{LaTeX2e}

\LoadClass[10pt,twocolumn,letterpaper]{article}

% Required packages
\RequirePackage{times}
\RequirePackage{epsfig}
\RequirePackage{graphicx}
\RequirePackage{amsmath}
\RequirePackage{amssymb}
\RequirePackage{booktabs}

% Page layout
\setlength{\textheight}{9.0in}
\setlength{\textwidth}{6.5in}
\setlength{\columnsep}{0.25in}
\setlength{\topmargin}{-0.5in}
\setlength{\headheight}{0.0in}
\setlength{\headsep}{0.0in}
\setlength{\oddsidemargin}{0.0in}
\setlength{\evensidemargin}{0.0in}

% Define conference commands
\newcommand{\cvprPaperID}[1]{\def\@cvprPaperID{#1}}
\newcommand{\cvprfinalcopy}{\def\@cvprfinalcopy{}}

% Set default values
\cvprPaperID{****}
\def\confYear{YYYY}

% Header and footer
\def\ps@headings{%
  \def\@oddhead{\hfil}
  \def\@evenhead{\hfil}
  \def\@oddfoot{\hfil\thepage\hfil}
  \def\@evenfoot{\hfil\thepage\hfil}
}

% Title formatting
\renewcommand\maketitle{\par
  \begingroup
    \renewcommand\thefootnote{\@fnsymbol\c@footnote}%
    \def\@makefnmark{\rlap{\@textsuperscript{\normalfont\@thefnmark}}}%
    \long\def\@makefntext##1{\parindent 1em\noindent
            \hb@xt@1.8em{%
                \hss\@textsuperscript{\normalfont\@thefnmark}}##1}%
    \if@twocolumn
      \ifnum \col@number=\@ne
        \@maketitle
      \else
        \twocolumn[\@maketitle]%
      \fi
    \else
      \newpage
      \global\@topnum\z@   % Prevents figures from going at top of page.
      \@maketitle
    \fi
    \thispagestyle{empty}\@thanks
  \endgroup
  \setcounter{footnote}{0}%
  \global\let\thanks\relax
  \global\let\maketitle\relax
  \global\let\@maketitle\relax
  \global\let\@thanks\@empty
  \global\let\@author\@empty
  \global\let\@date\@empty
  \global\let\@title\@empty
  \global\let\title\relax
  \global\let\author\relax
  \global\let\date\relax
  \global\let\and\relax
}

\def\@maketitle{%
  \newpage
  \null
  \vskip 2em%
  \begin{center}%
  \let \footnote \thanks
    {\LARGE \@title \par}%
    \vskip 1.5em%
    {\large
      \lineskip .5em%
      \begin{tabular}[t]{c}%
        \@author
      \end{tabular}\par}%
    \vskip 1em%
    {\large \@date}%
  \end{center}%
  \par
  \vskip 1.5em}

% Abstract and keywords
\renewenvironment{abstract}{%
  \noindent\small
  \textbf{Abstract}
  \par\noindent\ignorespaces
}{}

% Section formatting
\renewcommand\section{\@startsection {section}{1}{\z@}%
                                   {-2.0ex \@plus -0.5ex \@minus -.2ex}%
                                   {1.5ex \@plus.2ex}%
                                   {\large\bfseries}}
\renewcommand\subsection{\@startsection{subsection}{2}{\z@}%
                                     {-1.8ex\@plus -0.5ex \@minus -.2ex}%
                                     {1.0ex \@plus .2ex}%
                                     {\normalsize\bfseries}}
\renewcommand\subsubsection{\@startsection{subsubsection}{3}{\z@}%
                                     {-1.8ex\@plus -0.5ex \@minus -.2ex}%
                                     {1.0ex \@plus .2ex}%
                                     {\normalsize\itshape}}

% Caption formatting
\long\def\@makecaption#1#2{%
  \vskip\abovecaptionskip
  \sbox\@tempboxa{\small #1: #2}%
  \ifdim \wd\@tempboxa >\hsize
    \small #1: #2\par
  \else
    \global \@minipagefalse
    \hb@xt@\hsize{\hfil\box\@tempboxa\hfil}%
  \fi
  \vskip\belowcaptionskip}

% Bibliography
\def\thebibliography#1{\section*{References}\small
  \list{\arabic{enumi}}
  {\settowidth\labelwidth{[#1]}\leftmargin\labelwidth
    \advance\leftmargin\labelsep
    \usecounter{enumi}}
  \def\newblock{\hskip .11em plus .33em minus -.07em}
  \sloppy
  \sfcode`\.=1000\relax}
\let\endthebibliography=\endlist

% Enable subfigures (optional)
\newcommand{\subfigure}[2][]{%
  \begin{tabular}{c}
    #2 \\
    \textsf{\footnotesize #1}
  \end{tabular}%
} 