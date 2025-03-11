\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{listings}
\usepackage{xcolor}
\usepackage{hyperref}

\definecolor{codegray}{rgb}{0.95,0.95,0.95}
\lstset{
  basicstyle=\small\ttfamily,
  backgroundcolor=\color{codegray},
  frame=single,
  columns=flexible,
  keepspaces=true
}

\begin{document}

\title{Occupancy Grid Mapping \& A* Planning (Brief)}
\author{Your Name}
\date{\today}
\maketitle

\section{Overview}
This document demonstrates:
\begin{itemize}
  \item Building a \textbf{fine occupancy grid} from a robot path (skipping pure turns).
  \item \textbf{Merging} or down-sampling the grid to a coarser resolution.
  \item Optionally \textbf{dilating} path cells to widen corridors.
  \item \textbf{Saving} the final grid as a \verb|.npy| file.
  \item \textbf{Loading} that file in a separate script and running \textbf{A*}.
\end{itemize}

\section{Main Code (Build \& Save)}
Below is a minimal snippet of Python that uses \verb|vis_nav_game| to record poses and build the grid:
\begin{lstlisting}[language=Python]
# main_build_map.py

OCC_RESOLUTION = 0.1
CORRIDOR_OFFSET = 1.5
NEW_RESOLUTION = 0.3

# Build the fine grid from path, mark PATH=1, WALL=0, UNKNOWN=-1
# Merge to coarser resolution, optionally dilate the path, then save:
merged_dil = dilate_path_cells( merged_occ, kernel_size=7, iterations=2 )
np.save("my_occupancy_map.npy", merged_dil)
\end{lstlisting}

\section{Separate Code (Load \& A*)}
Here is a second script that \emph{loads} the grid, treats \verb|-1| or \verb|1| as free and \verb|0| as blocked, then runs A*:
\begin{lstlisting}[language=Python]
# main_run_planner.py

occ = np.load("my_occupancy_map.npy")
free_cells = np.argwhere( (occ==1) | (occ==-1) )
start = tuple(random.choice(free_cells))
goal  = tuple(random.choice(free_cells))
path  = astar_occupancy(occ, start, goal)
\end{lstlisting}

\vspace{1em}
You can then plot or visualize the path in your preferred manner.

\end{document}
