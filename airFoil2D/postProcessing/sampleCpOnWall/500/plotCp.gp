reset

file = "static(p)_coeff_airfoilWall.raw"

# --- find chord ---
stats file using 1 nooutput
xmin = STATS_min
xmax = STATS_max
c = xmax - xmin
print sprintf("Chord c = %.6f", c)

# --- terminal & axes ---
set terminal qt size 900,600 font "Sans,18" noenhanced
set grid
set key top right font ",18"
set xlabel "x/c" font ",18"
set ylabel "Cp" font ",18"
set title "Cp(x/c) at iteration 500" font ",22"
set yrange [*:*] reverse

# --- plot ---
# Filter by returning 1/0 for x and y when a point should be skipped.
# This avoids a 3rd using column (which is what was breaking your colors).

plot \
  file using ( ($2>0 && $3==0) ? (($1-xmin)/c) : 1/0 ):( ($2>0 && $3==0) ? $4 : 1/0 ) \
    with points lt 1 pt 6 ps 1.2 title "Upper surface", \
  file using ( ($2<0 && $3==0) ? (($1-xmin)/c) : 1/0 ):( ($2<0 && $3==0) ? $4 : 1/0 ) \
    with points lt 2 pt 6 ps 1.2 title "Lower surface"

pause -1

