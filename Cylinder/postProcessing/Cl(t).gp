reset
set term qt size 1100,650
set grid
set key outside right
set xlabel "Time [s]"
set ylabel "Lift coefficient C_l"
set xrange [0:200]
set yrange [-1.5:1.5]


# ---- EDIT THIS PATH (keep quotes!) ----
base = "/Users/mukhtaromer/Desktop/Openfoam/run/ov8a/postProcessing/forceCoeffsIncompressible"

# columns: Time=1, Cl=5
colT  = 1
colCl = 5

# ---- EDIT THESE Re labels to what you want shown in legend ----
Re_0   = 100
Re_50  = 400
Re_100 = 1000
Re_150 = 4000

# Build filenames safely
f0   = sprintf("%s/0/coefficient.dat",   base)
f50  = sprintf("%s/50/coefficient.dat",  base)
f100 = sprintf("%s/100/coefficient.dat", base)
f150 = sprintf("%s/150/coefficient.dat", base)

# (optional) print filenames to confirm paths
print "Reading: ", f0
print "Reading: ", f50
print "Reading: ", f100
print "Reading: ", f150

# line styles / colors
set style line 1 lw 2 lc rgb "#1f77b4"   # blue
set style line 2 lw 2 lc rgb "#d62728"   # red
set style line 3 lw 2 lc rgb "#2ca02c"   # green
set style line 4 lw 2 lc rgb "#9467bd"   # purple

plot \
  f0   using colT:colCl with lines ls 1 title sprintf("Re = %g (0–50)",     Re_0), \
  f50  using colT:colCl with lines ls 2 title sprintf("Re = %g (50–100)",   Re_50), \
  f100 using colT:colCl with lines ls 3 title sprintf("Re = %g (100–150)",  Re_100), \
  f150 using colT:colCl with lines ls 4 title sprintf("Re = %g (150–200)",  Re_150)

