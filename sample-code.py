# This is the same starting code as found in the web IDE

import sys
import math
import time

# Auto-generated code below aims at helping you parse
# the standard input according to the problem statement.

width, height = [int(i) for i in input().split()]

# game loop
while True:
    start_time = time.time_ns()
    my_matter, opp_matter = [int(i) for i in input().split()]
    for i in range(height):
        for j in range(width):
            # owner: 1 = me, 0 = foe, -1 = neutral
            scrap_amount, owner, units, recycler, can_build, can_spawn, in_range_of_recycler = [int(k) for k in input().split()]

    # Write an action using print
    # To debug: print("Debug messages...", file=sys.stderr, flush=True)

    print("Time: " + str((time.time_ns() - start_time) / 1000000), file=sys.stderr, flush=True)
    print("WAIT")
