# Project thesis simulator
This Git contains the path planning simulator used in my project thesis. See the appendix of the `guidance_systems_for_asv_thomas_stenersen.pdf` file for more details.

## Algorithms
### Hybrid State A*
This is an implementation of the hybrid state A* algorithm as proposed by
(Dolgov et. al., 2008/2010) in Python.

### A* search
Implementation of the classic A* search.

### Virtual Force Field
Implementation of the virtual force field collision avoidance algorithm (Borenstein, 1989).

### Dynamic Window Approach
Implementation of the dynamic window approach (Fox, 1997).

---

## Requirements
1. NumPy/SciPy
2. Matplotlib
3. Dubins module `sudo pip install dubins`

** NOTE: There may be some kinks here and there. Start with `scenario.py` and work your way from there **
