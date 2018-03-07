# Alias Method 

The alias method is a method for randomly selecting an element from a set in
wich each element has an arbitrary probability of being selected.

The usual way this task is performed is by doing a binary search on the
accumulated distribution array, this requires O(log n) comparisions, which for
most cases is acceptable.

But for big sets (with approximately a million elements) and large number of
choices (O(n log(n)) choices) the alias method is vastly supperior.

![Time taken to perform approximately a billion choices](https://raw.githubusercontent.com/lgmsantos/pyalias/master/figures/time.png)

![Time taken to perform approximately a billion choices](https://raw.githubusercontent.com/lgmsantos/pyalias/master/figures/time.svg)

