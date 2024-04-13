





"""
function specification flow:
- write tests
- write function signature
- write step-by-step of what needs to happen
- go to steps checking flow
"""
"""
steps checking flow:
- get list of all functions we already have > include import code
- for each step, check if we have a function that does that, or if it can be done via a few lines of base python
  - if yes: go to function creation flow
  - if no:
    - go to subfunction creation flow
    - reset and restart steps checking flow
"""

"""
function creation flow:
- write function
- run tests
- if tests pass, commit function
- if tests fail, go to debug flow
"""

"""
debug flow:
- try to get a passing version of each failed test
- take the best version out of the pool, and try to combine it with the ones that passed tests that it failed
"""






# ....
# > ---0.2.2---
# > (commit)
# > call existing functions directly via code of thoughts
# > combine existing functions to write new functions
# > tester agent (default human) to test functions written
# > output function needs to take in artifacts at location (or text) instead of raw data
# > needs to always output its results to a file
