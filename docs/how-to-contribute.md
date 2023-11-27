# Contributing

## Pull requests

1. Always use full english names for variable names

    - Only the following exceptions are allowed:
      - "number" should be "num"
      - "argument" should be "arg"
      - "width" can be "W" if the context is clear.
      - "height" can be "H" if the context is clear.

2. Functions should be small, approximately ~6 lines:

Functions should only do one thing.
Certain aspects of the code base don't reflect this but we are working on changing this.
Use PEP8 syntax conventions:

This can be easily achieved when you install a linter e.g. flake8
If new functionality is added please include unit-tests for it.

Please make sure that all unit-tests are passing before your make your PR.

Commits should try to have the following structure:

Commits are titles:
Start with a capital letter
Don't end the commit with a period
Commits should be written to answer: If applied, this commit will A good commit would then look like: "Remove deprecated backend function"
Find more information about how to write good commits here.
Provide documentation of new features:

Use the documentation syntax of the repository
If new functionality is added please add your function to paz/docs/structure.py
After looking into the points here discussed, please submit your PR such that we can start a discussion about it and check that all tests are passing.