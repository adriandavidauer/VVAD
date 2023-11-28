# Contributing

## Pull requests

1. Always use full english names for variable names

    - Only the following exceptions are allowed:
      - "number" should be "num"
      - "argument" should be "arg"
      - "width" can be "W" if the context is clear.
      - "height" can be "H" if the context is clear.
      
2. Functions should be small, approximately < 20 lines:

   - Functions should only do one thing.
   - Certain aspects of the code base don't reflect this, but we are working on changing this. 

3. Use PEP8 syntax conventions:

   - This can be easily achieved when you install a linter e.g. flake8 or ruff
   - In the pipeline, the ruff linter is used
     - It is ignoring the following issues on purpose: **F405**, **F403**, **E402**, **E722**

4. If new functionality is added please include unit-tests for it.
   - Unit tests are divided into sample and dataset tests and usual utilities

5. Please make sure that all unit-tests are passing before your make your PR.

6. Commits should try to have the following structure:
   - Commits MUST be prefixed with a type, which consists of a noun, feat, fix, etc., followed by a colon and a space.
     - **feat**: (new feature for the user, not a new feature for build script)
     - **fix**: (bug fix for the user, not a fix to a build script)
     - **docs**: (changes to the documentation)
     - **style**: (formatting, missing semi colons, etc; no production code change)
     - **refactor**: (refactoring production code, eg. renaming a variable)
     - **test**: (adding missing tests, refactoring tests; no production code change)
     - **chore**: (updating grunt tasks etc; no production code change)
   - Commits are titles:
     - Start with a capital letter
     - Don't end the commit with a period
     - Commits should be written to answer: If applied, this commit will A good commit would then look like: "Remove deprecated backend function"

7. Provide documentation of new features:

   - Use the documentation syntax of the repository
   
8. After looking into the points here discussed, please submit your PR such that we can start a discussion about it and check that all tests are passing.