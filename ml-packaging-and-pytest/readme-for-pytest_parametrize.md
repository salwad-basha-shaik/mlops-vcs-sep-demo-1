
# Pytest Parameterization Demo

This repository demonstrates how to use the pytest framework with parameterization to test functions in Python. Specifically, the code tests whether a given word exists in a string of text, using both positive and negative test cases.

## Repository Overview

The repository contains a test script, `pytest_parametrize.py`, that checks if a word is present in a text string. It uses `pytest.mark.parametrize` to run the test function multiple times with different input data.

### Files in the repository:

- **pytest_parametrize.py**: Python script for testing text content using parameterized test cases.
- **test_math.py**: A sample script testing basic math operations (e.g., addition, multiplication, trigonometric sine).

## Script Breakdown

### pytest_parametrize.py

This script contains the following key components:

1. **Function: `text_contain_word(word: str, text: str) -> bool`**
    - This function checks whether a specific word is contained within a given text string.
    - If the word is found in the text, it returns True; otherwise, it returns False.

```python
def text_contain_word(word: str, text: str) -> bool:
    '''Find whether the text contains a particular word'''
    return word in text
```

2. **Test Cases**:
    - A list of test cases, `test`, is defined to test the function. Each test consists of a string and an expected output (True or False).

```python
test = [
    ('There is a duck in this text', True),
    ('There is nothing here', False)
]
```

3. **Parameterized Testing**:
    - The `pytest.mark.parametrize` decorator is used to run the `test_text_contain_word` function with multiple sets of input values (the test data).

```python
@pytest.mark.parametrize('sample, expected', test)
def test_text_contain_word(sample, expected):
    word = 'duck'
    assert text_contain_word(word, sample) == expected
```

The function checks if the word 'duck' is present in the input string and compares the result with the expected output.

## Running the Tests

To run the test cases using pytest, follow these steps:

1. **Install Pytest**:
   Install pytest if you don’t already have it:

```bash
pip install pytest
```

2. **Run the Tests**:
   To run the test script `pytest_parametrize.py`, use the following command:

```bash
pytest -v pytest_parametrize.py
```

3. **Expected Output**:
   After running the test, you will get output similar to the following:

```bash
collected 2 items

pytest_parametrize.py::test_text_contain_word[There is a duck in this text-True] PASSED            [ 50%]
pytest_parametrize.py::test_text_contain_word[There is nothing here-False] PASSED                  [100%]

=================================================== 2 passed in 0.01s ====================================================
```

### Explanation of the Output:

- pytest collects 2 test cases from the parameterized data.
- The first test ('There is a duck in this text') passes, and pytest shows [50%], indicating that half the tests have been completed.
- The second test ('There is nothing here') passes, and pytest shows [100%], meaning all tests have successfully passed.
