
# Sep_vcs_demo

This repository contains **ML-packaging demos** to showcase configurations for different environments.

---

## ML-Packaging Demos

### Demo-1: Environment Configurations

This demo explains how to configure and run the application for different environments.

#### Steps to Follow:

1. **Install Prerequisites**
   - Install **Visual Studio Code (VS Code)**.
   - Install **Anaconda**.
   
2. **Activate Anaconda**
   After installing Anaconda, activate the environment by running:
   
   ```bash
   conda activate
   ```

3. **Run the Configuration and Application**
   Navigate to the `ml-packaging-and-pytest` folder and execute the following steps:

   - **Step 1**: Run the configuration script.
   
     ```bash
     python config.py
     ```

   - **Step 2**: Set the environment to production and run the application.
   
     ```bash
     export ENV=production
     python app.py
     ```

---

### Demo-2: Testing Environment File with Unit Test Cases

This demo provides steps to test the environment configuration using unit test cases.

#### Steps to Follow:

1. Set the environment to development:
   
   ```bash
   export ENV=development
   ```

2. Run the unit tests:

   ```bash
   python test_config.py
   ```

---


### Demo-3: YAML Configuration Example

In addition to the previous methods, configurations can also be set using a YAML file. 
An example of this can be found in the `ml-packaging-and-pytest/config1.yml` file.

#### To Read and Print Configuration Settings:

You can read and print all the configuration settings using the following command:

```bash
python ml-packaging-and-pytest/explain_yaml.py
```

---

### Demo-4: virtual environments in python Example

Read the below file to understand and know the importance of two different environments in same machine to handle 2 different models that are required two different versions of dependancies.

```bash
sh ml-packaging-and-pytest/virtualenv-demo.sh
```
To create virutal environments we can use below commands
```bash
virtualenv demo1
python -m venv demo2
```

To activate your virutal environments we can use below commands
```bash
source demo1/bin/activate
source demo2/bin/activate
```

---

### Demo-5: Pytest demo-1

This will check basic unit tests using assertions for addition, multiplication, and sine calculation using Python’s math library.

To run this 
```bash
python -m pytest test_math.py
```
---

### Demo-6: Pytest demo-2

This will check the pytest framework with parameterized tests to check if a given word (“duck”) exists in provided text samples. And see [ml-packaging-and-pytest/readme-for-pytest_parametrize.md](readme-for-pytest_parametrize.md) file clear explanation.

To run this 
```bash
pytest -v ml-packaging-and-pytest/pytest-more-demos/pytest_parametrize.py
```

---


### Demo-7: Pytest demo-3

this will demonstrates the use of a pytest fixture to provide test data to a test function.

Code Explanation

	1.	Importing pytest:
	•	The code begins by importing the pytest library, which is essential for running tests and using fixtures.
	2.	Creating a Fixture:
	•	The @pytest.fixture decorator is used to define a fixture named sample_data.
	•	This fixture, sample_data, returns a list of integers: [1, 2, 3, 4, 5].
	•	Fixtures in pytest are useful for setting up data or states that can be used across multiple test functions.
	3.	Defining the Test Function:
	•	The function test_sum takes sample_data as an argument. Pytest automatically recognizes that sample_data is a fixture and provides its return value to the test function.
	•	Inside test_sum, the code uses assert to check if the sum of sample_data equals 15.
	•	If the sum is indeed 15, the test will pass. Otherwise, it will fail.


To run this 
```bash
pytest -v test_fixture_example.py  -- to clear output like how much percentage completed by each testcase.
python -m pytest pytest_parametrize.py - for normal output that displays how much combined percentage at the end of all testcases.
```