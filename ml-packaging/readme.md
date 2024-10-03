
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
   Navigate to the `ml-packaging` folder and execute the following steps:

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
An example of this can be found in the `ml-packaging/config1.yml` file.

#### To Read and Print Configuration Settings:

You can read and print all the configuration settings using the following command:

```bash
python ml-packaging/explain_yaml.py
```

---

### Demo-4: virtual environments in python Example

Read the below file to understand and know the importance of two different environments in same machine to handle 2 different models that are required two different versions of dependancies.

```bash
sh ml-packaging/virtualenv-demo.sh
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
