
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
   python -m unittest -v ml-packaging/test_config.py
   ```

---

