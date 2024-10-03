
# Streamlit and flask demos

This repository contains **Streamlit and flask demos** to showcase configurations for different environments.

---

## Streamlit and flask demos Demos

### Demo-1: Streamlit basic app running and exposing as a https URL using localtunnel

#### Steps to Follow:
   
1. **Activate Anaconda**
   After installing Anaconda, activate the environment by running:
   
   ```bash
   conda activate
   ```

2. **Create virtual environments and activate them (optional)**

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

2. **Install Prerequisites**
   - Install **Streamlit**.
   - Install **Node.js and npm (for npx support)**.

    ```bash
   pip install streamlit
   conda install -c conda-forge nodejs
   npx localtunnel
   ```

3. **Run the basic streamlit Application**
   Navigate to the `streamlit-and-flask_app` folder and execute the following steps:

   - **Step 1**: Run the sample app. ( below will run the streamlit app and then pass output to node package module to install localtunnel and then forward your localport 8501 to https URL)
   
     ```bash
     streamlit run app_streamlit.py & npx localtunnel --port 8501
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
