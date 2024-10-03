
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

   - **Step 1**: Run the sample app. ( below will run the streamlit app and then pass output to node package module to install localtunnel and then forward your localport 8501 to https URL), password is your public-ip
   
   For getting the publicip
   ```bash
     curl ipv4.icanhazip.com
     (or)
     wget -q -O - ipv4.icanhazip.com
     ```
   To run the app
     ```bash
     streamlit run app_streamlit.py & npx localtunnel --port 8501
     ```
---

### Demo-2: Fetching the dataset from openml and train and deploy to streamlit, localtunnel


#### Steps to Follow:

   For getting the publicip

   ```bash
     curl ipv4.icanhazip.com
     (or)
     wget -q -O - ipv4.icanhazip.com
     ```
   To run the app
   
     ```bash
     streamlit run app_ML_streamlit.py & npx localtunnel --port 8501
     ```

---