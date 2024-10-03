
# Streamlit and Flask Demos

This repository contains **Streamlit and Flask demos** to showcase configurations for different environments.

---

## Streamlit and Flask Demos

### Demo-1: Running a Basic Streamlit App and Exposing It via HTTPS using Localtunnel

#### Steps to Follow:

1. **Activate Anaconda**
   After installing Anaconda, activate your environment by running:
   
   ```bash
   conda activate
   ```

2. **(Optional) Create and Activate Virtual Environments**
   
   You can create virtual environments using the following commands:
   
   ```bash
   virtualenv demo1  # Using virtualenv
   python -m venv demo2  # Using venv
   ```

   To activate the virtual environments:
   
   ```bash
   source demo1/bin/activate  # Activating virtualenv
   source demo2/bin/activate  # Activating venv
   ```

3. **Install Prerequisites**
   
   - Install **Streamlit**.
   - Install **Node.js** and **npm** (for `npx` support).

   ```bash
   pip install streamlit
   conda install -c conda-forge nodejs
   ```

4. **Run the Basic Streamlit Application**
   
   Navigate to the `streamlit-and-flask_app` folder and execute the following steps:

   - **Step 1**: Run the sample app. The command below will run the Streamlit app, pass the output to Node.js, and install Localtunnel to expose port 8501 as a secure HTTPS URL.

   - To retrieve your public IP address (which will be used as the password):
   
     ```bash
     curl ipv4.icanhazip.com
     # or
     wget -q -O - ipv4.icanhazip.com
     ```

   - To run the app:
   
     ```bash
     streamlit run app_streamlit.py & npx localtunnel --port 8501
     ```

---

### Demo-2: Fetching a Dataset from OpenML, Training, and Deploying to Streamlit via Localtunnel

#### Steps to Follow:

1. **Retrieve Your Public IP Address**

   Before starting the application, obtain your public IP address:

   ```bash
   curl ipv4.icanhazip.com
   # or
   wget -q -O - ipv4.icanhazip.com
   ```

2. **Run the App**

   To run the app, execute the following:

   ```bash
   streamlit run app_ML_streamlit.py & npx localtunnel --port 8501
   ```

---
