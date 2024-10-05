
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

---

### Demo-3: Fetching Iris Dataset from sklearn library, Training, and save it to .pkl file and Deploying to Streamlit via Localtunnel

#### Steps to Follow:

1. **Retrieve Your Public IP Address**

   Before starting the application, obtain your public IP address:

   ```bash
   curl ipv4.icanhazip.com
   # or
   wget -q -O - ipv4.icanhazip.com
   ```

2. **To Dataset load, train and save the model into .pkl file**

   Execute the following:

   ```bash
   conda activate  # Activate conda if it is not activated
   python iris_train_file_pkl.py
   ```

   After running above command you can observe that .pkl file will be created or whatever file you have it will updated.

3. **To Deploy app to streamlit and then expose it as HTTPS via localtunnel**

   To run the app, execute the following:

   ```bash
   streamlit run app_ML_iris_serve.py & npx localtunnel --port 8501
   ```
   After deploying you can validate in your local as well as in using localtunnel URL with public ip as a password for your tunnnel.

---
### Demo-5: run sample flask API and access it.

Execute the following:

   ```bash
   conda activate  # Activate conda if it is not activated
   python streamlit-and-flask_app/simple_flask_app.py
   ```


---

### Demo-6: Creating flask Api for the Iris Dataset from sklearn library that we have trained and saves it to .pkl file , after creating accessing this API by 3 ways ( python requests library, postman, CURL command)

#### Steps to Follow:

1. **Ensure that you have iris_model.pkl that we have generated from iris_train_file_pkl.py file**

2. **Run the microservice that we have deployed using Flask**

   Execute the following:

   ```bash
   conda activate  # Activate conda if it is not activated
   python streamlit-and-flask_app/modelExecute.py
   ```

   After running above command you can observe that flask is exposing our api using 5000 port. you can access it and for our model use /predict as a path to the api.

3. **After running the microservice you can access it using 3 ways**

   1st way: To access it using python 'requests' library to access API and validate. execute the following from new terminal windows:

   ```bash
   python streamlit-and-flask_app/modelLoadUsingRequest.py
   ```
   you can check output and validate.

   2nd way: To access it using python 'postman'(application). Just add a http post method request with the below URL and then add below json raw data as a input and then hit Send button.

   ```bash
   URL: http://127.0.0.1:5000/predict

   json data:

   {
    "features":[5.5, 2.5, 4.0, 1.3]
   }
   ```
   
   3rd way: Using CURL command with POST method by passing the input arguments to the api endpoint.

   CURL command:

   ```bash
   curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d '{"features":[5.5, 2.5, 4.0, 1.3]}'
   ```
   you can check output and validate.
