
# Google Cloud SDK Installation and DVC Setup Guide

This guide provides step-by-step instructions for installing the Google Cloud SDK (gcloud CLI) in a Linux environment (Anaconda), as well as how to use DVC to push your dataset to Google Cloud Storage.

## Prerequisites

- [Anaconda](https://www.anaconda.com/products/distribution#download-section) environment set up on your Linux machine.
- Access to a Google Cloud Platform (GCP) project for storing datasets.

## Steps to Install Google Cloud SDK on Linux

1. **Download the Google Cloud SDK**  
   Navigate to your home directory where you want to download the SDK file, then run:
   
   ```bash
   curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-linux-x86_64.tar.gz
   ```

2. **Extract the SDK**  
   To extract the contents of the file to your home directory, use:
   
   ```bash
   tar -xf google-cloud-cli-linux-x86_64.tar.gz
   ```

3. **Run the Installation Script**  
   Add the gcloud CLI to your path by running the installation script from the root of the extracted folder:
   
   ```bash
   ./google-cloud-sdk/install.sh
   ```
   
   - **Non-Interactive Install**: You can run the installation script with flags for non-interactive setups, e.g., in a script:
   
     ```bash
     ./google-cloud-sdk/install.sh --help
     ```

4. **Initialize the gcloud CLI**  
   Run the following command to initialize the gcloud CLI (this step does not involve pushing any data):
   
   ```bash
   ./google-cloud-sdk/bin/gcloud init
   ```

5. **Log In to GCP Console**  
   To authenticate your Anaconda environment, use:
   
   ```bash
   /Users/salwad/google-cloud-sdk/bin/gcloud auth application-default login
   ```

## Setting Up DVC for Google Cloud Storage

1. **Install the `dvc-gs` Dependency**  
   Run the following command to install the `dvc-gs` package:
   
   ```bash
   pip install dvc-gs
   ```

2. **Configure a Google Cloud Storage Remote for DVC**  
   Once installed, you can add your GCP bucket as a remote and push datasets with DVC. Run:
   
   ```bash
   dvc add remote -d myremote gs://dvcdemo3
   ```

3. **Push Your Dataset to GCP**  
   Use DVC to push your dataset to Google Cloud Storage:
   
   ```bash
   dvc push
   ```

   After pushing, you can verify that the dataset is stored in your GCP bucket.

## Optional: Try with AWS

You can also configure DVC to work with AWS S3 by running `aws configure` and setting up a new DVC remote for your S3 bucket. Then, use `dvc push` to upload your datasets to S3.

## Retrieving Dataset on Another Machine

When you `dvc pull` on another machine that has a cloned repository, you will get the datasets and their version history.

---

For more details, refer to the official Google Cloud SDK installation guide: [Google Cloud SDK Documentation](https://cloud.google.com/sdk/docs/install#linux).
