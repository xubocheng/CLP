# CLP
## Setup
1. **Clone the repository**
2. **Create and activate a virtual environment**
3. **Install the required dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
4. **Download the model files from Hugging Face**:
    - Visit [Hugging Face ESM2 Model](https://huggingface.co/facebook/esm2_t33_650M_UR50D).
    - Download the necessary files and place them in the `esm_model` directory .
5. **Download the trained model parameters**:
    - If you want to use the fine-tuned model weights directly, visit [Zenodo link](https://zenodo.org/record/xxxxxxx).
    - Download the trained model parameters and place them in the appropriate directory in your project.

## Usage
1. **Train Classification Model**:
     ```bash
    bash classification.sh
    ```
2. **Train Regression Model**:
   ```bash
    bash regression.sh
    ```
3. **Run Pipeline**:
     To load the trained models and perform classification and regression to get candiadate seqeunces, run:
   ```bash
    python pipeline.py
    ```
     We have already screened the randomly generated sequence space, and the screening results are in the `candidate_sequences` directory.
