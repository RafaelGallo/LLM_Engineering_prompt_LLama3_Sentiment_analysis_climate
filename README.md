# LLM Engineering prompt Sentiment analysis climate with Llama 3

[![GPLv3 License](https://img.shields.io/badge/License-GPL%20v3-yellow.svg)](https://opensource.org/licenses/)
[![AGPL License](https://img.shields.io/badge/license-AGPL-blue.svg)](http://www.gnu.org/licenses/agpl-3.0)
[![author](https://img.shields.io/badge/author-RafaelGallo-red.svg)](https://github.com/RafaelGallo?tab=repositories) 
[![](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/release/python-374/) 
[![](https://img.shields.io/badge/R-3.6.0-red.svg)](https://www.r-project.org/)
[![](https://img.shields.io/badge/ggplot2-white.svg)](https://ggplot2.tidyverse.org/)
[![](https://img.shields.io/badge/transformers-white.svg)](https://huggingface.co/docs/transformers)
[![](https://img.shields.io/badge/Google_Cloud-white.svg)](https://huggingface.co/docs/transformers)
[![](https://img.shields.io/badge/dplyr-blue.svg)](https://dplyr.tidyverse.org/)
[![](https://img.shields.io/badge/readr-green.svg)](https://readr.tidyverse.org/)
[![](https://img.shields.io/badge/ggvis-black.svg)](https://ggvis.tidyverse.org/)
[![](https://img.shields.io/badge/Shiny-red.svg)](https://shiny.tidyverse.org/)
[![](https://img.shields.io/badge/plotly-green.svg)](https://plotly.com/)
[![](https://img.shields.io/badge/XGBoost-red.svg)](https://xgboost.readthedocs.io/en/stable/#)
[![](https://img.shields.io/badge/Tensorflow-orange.svg)](https://powerbi.microsoft.com/pt-br/)
[![](https://img.shields.io/badge/Keras-red.svg)](https://powerbi.microsoft.com/pt-br/)
[![](https://img.shields.io/badge/CUDA-gree.svg)](https://powerbi.microsoft.com/pt-br/)
[![](https://img.shields.io/badge/Caret-orange.svg)](https://caret.tidyverse.org/)
[![](https://img.shields.io/badge/Pandas-blue.svg)](https://pandas.pydata.org/) 
[![](https://img.shields.io/badge/Matplotlib-blue.svg)](https://matplotlib.org/)
[![](https://img.shields.io/badge/Seaborn-green.svg)](https://seaborn.pydata.org/)
[![](https://img.shields.io/badge/Matplotlib-orange.svg)](https://scikit-learn.org/stable/) 
[![](https://img.shields.io/badge/Scikit_Learn-green.svg)](https://scikit-learn.org/stable/)
[![](https://img.shields.io/badge/Numpy-white.svg)](https://numpy.org/) 

<div align="center">
    <img src="https://img.freepik.com/fotos-gratis/chamines-contra-uma-paisagem-industrial-de-ceu-limpo_91128-4692.jpg?t=st=1727545980~exp=1727549580~hmac=3ad404a9538b0cff5eed7ac466180b5e6b5fffa776daa61b06c27e8b8e3e6f6e&w=740" />
</div>

## Objective

The project aims to leverage LLaMA 3, a state-of-the-art Large Language Model (LLM), for sentiment analysis of tweets discussing climate change. Tweets are categorized into positive, neutral, or negative sentiments. Insights derived from this analysis will aid in understanding public opinion, identifying key trends, and supporting decision-making for environmental campaigns, policymaking, and corporate strategies.

## Key Questions

1 What are the dominant sentiments expressed in daily climate change discussions on Twitter?
2 How do sentiments evolve over time, and what events influence these changes?
3 What factors (e.g., hashtags, keywords, influencers) correlate with positive or negative sentiments?
4 Can actionable insights be drawn for stakeholders from this sentiment data?

# Dataset Overview

1 Source: Daily tweets from January 1, 2022, to July 19, 2022, containing the term "Climate Change."
2 Columns: Tweet text, user metadata, engagement metrics (likes, retweets), timestamps, etc.
3 Volume: ~500K tweets.
4 Preprocessing:
5 Remove duplicates, clean links/hashtags, handle missing values, and tokenize the text.

## Dataset link

This dataset here

Link: https://www.kaggle.com/datasets/die9origephit/climate-change-tweets/data

# Accessing the Project's Notebooks

The repository includes a notebook folder containing Jupyter notebooks that showcase the functionality of this project, such as data processing, sentiment analysis, and visualization.

## Steps to Access the Notebooks

1 Clone the Repository First, clone the repository to your local machine:

```bash
git clone https://github.com/RafaelGallo/LLM_Llama3_1_v2_Sentiment_Analysis.git
cd LLM_Llama3_1_v2_Sentiment_Analysis
```

2 Navigate to the notebook Folder Once inside the repository, navigate to the notebook directory where the project notebooks are stored:

```bash
cd notebook
```

3 Open the Notebooks Use your preferred method to open and run the notebooks:

Using Jupyter Notebook:

```bash
jupyter notebook
```

Using JupyterLab

```bash
jupyter lab
```

4 Dependencies Ensure all required Python libraries are installed. You can install them using the requirements.txt file:

```bash
pip install -r requirements.txt
```

## Visualizations

1. Word Cloud: General Clean Text
The word cloud highlights the most common words used in the dataset, providing a quick visual representation of frequent topics related to climate change.

<div align="center"> <img src="https://github.com/RafaelGallo/LLM_Llama3_1_v2_Sentiment_Analysis/blob/main/img/Captura%20de%20tela%202024-11-26%20145522.png?raw=true" alt="Distribution of Sentiments" width="600" /> </div>

3. Top 20 Most Common Tokens
The bar chart shows the frequency of the top 20 tokens in the dataset.

<div align="center"> <img src="https://github.com/RafaelGallo/LLM_Llama3_1_v2_Sentiment_Analysis/blob/main/img/Captura%20de%20tela%202024-11-26%20145533.png?raw=true" alt="Top Tokens" width="600" /> </div>

## HuggingFace CLI Login

To use HuggingFace's pre-trained models, authenticate using the following command:

```bash
huggingface-cli login
```

This allows access to pre-trained models like meta-llama/Llama-3.1-8B-Instruct for text processing.

## Device Selection: GPU or CPU
The following code dynamically checks the availability of a GPU and selects it for processing if available. Otherwise, it defaults to using the CPU:

```bash
# Check if a GPU is available
# If a GPU is available, it will use "cuda"; otherwise, it will default to "cpu"
device = "cuda" if torch.cuda.is_available() else "cpu"
```

# Explanation

- Purpose: Efficiently utilizes hardware resources for faster model execution.
torch.cuda.is_available(): Checks if a CUDA-enabled GPU is accessible on the system.

- device Variable:

* "cuda": Indicates GPU will be used for model computations.
* "cpu": Falls back to CPU when no GPU is detected.

- Why This Matters?

Using a GPU can significantly speed up the execution of large models like LLaMA, especially for tasks such as text classification or generation.
This approach ensures compatibility across various hardware setups without manual intervention.

## Model Initialization

The model and tokenizer are initialized using the HuggingFace Transformers library. It automatically maps the device (GPU/CPU) for efficient memory utilization.

```bash
from transformers import AutoTokenizer, AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct",
                                             torch_dtype=torch.float16,
                                             device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

```

## Pipeline Initialization

The pipeline is configured for zero-shot classification, using the loaded model and tokenizer:

```bash
from transformers import pipeline

labels = ["Positive", "Negative", "Neutral"]
classifier = pipeline('zero-shot-classification',
                      model=model,
                      tokenizer=tokenizer)

```

# Engineering prompt

## Generating Text

Use the model to generate text responses based on prompts:

```bash
def generate_text(prompt, max_length=150):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(inputs.input_ids, max_length=max_length)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

response = generate_text("What is exoplanet discovery?")
print(response)

```

## Sentiment Classification on Dataset

This project includes a sentiment classification pipeline that iterates over a dataset column (Text_Limpo) and classifies the sentiment of each text as Positive, Negative, or Neutral using a zero-shot classification approach. Below is the code example:

```bash

%%time

# Check if the "Text_Limpo" column exists in the dataset
if 'Text_Limpo' not in df.columns:
    raise ValueError("The dataset must contain a column named 'Text_Limpo'.")

# Initialize an empty list to store results
results = []

print("Starting sentiment classification...")
for index, row in df.iterrows():
    text = row['Text_Limpo']

    # Create a specific prompt for each text
    prompt = (
        f"Analyze the sentiment of the following text and classify it as Positive, Negative, or Neutral.\n"
        f"Text: \"{text}\"\n"
        f"Sentiment:"
    )

    try:
        # Perform zero-shot classification
        result = classifier(
            sequences=text,
            candidate_labels=labels,
            hypothesis_template="This text expresses a {} sentiment."
        )

        # Extract the label with the highest score
        sentiment = result['labels'][0]
    except Exception as e:
        print(f"Error processing text: {text[:30]}... - {e}")
        sentiment = "Error"

    results.append(sentiment)

    # Optional: Display progress for every 100 processed texts
    if (index + 1) % 100 == 0:
        print(f"{index + 1} texts processed...")
```

## Explanation of the Code

1. Column Validation: Ensures that the dataset contains a column named Text_Limpo. If not, raises an error to prevent execution.

```bash
if 'Text_Limpo' not in df.columns:
    raise ValueError("The dataset must contain a column named 'Text_Limpo'.")

```

2. Prompt Engineering: Creates a prompt for each text entry in the dataset to guide the model in classifying sentiment.

```bash
prompt = (
    f"Analyze the sentiment of the following text and classify it as Positive, Negative, or Neutral.\n"
    f"Text: \"{text}\"\n"
    f"Sentiment:"
)
```

3. Zero-Shot Classification: Leverages the Hugging Face pipeline for zero-shot classification to process each text entry and classify its sentiment.

```bash
result = classifier(
    sequences=text,
    candidate_labels=labels,
    hypothesis_template="This text expresses a {} sentiment."
)
```

4. Error Handling: Catches any errors during processing and logs them for review.

```bash
except Exception as e:
    print(f"Error processing text: {text[:30]}... - {e}")
    sentiment = "Error
```

5. Progress Tracking: Optionally, displays progress after every 100 texts to monitor execution in larger datasets.

```bash
if (index + 1) % 100 == 0:
    print(f"{index + 1} texts processed...")
```

## Execution Time
The code uses %%time to display the total execution time for processing all texts in the dataset.

# Sentiment Classification Results

The dataset was analyzed using a zero-shot classification pipeline to determine the sentiment of each text entry. Below is a sample of the resulting classification, where the Text_Limpo column contains the preprocessed text, and the Sentiment_LLM column represents the sentiment predicted by the LLaMA 3.1 model.

Sample Output
<div align="center"> <img src="https://github.com/RafaelGallo/LLM_Llama3_1_v2_Sentiment_Analysis/blob/main/img/Captura%20de%20tela%202024-11-26%20150745.png?raw=true" alt="Sample Classification Table" width="600" /> </div>

## Explanation of the Results

1. Columns in the Output:
2. Text_Limpo: Preprocessed text from the dataset, ready for analysis.
3. Sentiment_LLM: Predicted sentiment for each text, classified as Positive, Negative, or Neutral.
   
## Key Observations:

1. Positive sentiments often correlate with constructive or hopeful messages about climate action.
2. Negative sentiments typically highlight challenges, frustration, or skepticism.
3. Neutral sentiments are more descriptive or factual.

# How to Interpret the Results

* Positive Sentiment: Represents optimism or favorable discussions about climate change. Example: "Climate change is one of the world's most pressing issues."
* Negative Sentiment: Highlights concerns, fears, or criticisms about climate-related topics. Example: "The only solution I've ever heard is left propaganda."
* Neutral Sentiment: Indicates statements that are factual or lack strong emotional polarity. Example: "Could have material impact on this year's prices."

# Sentiment Analysis Results and Visualizations

The sentiment analysis results are visualized to provide a clear understanding of the distribution of sentiments and frequently occurring words within each sentiment category.

1. Sentiment Distribution
The bar chart below shows the distribution of sentiments classified by the model (Positive, Negative, and Neutral). The majority of the texts reflect positive sentiments towards climate change discussions.

<div align="center"> <img src="https://github.com/RafaelGallo/LLM_Llama3_1_v2_Sentiment_Analysis/blob/main/img/Captura%20de%20tela%202024-11-26%20150822.png?raw=true" alt="Distribution of Sentiments" width="600" /> </div>

2. Word Clouds
   
Word clouds are generated to highlight the most frequently used words within each sentiment category. These visualizations provide insight into the themes and keywords that dominate each sentiment type.

Positive Sentiment
Words associated with optimism, solutions, and constructive discussions are prevalent, such as "justice," "research," "environmental," and "fossil fuels."

<div align="center"> <img src="https://github.com/RafaelGallo/LLM_Llama3_1_v2_Sentiment_Analysis/blob/main/img/Captura%20de%20tela%202024-11-26%20150850.png?raw=true" alt="Word Cloud - Positive Sentiment" width="600" /> </div>

Negative Sentiment
This category reflects challenges, frustration, or criticisms, with frequent words like "worse," "fails," "propaganda," and "control."

<div align="center"> <img src="https://github.com/RafaelGallo/LLM_Llama3_1_v2_Sentiment_Analysis/blob/main/img/Captura%20de%20tela%202024-11-26%20150859.png?raw=true" alt="Word Cloud - Negative Sentiment" width="600" /> </div>

Neutral Sentiment
Neutral sentiments are more factual or descriptive, with words like "change," "future," "year," and "environmental."

<div align="center"> <img src="https://github.com/RafaelGallo/LLM_Llama3_1_v2_Sentiment_Analysis/blob/main/img/Captura%20de%20tela%202024-11-26%20150910.png?raw=true" alt="Word Cloud - Neutral Sentiment" width="600" /> </div>

# Key Insights

Positive Sentiment Dominance: A significant portion of the dataset reflects positive sentiments, indicating optimism in climate-related discussions.
Negative Sentiments: Highlight concerns, challenges, or frustrations that can guide targeted interventions or awareness campaigns.
Neutral Sentiments: Offer a factual basis for understanding the general narrative.
These insights are critical for stakeholders aiming to understand public sentiment on climate change and to craft strategies for engagement or action.

# Conclusion

The sentiment classification results provide a foundation for understanding public opinion on climate change. These insights can be visualized or used for actionable strategies in policymaking, marketing, and advocacy efforts.
