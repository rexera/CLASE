# CLASE: A Hybrid Method for Chinese Legalese Stylistic Evaluation

**CLASE** (**C**hinese **L**eg**A**lese **S**tylistic **E**valuation) is a hybrid evaluation framework designed to assess the stylistic fidelity of legal text generation. It combines objective linguistic feature analysis with experience-guided LLM evaluation to providing a transparent, reference-free, and interpretable assessment.

## Features

- **Hybrid Scoring**: Combines objective linguistic features (z-score normalized) with subjective LLM-as-a-judge assessments.
- **Contrastive Learning**: Automatically learns stylistic criteria from authentic vs. restored document pairs without manual annotation.
- **Interpretable Feedback**: Provides detailed, natural language feedback on stylistic deficiencies (e.g., lexical choice, sentence structure).
- **Reference-Free**: Evaluates generated text quality without requiring a gold-standard reference during inference.

## Project Structure

- `exp_train_parallel.py`: Script for **Training-Free Contrastive Learning**. Extracts positive and negative stylistic examples from document pairs.
- `objective_scoring.py`: Computes the **Objective Score** using logistic regression on linguistic features.
- `subjective_scoring.py`: Computes the **Subjective Score** using an LLM judge with retrieval-augmented examples.
- `linguistic_features/`: Contains modules for extracting shallow, syntactic, and discourse-level features.

## Data

The dataset consists of Chinese legal documents structured for stylistic evaluation and restoration tasks.

- **`data/train/`**: Training data (4,000 samples).
  - `gist_4k.jsonl`: Contains the original case gists (summaries).
    - Fields: `index`, `reserved`, `gist`
  - `restored_4k.jsonl`: Contains the restored judgments corresponding to the gists.
    - Fields: `index`, `reserved`, `gist`, `restored`
  - `reason_4k.json`: Contains detailed legal reasoning and provisions.
    - Fields: `index`, `header`, `reason`, `provisions`

- **`data/test/`**: Test data (200 samples).
  - `restored_4001-4200.jsonl`: Test set with gold standard and generated texts.
    - Fields: `index`, `gold` (original judgment), `generated` (model output)

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/rexera/CLASE.git
   cd CLASE
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Environment Setup**:
   Create a `.env` file in the root directory with your API keys:
   ```env
   # OpenAI / LLM API Configuration
   OPENAI_API_KEY=your_api_key_here
   BASE_URL=https://api.openai.com/v1  # or your custom endpoint
   MODEL=gpt-4o-mini
   
   # Specific configs for scoring (optional overrides)
   GENERATION_API_KEY=your_generation_key
   GENERATION_BASE_URL=...
   EMBEDDING_API_KEY=your_embedding_key
   EMBEDDING_BASE_URL=...
   ```

## Usage

### 1. Construct Experience Pool (Contrastive Learning)
Extract stylistic examples from your training pairs (authentic vs. restored documents).

```bash
python exp_train_parallel.py
```
*Input*: `data/train/reason_4k.json`, `data/train/restored_4k.jsonl`  
*Output*: `clase_exp/model_output/examples.jsonl`

### 2. Objective Scoring
Train the logistic regression model on the experience pool and score new texts.

```bash
python objective_scoring.py
```
*Outputs*: Trained model weights and scores in `output/objective_scores/`.

### 3. Subjective Scoring
Run the LLM-as-a-judge evaluation with retrieval-augmented examples.

```bash
python subjective_scoring.py
```
*Outputs*: Detailed scoring and feedback in `output/subjective_scores/`.

## Citation

If you use CLASE in your research, please cite our LREC 2026 paper (full bibtex is pending):

```bibtex
@inproceedings{ma2026clase,
  title={CLASE: A Hybrid Method for Chinese Legalese Stylistic Evaluation},
  author={Ma, Yiran Rex and Ye, Yuxiao and Xie, Huiyuan},
  booktitle={Proceedings of the 15th Biennial Language Resources and Evaluation Conference (LREC 2026)},
  year={2026}
}
```

## Credits

Objective scoring is an open-source reproduction of: [Qiu, X., Deng, K., Qiu, L., Wang, X. (2018). Exploring the Impact of Linguistic Features for Chinese Readability Assessment.](https://doi.org/10.1007/978-3-319-73618-1_67)

We acknowledge the following resources used in our linguistic feature extraction:

- **Features 1-3**: [Table of General Standard Chinese Characters](https://github.com/jaywcjlove/table-of-general-standard-chinese-characters/tree/8d22e19b7e77194cdfb3ee103d7b7d300401eee8)
- **Features 4-7**: [Zi Character Dataset](https://github.com/secsilm/zi-dataset)
- **Features 54-59**: [THUOCL (Tsinghua Open Chinese Lexicon)](https://github.com/thunlp/THUOCL)
  > Shiyi Han, Yuhui Zhang, Yunshan Ma, Cunchao Tu, Zhipeng Guo, Zhiyuan Liu, Maosong Sun. THUOCL: Tsinghua Open Chinese Lexicon. 2016.

## License

MIT

