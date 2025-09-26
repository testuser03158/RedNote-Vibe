# RedNote-Vibe Dataset

**A Dataset for Capturing Temporal Dynamics of AI-Generated Text in Social Media** 

![Dataset Overview](assets/xhs_dist.png)

## TL DR 

The proliferation of Large Language Models (LLMs) has led to widespread AI-Generated Text (AIGT) on social media platforms, creating unique challenges where content dynamics are driven by user engagement and evolve over time. However, existing datasets mainly depict static AIGT detection. In this work, we introduce **RedNote-Vibe**, the first longitudinal (5-years) dataset for social media AIGT analysis. This dataset is sourced from Xiaohongshu platform, containing user engagement metrics (e.g., likes, comments) and timestamps spanning from the pre-LLM period to July 2025, which enables research into the temporal dynamics and user interaction patterns of AIGT.


## 📁 Dataset Structure

```
datasets/
├── training_set_human.jsonl     # Human-authored posts (pre-LLM period, before Nov 2022)
├── training_set_aigc.jsonl      # AI-generated posts using 17 LLMs
└── exploration_set.jsonl        # Post-LLM period posts (2023-2025) for temporal analysis
```# RedNote-Vibe Dataset

**A Dataset for Capturing Temporal Dynamics of AI-Generated Text in Social Media** 

![Dataset Overview](assets/overview.jpg)

## Abstract

The proliferation of Large Language Models (LLMs) has led to widespread AI-Generated Text (AIGT) on social media platforms, creating unique challenges where content dynamics are driven by user engagement and evolve over time. However, existing datasets mainly depict static AIGT detection. In this work, we introduce **RedNote-Vibe**, the first longitudinal (5-years) dataset for social media AIGT analysis. This dataset is sourced from Xiaohongshu platform, containing user engagement metrics (e.g., likes, comments) and timestamps spanning from the pre-LLM period to July 2025, which enables research into the temporal dynamics and user interaction patterns of AIGT.


## Dataset Structure

```
datasets/
├── training_set_human.jsonl     # Human-authored posts (pre-LLM period, before Nov 2022)
├── training_set_aigc.jsonl      # AI-generated posts using 17 LLMs
└── exploration_set.jsonl        # Post-LLM period posts (2023-2025) for temporal analysis
```

### Data Format

Each entry contains the following fields:

```json
{
  "note_title": "Post title",
  "local_time": "Publication timestamp (YYYYMMDDHH)",
  "note_content": "Main text content",
  "likes": 123,
  "collections": 45,
  "comments": 67,
  "domain": "Content category",
  "model_family": "AI provider (for AIGC data)",
  "model": "Specific AI model (for AIGC data)"
}
```

## 🚀 Quick Start

### Prerequisites

```bash
pip install torch transformers scikit-learn pandas numpy datasets
```

### Basic Usage

Run the quick start example to train a BERT model for AIGT detection:

```bash
python quick_start.py
```

This script will:
1. Load the human and AI-generated datasets
2. Prepare balanced training/validation/test splits
3. Fine-tune a BERT model for binary classification
4. Evaluate performance and show prediction examples
5. Save the trained model for future use


## Supported Tasks

### 1. AIGT Classification (Binary)
Distinguish between human-written and AI-generated content.

### 2. AI Provider Identification (6-way)
Identify the source among six major AI providers:
- OpenAI (GPT-3.5, GPT-4, GPT-4o, GPT-o3)
- Google (Gemini-1.5, Gemini-2.0)
- Anthropic (Claude-3.5-Sonnet, Claude-3.5-Haiku)
- DeepSeek (DeepSeek-V2.5, DeepSeek-V3)
- Alibaba (Qwen-2.5)
- Others (Llama-3.1, etc.)

### 3. Model Identification (17-way)
Fine-grained identification among 17 specific AI models.



## Citation

If you use this dataset in your research, please cite our paper:

```bibtex

```

## License

This dataset is released under the [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) License.


### Data Format

Each entry contains the following fields:

```json
{
  "note_title": "Post title",
  "local_time": "Publication timestamp (YYYYMMDDHH)",
  "note_content": "Main text content",
  "likes": 123,
  "collections": 45,
  "comments": 67,
  "domain": "Content category",
  "model_family": "AI provider (for AIGC data)",
  "model": "Specific AI model (for AIGC data)"
}
```

## 🚀 Quick Start

### Prerequisites

```bash
pip install torch transformers scikit-learn pandas numpy datasets
```

### Basic Usage

Run the quick start example to train a BERT model for AIGT detection:

```bash
python quick_start.py
```

This script will:
1. Load the human and AI-generated datasets
2. Prepare balanced training/validation/test splits
3. Fine-tune a BERT model for binary classification
4. Evaluate performance and show prediction examples
5. Save the trained model for future use


## Supported Tasks

### 1. AIGT Classification (Binary)
Distinguish between human-written and AI-generated content.

### 2. AI Provider Identification (6-way)
Identify the source among six major AI providers:
- OpenAI (GPT-3.5, GPT-4, GPT-4o, GPT-o3)
- Google (Gemini-1.5, Gemini-2.0)
- Anthropic (Claude-3.5-Sonnet, Claude-3.5-Haiku)
- DeepSeek (DeepSeek-V2.5, DeepSeek-V3)
- Alibaba (Qwen-2.5)
- Others (Llama-3.1, etc.)

### 3. Model Identification (17-way)
Fine-grained identification among 17 specific AI models.



## Citation

If you use this dataset in your research, please cite our paper:

```bibtex

```

## License

This dataset is released under the [MIT License](LICENSE).
