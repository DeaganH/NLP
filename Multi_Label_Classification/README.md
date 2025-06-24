# Multi-Label Emotion Classification with BERT

An example of multi-label classification using transfer learning from BERT for emotion detection in text.

There is also a demonstration as to how you can use a local LLM to generate additional labels which you can then use to further finetune or augment existing datasets.

## 📋 Overview

This project demonstrates multi-label emotion classification on Twitter data using BERT (Bidirectional Encoder Representations from Transformers). The implementation includes:

- Transfer learning with pre-trained BERT models
- Multi-label classification for 11 different emotions
- Data augmentation using local LLMs
- Label validation and re-classification techniques

## 🎯 Dataset

The project uses a Twitter emotion classification dataset with the following emotions:
- anger
- anticipation  
- disgust
- fear
- joy
- love
- optimism
- pessimism
- sadness
- surprise
- trust

Each tweet can be associated with multiple emotions (multi-label classification).

## 🚀 Features

### Core Implementation
- **BERT-based Classification**: Uses `bert_en_uncased_L-12_H-768_A-12_4` model for transfer learning
- **Text Preprocessing**: Emoji conversion, text cleaning, and normalization
- **Multi-label Support**: Handles classification across 11 emotion categories simultaneously
- **Performance Metrics**: Accuracy and F1-score evaluation per emotion and macro-averaged

### Advanced Techniques
- **LLM-based Data Augmentation**: Uses Qwen 7B local LLM to generate additional training samples
- **Label Validation**: Validates existing labels using LLM for logical consistency
- **Data Re-classification**: Improves label quality through LLM-assisted re-labeling

## 📂 Project Structure

```
Multi_Label_Classification/
├── README.md
└── notebook/
    ├── EmotionClassification_with_BERT.ipynb    # Main classification notebook
    ├── Validating Labels Using an LLM.ipynb    # LLM-based data augmentation
    └── official/                                # TensorFlow Model Garden components
        ├── benchmark/                           # BERT benchmarking utilities
        ├── nlp/                                # NLP modeling components
        └── ...
```

## 🛠️ Technical Requirements

- TensorFlow 2.x
- TensorFlow Hub
- TensorFlow Text
- Transformers (for LLM components)
- PyTorch (for local LLM)
- pandas, numpy, scikit-learn
- emoji (for emoji processing)

## 📋 Installation & Setup

### Prerequisites
- Python 3.7+
- GPU support recommended for training

### Environment Setup
```bash
pip install tensorflow>=2.8.0
pip install tensorflow-hub
pip install tensorflow-text
pip install transformers
pip install torch
pip install pandas numpy scikit-learn
pip install emoji
```

### Dataset Preparation
1. Download the Twitter emotion dataset (E-c-En-train.csv, E-c-En-dev.csv)
2. Place the dataset files in the `datasets/` directory
3. The dataset should contain:
   - `Tweet`: Text content
   - Emotion columns: Binary labels for each of the 11 emotions

## 🔬 Model Architecture

### BERT Configuration
- **Model**: `bert_en_uncased_L-12_H-768_A-12_4` from TensorFlow Hub
- **Architecture**: 12 layers, 768 hidden units, 12 attention heads
- **Parameters**: ~110M parameters
- **Input**: Preprocessed tweet text (max length: 512 tokens)
- **Output**: 11 binary classifiers (one per emotion)

### Data Preprocessing Pipeline
1. **Emoji Conversion**: Emojis → descriptive text using `emoji.demojize()`
2. **Text Cleaning**: 
   - Lowercase conversion
   - Mention removal (`@username`)
   - Contraction handling
   - Special character normalization
   - Number removal
   - Whitespace normalization

### Training Configuration
- **Optimizer**: Adam with learning rate scheduling
- **Loss Function**: Binary crossentropy (per emotion)
- **Batch Size**: 32 (configurable)
- **Epochs**: Variable based on convergence
- **Validation**: Per-emotion accuracy and F1-score

## 📈 Performance Analysis

### Emotion-wise Performance (Original Dataset)
| Emotion | Support | Accuracy | F1-Score |
|---------|---------|----------|----------|
| anger | 1007 | 89.2% | 59.3% |
| anticipation | 705 | 86.9% | 42.8% |
| disgust | 795 | 90.3% | 50.8% |
| fear | 1045 | 82.5% | 67.4% |
| joy | 2137 | 86.2% | 80.5% |
| love | 833 | 89.2% | 59.3% |
| optimism | 1218 | 81.2% | 70.1% |
| pessimism | 795 | 87.2% | 36.2% |
| sadness | 2008 | 88.1% | 80.5% |
| surprise | 361 | 94.9% | 7.5% |
| trust | 357 | 94.8% | 3.8% |

### Key Observations
- **High-performing emotions**: joy, sadness (high support + clear linguistic indicators)
- **Challenging emotions**: surprise, trust (low F1 despite high accuracy due to class imbalance)
- **Moderate performers**: fear, optimism (balanced support with reasonable F1-scores)

## 🤖 LLM Data Augmentation Details

### Local LLM Setup (Qwen 7B)
- **Model**: Qwen-7B-Chat version
- **Purpose**: Generate synthetic tweets and validate existing labels
- **Hardware**: CUDA-compatible GPU recommended (8GB+ VRAM)

### Augmentation Strategy
1. **Underrepresented Emotion Targeting**: Focus on emotions with <500 training samples
2. **Synthetic Data Generation**: Create diverse tweets across different contexts:
   - Personal experiences
   - News/events
   - Social interactions
   - Aspirational content
3. **Label Validation**: Re-classify existing labels for logical consistency
4. **Quality Control**: Manual review of generated samples

### Enhancement Results
- **Training samples increased by**: ~25%
- **Macro F1-Score improvement**: +12 percentage points
- **Most improved emotions**: pessimism (+15%), surprise (+20%), trust (+30%)
 
## 📚 References & Citation

### Academic Papers
- Devlin, J., et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
- Mohammad, S., et al. (2018). SemEval-2018 Task 1: Affect in Tweets

### Datasets
- **Twitter Emotion Dataset**: [Kaggle Competition](https://www.kaggle.com/competitions/tweet-sentiment-extraction)
- **SemEval-2018 Task 1**: Affect in Tweets

### Model Resources
- **TensorFlow Hub**: [BERT Models](https://tfhub.dev/s?q=bert)
- **Hugging Face**: [Qwen Models](https://huggingface.co/Qwen)

## 📄 License & Model Usage Rights

### Project Code License
This project code is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Open Source Model Licensing

#### BERT Models (Google Research)
- **License**: Apache 2.0 License
- **Commercial Use**: ✅ **Fully Permitted**
- **Redistribution**: ✅ **Allowed with attribution**
- **Modification**: ✅ **Permitted**
- **Patent Rights**: ✅ **Granted**
- **Attribution Required**: ✅ **Yes**

**Key Rights with BERT:**
- Free to use for any purpose (personal, academic, commercial)
- Can modify and redistribute the models
- Can integrate into proprietary software
- Can create derivative works
- No royalty payments required

**Attribution Requirements:**
- Include Apache 2.0 license text
- Provide attribution to Google Research
- Cite the original BERT paper:
  ```
  @article{devlin2018bert,
    title={BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding},
    author={Devlin, Jacob and Chang, Ming-Wei and Lee, Kenton and Toutanova, Kristina},
    journal={arXiv preprint arXiv:1810.04805},
    year={2018}
  }
  ```

#### Qwen Models (Alibaba Cloud)
- **License**: Dual licensing scheme (size-dependent)
- **Qwen-7B/14B/72B**: Tongyi Qianwen LICENSE AGREEMENT (Commercial use requires approval)
- **Qwen-1.8B**: Tongyi Qianwen RESEARCH LICENSE AGREEMENT (Research only)
- **Source Code**: Apache 2.0 License

**Usage Rights by Model Size:**

**Qwen-7B, Qwen-14B, Qwen-72B:**
- **Research Use**: ✅ **Free without restrictions**
- **Personal Use**: ✅ **Permitted**
- **Commercial Use**: ⚠️ **Requires License Application**
- **Attribution Required**: ✅ **Yes**

**Qwen-1.8B:**
- **Research Use**: ✅ **Free for academic/research purposes**
- **Personal Use**: ✅ **Permitted for non-commercial purposes**
- **Commercial Use**: ❌ **Not permitted without separate agreement**
- **Attribution Required**: ✅ **Yes**

**Commercial Use Process for Qwen:**
1. **For Qwen-7B**: Apply at [Qwen-7B Commercial Form](https://dashscope.console.aliyun.com/openModelApply/qianwen)
2. **For Qwen-14B**: Apply at [Qwen-14B Commercial Form](https://dashscope.console.aliyun.com/openModelApply/Qwen-14B-Chat)
3. **For Qwen-72B**: Apply at [Qwen-72B Commercial Form](https://dashscope.console.aliyun.com/openModelApply/Qwen-72B-Chat)
4. **For Qwen-1.8B**: Contact Alibaba directly for commercial licensing

### Practical Usage Guidelines

#### ✅ What You Can Do Freely:
- Use BERT models for any purpose without restrictions
- Use Qwen models for research and personal projects
- Modify and fine-tune both model types
- Publish research papers using these models
- Create educational content and tutorials
- Distribute modified versions (with proper attribution)

#### ⚠️ What Requires Additional Steps:
- **Commercial products using Qwen models**: Need license approval
- **Large-scale commercial deployment of Qwen**: Requires formal agreement
- **Revenue-generating applications with Qwen**: Need commercial license

#### ❌ What You Cannot Do:
- Remove or modify license attributions
- Use Qwen-1.8B in commercial products without agreement
- Claim ownership of the original models
- Use models for illegal or harmful purposes

### Model Attribution Template

When using these models, include appropriate attribution:

## Model Attribution

This project uses the following open source models:

### BERT
- **Model**: BERT (Bidirectional Encoder Representations from Transformers)
- **Source**: Google Research
- **License**: Apache 2.0
- **Citation**: Devlin et al. (2018) - BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding

### Qwen
- **Model**: Qwen-7B Large Language Model
- **Source**: Alibaba Cloud / QwenLM
- **License**: Tongyi Qianwen LICENSE AGREEMENT
- **Usage**: Research and educational purposes
- **Citation**: Qwen Technical Report (2023)

### Compliance Recommendations

1. **For Academic/Research Use**: Both models are freely available
2. **For Personal Projects**: Both models can be used without restrictions
3. **For Commercial Applications**: 
   - BERT: Use freely with attribution
   - Qwen: Apply for commercial license first
4. **For Open Source Projects**: Both models are suitable with proper attribution
5. **For Enterprise Deployment**: Review license terms and apply for Qwen commercial license if needed

### Legal Disclaimer

This summary is provided for informational purposes only and does not constitute legal advice. Always review the original license agreements for authoritative terms and conditions. For commercial use of Qwen models, consult with Alibaba Cloud directly or seek legal counsel.