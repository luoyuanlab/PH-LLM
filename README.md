# PH-LLM: **Public Health Large Language Models for Infoveillance**

Here we introduce Public Health Large Language Models for Infoveillance (PH-LLM). PH-LLM can analyze most public health topics on multilingual social networks without the need for task-specific data annotation and model finetuning, with state-of-the-art zero-shot performance. Annotation and finetuning are time consuming, result in the lack of real-time infoveillance and delay in evidence-based health policy adjustment and refinement.

Check out a simple demo on how to use our model with QLoRA [here](https://colab.research.google.com/drive/1pqiTiwJ1ocr0sRxslBtQEEQHOBlxvmkn?usp=sharing).
We used the GitHub repository [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) to run our experiments. It is also a useful tool for running our models, and we highly recommend checking it out.

In this repository, we present [our models](https://huggingface.co/collections/xinyuzhou/ph-llm-689de6703968d3a550e980ca) and the code for generating our datasets. Our preprint is available at [medRxiv](https://pmc.ncbi.nlm.nih.gov/articles/PMC11844576/).

## Models

| Model                                                    | LoRA weights                                                       | Backbone model        |
| -------------------------------------------------------- | ------------------------------------------------------------------ | --------------------- |
| [PH-LLM-32B](https://huggingface.co/xinyuzhou/PH-LLM-32B)   | [PH-LLM-32B-LoRA](https://huggingface.co/xinyuzhou/PH-LLM-32B-LoRA)   | Qwen2.5-32B-Instruct  |
| [PH-LLM-14B](https://huggingface.co/xinyuzhou/PH-LLM-14B)   | [PH-LLM-14B-LoRA](https://huggingface.co/xinyuzhou/PH-LLM-14B-LoRA)   | Qwen2.5-14B-Instruct  |
| [PH-LLM-7B](https://huggingface.co/xinyuzhou/PH-LLM-7B)     | [PH-LLM-7B-LoRA](https://huggingface.co/xinyuzhou/PH-LLM-7B-LoRA)     | Qwen2.5-7B-Instruct   |
| [PH-LLM-3B](https://huggingface.co/xinyuzhou/PH-LLM-3B)     | [PH-LLM-3B-LoRA](https://huggingface.co/xinyuzhou/PH-LLM-3B-LoRA)     | Qwen2.5-3B-Instruct   |
| [PH-LLM-1.5B](https://huggingface.co/xinyuzhou/PH-LLM-1.5B) | [PH-LLM-1.5B-LoRA](https://huggingface.co/xinyuzhou/PH-LLM-1.5B-LoRA) | Qwen2.5-1.5B-Instruct |
| [PH-LLM-0.5B](https://huggingface.co/xinyuzhou/PH-LLM-0.5B) | [PH-LLM-0.5B-LoRA](https://huggingface.co/xinyuzhou/PH-LLM-0.5B-LoRA) | Qwen2.5-0.5B-Instruct |

## 1. Training sets

### 1.1 Infoveillance Instructions ($I^2$)

#### 1.1.1 Vaccine Attitudes

| Dataset (short name + full name) | Paper Link                                       | Download Link          | Language | Size   |
| -------------------------------- | ------------------------------------------------ | ---------------------- | -------- | ------ |
| WHV (Weibo HPV vaccine)          | [link](https://doi.org/10.1101/2023.12.07.23299667) | Not publicly available | Chinese  | 23,000 |
| TCV (Twitter COVID-19 vaccine)   | [link](https://doi.org/10.2471/BLT.23.289682)       | Not publicly available | English  | 53,000 |

#### 1.1.2 Mental Health

| Dataset (short name + full name)      | Paper Link                                        | Download Link                                                                                                                                         | Language      | Size   |
| ------------------------------------- | ------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- | ------------- | ------ |
| WCE (Weibo COVID emotion)             | [link](https://doi.org/10.1007/978-3-030-60450-9_56) | [GitHub](https://github.com/COVID-19-Weibo-data/COVID-19-sentiment-analysis-dataset-Weibo/blob/master/%E6%83%85%E6%84%9F%E8%AE%AD%E7%BB%83%E9%9B%86.csv) | Chinese       | 10,500 |
| SR (Stress – Reddit)                 | [link](https://doi.org/10.48550/arXiv.1911.00133)    | [columbia.edu](http://www.cs.columbia.edu/~eturcan/data/dreaddit.zip)                                                                                    | English       | 3,000  |
| DR (Depression – Reddit)             | [link](https://doi.org/10.1145/3485447.3512128)      | [GitHub](https://github.com/usmaann/Depression_Severity_Dataset)                                                                                         | English       | 500    |
| PEH (Perceived Emotions in Hurricane) | [link](https://doi.org/10.48550/arXiv.2004.14299)    | [GitHub](https://github.com/shreydesai/hurricane)                                                                                                        | English       | 10,000 |
| UEC (Urdu Emotion Classification)     | [link](https://doi.org/10.1145/3574318.3574327)      | [GitHub](https://sites.google.com/view/multi-label-emotionsfire-task/dataset)                                                                            | Urdu          | 6,000  |
| SemEval-2020 Task 9                   | [link](https://doi.org/10.48550/arXiv.2008.04277)    | [Zenodo](https://zenodo.org/records/3974927#.XyxAZCgzZPZ)                                                                                                | Hindi-English | 12,000 |
| TO (Twitter Optimists)                | [link](https://doi.org/10.18653/v1/P16-2052)         | [umich.edu](https://lit.eecs.umich.edu/downloads.html#Twitter%20Optimism%20Dataset)                                                                      | English       | 6,000  |
| VT (Vulgarity on Twitter)             | [link](https://aclanthology.org/C18-1248)            | [GitHub](https://github.com/ericholgate/vulgartwitter/tree/master/data)                                                                                  | English       | 2,500  |

#### 1.1.3 Nonpharmacological Interventions

| Dataset (short name + full name) | Paper Link                         | Download Link          | Language | Size    |
| -------------------------------- | ---------------------------------- | ---------------------- | -------- | ------- |
| WCT (Weibo COVID test)           | [link](https://doi.org/10.2196/26895) | Not publicly available | Chinese  | 115,000 |

#### 1.1.4 Hate Speech & Offensive Language

| Dataset (short name + full name)                          | Paper Link                                         | Download Link                                                                                                       | Language                    | Size   |
| --------------------------------------------------------- | -------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- | --------------------------- | ------ |
| IHS (Indonesian Hate Speech)                              | [link](https://doi.org/10.18653/v1/W19-3506)          | [GitHub](https://github.com/okkyibrohim/id-multi-label-hate-speech-and-abusive-language-detection)                     | Indonesian                  | 10,000 |
| BHS (Bengali Hate Speech)                                 | [link](https://doi.org/10.48550/arXiv.2012.09686)     | [Kaggle](https://www.kaggle.com/datasets/naurosromim/bengali-hate-speech-dataset)                                      | Bengali                     | 20,000 |
| KHS (Korean Hate Speech)                                  | [link](https://doi.org/10.48550/arXiv.2005.12503)     | [GitHub](https://github.com/kocohub/korean-hate-speech)                                                                | Korean                      | 8,500  |
| ToLD-BR (Brazilian Portuguese Toxic Language)             | [link](https://doi.org/10.48550/arXiv.2010.04543)     | [GitHub](https://github.com/JAugusto97/ToLD-Br)                                                                        | Portuguese                  | 11,000 |
| YAB (YouTube Anti-social Behavior)                        | [link](https://doi.org/10.1016/j.procs.2018.10.473)   | [OneDrive](https://onedrive.live.com/?authkey=!ACDXj_ZNcZPqzy0&id=6EF6951FBF8217F9!105&cid=6EF6951FBF8217F9)           | Arabic                      | 5,000  |
| AD (Aggression Detection)                                 | [link](https://doi.org/10.48550/arXiv.1803.09402)     | [GitHub](https://github.com/SilentFlame/AggressionDetection/blob/master/DataPre-Processing/processedDataWithoutID.txt) | Hindi-English               | 8,000  |
| UTT (Urdu Threatening Tweets)                             | [link](https://doi.org/10.1145/3574318.3574327)       | [GitHub](https://sites.google.com/view/multi-label-emotionsfire-task/dataset)                                          | Urdu                        | 4,000  |
| HSTW (Hate Speech from Twitter/Whisper)                   | [link](https://doi.org/10.1145/3078714.3078723)       | [GitHub](https://github.com/Mainack/hatespeech-data-HT-2017)                                                           | English                     | 1,000  |
| HSOL (Hate Speech and Offensive Language)                 | [link](https://doi.org/10.48550/arXiv.1703.04009)     | [GitHub](https://github.com/t-davidson/hate-speech-and-offensive-language/tree/master/data)                            | English                     | 48,000 |
| SemEval-2016 Task 6                                       | [link](https://aclanthology.org/S16-1003/)            | [saifmohammad.com/](https://www.saifmohammad.com/WebPages/StanceDataset.htm)                                           | English                     | 600    |
| SemEval-2019 Task 6                                       | [link](https://doi.org/10.48550/arXiv.1903.08983)     | [GitHub](https://github.com/ZeyadZanaty/offenseval/tree/master/datasets/training-v1)                                   | English                     | 7,900  |
| SemEval-2020 Task 12 (Subtask B, C)                       | [link](https://doi.org/10.18653/v1/2023.acl-short.66) | [Zenodo](https://zenodo.org/records/3950379#.XxZ-aFVKipp)                                                              | English                     | 3,000  |
| SemEval-2023 Task 10 (Subtask A)                          | [link](https://aclanthology.org/2023.semeval-1.305/)  | [GitHub](https://github.com/rewire-online/edos/tree/main/data)                                                         | English                     | 10,000 |
| SemEval-2023 Task 10 (Subtask B)                          | [link](https://aclanthology.org/2023.semeval-1.305/)  | [GitHub](https://github.com/rewire-online/edos/tree/main/data)                                                         | English                     | 2,000  |
| Let-Mi (Levantine Twitter Misogyny)                       | [link](https://doi.org/10.48550/arXiv.2103.10195)     | [Google Drive](https://drive.google.com/file/d/1mM2vnjsy7QfUmdVUpKqHRJjZyQobhTrW/view)                                 | Arabic                      | 5,500  |
| RP (Rheinische Post)                                      | [link](https://openreview.net/pdf?id=NfTU-wN8Uo)      | [Zenodo](https://zenodo.org/record/5291339)                                                                            | German                      | 3,000  |
| MLMA (MultiLingual and Multi-Aspect hate speech analysis) | [link](https://doi.org/10.48550/arXiv.1908.11049)     | [Hugging Face](https://huggingface.co/datasets/nedjmaou/MLMA_hate_speech)                                              | Arabic, French, and English | 4,000  |

#### 1.1.5 Misinformation

| Dataset (short name + full name)        | Paper Link                                        | Download Link                                                            | Language | Size  |
| --------------------------------------- | ------------------------------------------------- | ------------------------------------------------------------------------ | -------- | ----- |
| AFN (Arabic Fake News)                  | [link](https://doi.org/10.14569/IJACSA.2021.0120691) | [GitHub](https://github.com/yemen2016/FakeNewsDetection/tree/main)          | Arabic   | 1,500 |
| FC (Fact-Checking Public Health Claims) | [link](https://doi.org/10.48550/arXiv.2010.09926)    | [GitHub](https://github.com/neemakot/Health-Fact-Checking/tree/master/data) | English  | 8,000 |

### 1.1.6 Public Health QA

| Dataset (short name + full name) | Paper Link                                     | Download Link                                                                            | Language     | Size    |
| -------------------------------- | ---------------------------------------------- | ---------------------------------------------------------------------------------------- | ------------ | ------- |
| MedMCQA (Medical MCQA)           | [link](https://doi.org/10.48550/arXiv.2203.14371) | [Hugging Face](https://huggingface.co/datasets/openlifescienceai/medmcqa)                   | English      | 15,000  |
| MentalLLaMA QA                   | [link](https://arxiv.org/abs/2309.13567)          | [GitHub](https://github.com/SteveKGYang/MentalLLaMA)                                        | English      | 10,000  |
| PubMed Summarization             | [link](https://pubmed.ncbi.nlm.nih.gov/33085945/) | Not publicly available                                                                   | English      | 5,000   |
| Meadow (Medical Flashcards)      | [link](https://doi.org/10.48550/arXiv.2304.08247) | [Hugging Face](https://huggingface.co/datasets/medalpaca/medical_meadow_medical_flashcards) | English      | 1,400   |
| OpenOrca                         | [link](https://doi.org/10.48550/arXiv.2306.02707) | [Hugging Face](https://huggingface.co/datasets/Open-Orca/OpenOrca)                          | English      | 140,000 |
| Bactrian-X                       | [link](https://doi.org/10.48550/arXiv.2305.15011) | [Hugging Face](https://huggingface.co/datasets/MBZUAI/Bactrian-X)                           | 24 languages | 19,200  |

## 2. Evaluation datasets

### 2.1 English datasets

| Dataset (short name + full name)        | Paper Link                                      | Download Link                                                                                             | Language | Size   |
| --------------------------------------- | ----------------------------------------------- | --------------------------------------------------------------------------------------------------------- | -------- | ------ |
| CAVES (Concerns towards COVID Vaccines) | [link](https://doi.org/10.1145/3477495.3531745)    | [GitHub](https://github.com/sohampoddar26/caves-data)                                                        | English  | 13,839 |
| CC (COVID category)                     | [link](https://doi.org/10.3389/frai.2023.1023281)  | [GitHub](https://github.com/digitalepidemiologylab/covid-twitter-bert/tree/master/datasets/covid_category)   | English  | 704    |
| Ethos                                   | [link](https://doi.org/10.1007/s40747-021-00608-2) | [GitHub](https://github.com/intelligence-csd-auth-gr/Ethos-Hate-Speech-Dataset/tree/master/ethos/ethos_data) | English  | 200    |
| GHC (Gab Hate Corpus)                   | [link](https://doi.org/10.1007/s10579-021-09569-x) | [OSF](https://osf.io/edua3)                                                                                  | English  | 11,020 |
| MC (Misinformation during COVID-19)     | [link](https://doi.org/10.48550/arXiv.2008.00791)  | [Zenodo](https://doi.org/10.5281/zenodo.4024154)                                                             | English  | 3,055  |
| TCT (Twitter COVID-19 Test)             | [link](https://doi.org/10.2196/26895)              | Not publicly available                                                                                    | English  | 1,339  |

### 2.2 Multilingual datasets

| Dataset (short name + full name)                | Paper Link                                       | Download Link                                                                              | Language   | Size   |
| ----------------------------------------------- | ------------------------------------------------ | ------------------------------------------------------------------------------------------ | ---------- | ------ |
| AHSFN (Arabic COVID-19 Hate Speech & Fake News) | [link](https://doi.org/10.1016/j.procs.2021.05.086) | [GitHub](https://github.com/MohamedHadjAmeur/AraCOVID19-MFH)                                  | Arabic     | 14,943 |
| ITED (Indonesian Twitter Emotion Detection)     | [link](https://doi.org/10.1109/IALP.2018.8629262)   | [GitHub](https://github.com/meisaputri21/Indonesian-Twitter-Emotion-Dataset)                  | Indonesian | 881    |
| MAT (Misinformation on Arabic Twitter)          | [link](https://doi.org/10.48550/arXiv.2101.05626)   | [GitHub](https://github.com/SarahAlqurashi/COVID19-Misinformation-dataset-/tree/main/Dataset) | Arabic     | 1,284  |
| WCV (Weibo COVID-19 Vaccine)                    | [link](https://doi.org/10.2196/27632)               | Not publicly available                                                                     | Chinese    | 4,893  |

## Training and evaluation

In this study, we developed a novel suite of LLMs called PH-LLM, which is available in 6 sizes for various computing settings. PH-LLM models were instruction-based finetuned on top of **Qwen 2.5**, using 593,100 instruction tuning pairs based on 30 infoveillance datasets with a total of 96 public health infoveillance tasks and six question-answering datasets. The PH-LLM models were then evaluated in 39 multilingual tasks which were not shown to the model during the instruction-tuning process. We evaluated the performance of PH-LLM against state-of-the-art LLMs, including ChatGPT-4o, Llama-3.1-70B-Instruct, Mistral-Large-Instruct-2407, and Qwen2.5-72B-Instruct.

## Instruction-tuning hyperparameters

- **Training method**: QLoRA
- **Effective batch size**: 256
- **Epochs**: 3
- **Cut-off length**: 1024 tokens
- **Learning rate**: 0.00005, with cosine annealing and a warm-up ratio of 0.1
- **LoRA Plus learning rate ratio**: 16

## Legal disclaimer

This project and the associated PH-LLM models (hereinafter "the Models") are provided "as is" without any guarantees or warranties of any kind, either express or implied, including but not limited to warranties of accuracy, reliability, suitability for a particular purpose, or non-infringement.
The Models are intended solely for research, educational, or informational purposes. They should not be used in production environments without further validation and testing for safety and appropriateness for the intended task.
In no event shall the authors, contributors, or affiliated institutions be liable for any direct, indirect, incidental, special, exemplary, or consequential damages (including but not limited to procurement of substitute goods or services; loss of use, data, or profits; or business interruption) however caused and on any theory of liability, whether in contract, strict liability, or tort (including negligence or otherwise) arising in any way out of the use of the Model, even if advised of the possibility of such damage.
Users are responsible for evaluating and verifying the suitability of the Model for their particular use case. By using the Model, you agree that any consequences, including but not limited to ethical, legal, or operational outcomes, arising from the use of this model are solely your responsibility.
The authors strongly advise users to comply with all applicable laws, regulations, and ethical standards when using this Model. Misuse of the Model, including but not limited to its use for harmful or malicious purposes, is strictly prohibited.
This Model may rely on third-party libraries, data, or platforms. The authors make no representations or warranties regarding the security, accuracy, or reliability of these third-party components and disclaim any liability arising from their use.
Use of this Model does not imply endorsement by the authors or affiliated institutions for any specific purpose or application. Any outcomes resulting from the use of this Model are the sole responsibility of the user.
Contributions to this project, including modifications or enhancements, are welcomed under the terms of the applicable open-source license. However, contributors are also responsible for ensuring their contributions comply with applicable laws and standards.
