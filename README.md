# PH-LLM: **Public Health Large Language Models for Infoveillance**

Here we introduce Public Health Large Language Models for Infoveillance (PH-LLM). PH-LLM can analyze most public health topics on multilingual social networks without the need for task-specific data annotation and model finetuning, with state-of-the-art zero-shot performance. Annotation and finetuning are time consuming, result in the lack of real-time infoveillance and delay in evidence-based health policy adjustment and refinement.

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
