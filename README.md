
# DocApp using DeepLearning

Built a streamlit tool, which can parse documents in PDF documents, extract information and build a QA system using deep learning models (pre-trained models from HuggingFace).


## Transformers
There are three types of architectures in Transformers

- Encoder : suited for task requiring an understanding of the  full sentence, such as sentence classification, NER,extractive question Answering.
Example: BERT 

- Decoder : suited for task involving text generation. 
Example: GPT-3

- Encoder-Decoder : suited for tasks around generating new sentences depending on a given input such as summarization, Translation or generating question Answering. 
Example: t5, multilingual-mt5

## Model Building
BERT(Bidirectional Encoding Representational Transformer) model satisties the problem statement.

Majorly there are two important steps for model Building

- Pre-Training the model (Available in HuggingFace)


- Fine- Tuning the model ( Need to do train model by adding extra contextual information)

![pretraining and finetuning](https://blog.paperspace.com/content/images/size/w1000/2020/08/BERT.jpg)
## HuggingFace

For extractive question answering, where the model selects a span of text from the context as the answer to the given question, some of the best-performing pretrained models include:

- BERT,RoBERTa,DistilBERT,ALBERT,ELECTRA,TAPAS,MPNet 

Bert is common and most used for question answering

## Implementation

- Create a virtual environment 

- Install necessary libraries
```bash
  pip install streamlit transformers torch pdf2image
```
- Gain a detailed understanding of bert-base-uncased model architecture using this [link](https://huggingface.co/bert-base-uncased)


- Run the application
```bash
 streamlit run <fileName.py>
```
- Upload a pdf document in streamlit window. Automatically it extracts text and display the extracted text

- Now, user can enter any question related the extracted text in the input field .Press Enter to get the answer

- User can ask questions more number of times to the bot in the input field.
## References

- [DocVQA: A Dataset for VQA on Document Images](https://openaccess.thecvf.com/content/WACV2021/papers/Mathew_DocVQA_A_Dataset_for_VQA_on_Document_Images_WACV_2021_paper.pdf)  research paper on visual question anwering and usage of bert model and its children model.

- [How to Train A Question-Answering Machine Learning Model(BERT)](https://blog.paperspace.com/how-to-train-question-answering-machine-learning-models/#:~:text=Question%2DAnswering%20Models%20are%20machine,given%20options%2C%20and%20so%20on.) gives a detailed understanding of how bert pretraining model works.

