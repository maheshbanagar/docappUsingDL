import streamlit as st
import PyPDF2
from transformers import AutoTokenizer,BertForQuestionAnswering #AutoModelForQuestionAnswering
import torch



def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfFileReader(file)
    text = ""
    for page in range(pdf_reader.numPages):
        text += pdf_reader.getPage(page).extract_text()
    return text


def answer_question(context, question):
    tokenizer = AutoTokenizer.from_pretrained("deepset/bert-base-cased-squad2")
    model = BertForQuestionAnswering.from_pretrained("deepset/bert-base-cased-squad2")

    inputs = tokenizer.encode_plus(question, context, return_tensors="pt")
    input_ids = inputs["input_ids"].tolist()[0]

    text_tokens = tokenizer.convert_ids_to_tokens(input_ids)
    print(len(text_tokens))
    answer_start_scores, answer_end_scores = model(**inputs).values()

    answer_start = torch.argmax(answer_start_scores)
    answer_end = torch.argmax(answer_end_scores) + 1

    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
    return answer


def main():
    st.title("PDF Document Parser and QA Chatbot")

    
    file = st.file_uploader("Upload a PDF document", type="pdf")

    if file is not None:
        
        text = extract_text_from_pdf(file)
        
        
        st.subheader("Extracted Text:")
        st.text(" ".join(text.split()))
        
        
        st.subheader("Extracted text will be context,Now You can ask questions:")
        question = st.text_input("Enter your question:")
        
        try:
            if question:
                answer = answer_question(text, question)
                st.write("Answer:", answer)
        except Exception as e:
            st.write(e)

if __name__ == "__main__":
    main()
    
    
    
#layout LM