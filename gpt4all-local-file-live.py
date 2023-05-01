# Local gpt4all LLM Query system wuth access to local files. (No remote or cloud systems used)
#
# Start here : https://github.com/nomic-ai/gpt4all
# Code taken/ideas from:
#
# https://clemenssiebler.com/posts/chatting-private-data-langchain-azure-openai-service/
# https://blog.ouseful.info/2023/04/04/langchain-query-gpt4all-against-knowledge-source/
#
# model : https://github.com/nomic-ai/gpt4all-chat#manual-download-of-models
# 
# Save your short text file in store/txt
# There are a number of Python libraries needed 
# 
# pip3 install langchain tiktoken pyllamacpp pyllama transformers pygpt4all llama-cpp-python faiss-cpu 


GPT4ALL_MODEL_PATH = "./models/ggml-gpt4all-l13b-snoozy.bin"
from langchain.llms import LlamaCpp

#v Generate text
#response = model("Once upon a time, ")

from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import TextLoader
from langchain.text_splitter import TokenTextSplitter

from langchain.embeddings import LlamaCppEmbeddings
embeddings = LlamaCppEmbeddings(model_path=GPT4ALL_MODEL_PATH)


loader = DirectoryLoader('store/txt', glob="*.txt", loader_cls=TextLoader)

documents = loader.load()
text_splitter = TokenTextSplitter(chunk_size=100, chunk_overlap=0)
docs = text_splitter.split_documents(documents)


from langchain.vectorstores import FAISS
db = FAISS.from_documents(documents=docs, embedding=embeddings)

from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate



# Adapt if needed
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template("""Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:""")


llm = LlamaCpp(model_path=GPT4ALL_MODEL_PATH)
# added search_kwargs={"k": 1} to avoid "ValueError: Requested tokens exceed context window of 512"
qa = ConversationalRetrievalChain.from_llm(llm=llm,
                                           retriever=db.as_retriever(search_kwargs={"k": 1}),
                                           condense_question_prompt=CONDENSE_QUESTION_PROMPT,
                                           return_source_documents=True,
                                           verbose=False)

chat_history = []
query = """What did Obama do in his first 100 days ? Provide the answer in the form: 

- ITEM 1
- ITEM 2
- ITEM 3"""
result = qa({"question": query, "chat_history": chat_history})

print("Question:", query)
print("Answer:", result["answer"])
