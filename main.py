import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai import OpenAI
from pinecone import Pinecone
import ollama
import os
import requests
PINECONE_API_KEY=os.getenv("PINECONE_API_KEY")

client=OpenAI(base_url="http://localhost:11434/v1",api_key="ollama")
pc=Pinecone(api_key=PINECONE_API_KEY)

index=pc.Index("rag-python-pdf")


def extract_text_from_pdf(pdf_path):
    with open(pdf_path,"rb") as file:
        reader=PyPDF2.PdfReader(file)
        text=''
        for page in reader.pages:
            text+=page.extract_text()
        return text

def chunk_text(text,chunk_size=1000,chunk_overlap=200):
    text_splitter=RecursiveCharacterTextSplitter(
        separators=["\n\n","\n",".",","," "],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks=text_splitter.split_text(text)
    return chunks

def generate_embeddings(texts):
    embeddings = []
    for text in texts:
        response = requests.post(
            "http://localhost:11434/api/embed",
            json={"input": text, "model": "mxbai-embed-large"}
        )
        data = response.json()
        embeddings.append(data['embeddings'][0])
    return embeddings

def upsert_to_pinecone(chunks, embeddings):
    vectors = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        vectors.append({
            "id": f"chunk_{i}",
            "values": embedding,
            "metadata": {"text": chunk}
        })
    index.upsert(vectors=vectors, namespace="rag-python-pdf")
    print('upserted to pinecone for vectors', vectors)

# pdf_path="ai_evolution_summary.pdf"
# pdf_path="artificial_intelligence_tutorial.pdf"
# text=extract_text_from_pdf(pdf_path)
# print('this is the text',text)
# print('--------------------------------')
# chunks=chunk_text(text)
# print('--------------------------------')
# print('this are the chunks',chunks)
# print('--------------------------------')

# embeddings=generate_embeddings(chunks)
# print('this are the embeddings',embeddings)
# print('--------------------------------')
# upsert_to_pinecone(chunks,embeddings)

def query_rag(query):
    query_vector=generate_embeddings([query])
    results=index.query(
        vector=query_vector,
        top_k=15,
        include_metadata=True,
        namespace="rag-python-pdf"
    )
    context='\n'.join([result['metadata']['text'] for result in results['matches']])
    prompt=f"""
    You are a helpful assistant that can answer questions about the following text:
    {context}
    Question: {query}
    Answer:
    """

    response=client.chat.completions.create(
        model="gemma3:4b",
        messages=[{"role":"system","content":"You are a helpful assistant that can answer questions about the following text:"},
                  {"role":"user","content":prompt}]
    )
    return response.choices[0].message.content

    



def main():
    print('--------------------------------')
    print('RAG PDF')
    print('--------------------------------')
    print("Hello from rag-pdf!")

    while True:
        query=input("Enter your query: ")
        if query.lower() in ["exit","quit","bye"]:
            break
        context=query_rag(query)
        print('--------------------------------')
        print('Answer from Assistant:',context)
        print('--------------------------------')


if __name__ == "__main__":
    main()
