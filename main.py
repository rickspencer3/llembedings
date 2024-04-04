from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import GPT4All
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain.vectorstores.faiss import FAISS
from flask import Flask, render_template, request

embeddings = GPT4AllEmbeddings()
faiss_index = FAISS.load_local("/", embeddings, allow_dangerous_deserialization=True)
app = Flask(__name__)
llm = GPT4All(model="/models/mistral-7b-openorca.gguf2.Q4_0.gguf", max_tokens=20000)

@app.route('/render', methods=['POST'])
def invoke():
    question = request.form["input"]
    matched_docs = faiss_index.similarity_search(question, 4)
    context = ""
    sources = set()
    for doc in matched_docs:
        context = context + doc.page_content + " \n\n " 
        sources.add(doc.metadata["source"])

    template = """
    Use the following context to respond to the question.
    Context: {context}
    - -
    Question: {question}
    """
    prompt = PromptTemplate(template=template, input_variables=["context", "question"]).partial(context=context)
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    answer = llm_chain.invoke(question)
    return render_template('answer.html', question=question, answer=answer['text'], context=sources)

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)