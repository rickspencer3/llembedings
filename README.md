A project for knowledge embedding from a set of pdf files. It is designed so that the container will not need to use the network at all while building (though it still uses pip install). The two main technologies in use are:
 * langchain, for gluing everything together
 * gpt4all for the models

Very much a work in prorgress.

I threw a terrible web app in front of it to facilitate playing around with the content. There is nothing fancy like remember context or anything like that yet, but it shouldn't be too hard to add, since I believe that is all built into langchain.

# setup
1. create a folder called pdfs, and put the pdfs that you want to index in there.
2. Download mistral-7b-openorca.gguf2.Q4_0.gguf and all-MiniLM-L6-v2-f16.gguf from gpt4all and store them in the root directory.

# build and run
```
sudo docker build -t llm .
sudo docker run -p 5001:5001 llm
```

Then wait until the web app is ready. You will see this:
```bash
WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
 * Running on all addresses (0.0.0.0)
 * Running on http://127.0.0.1:5001
 * Running on http://172.17.0.2:5001
Press CTRL+C to quit
```
In your web browser go to localhost:5001. Enter a query, and then wait for the answer. 

