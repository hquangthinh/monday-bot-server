import os
os.environ["OPENAI_API_KEY"] = 'xxx'

from llama_index import GPTSimpleVectorIndex, SimpleDirectoryReader

documents = SimpleDirectoryReader('data').load_data()
index = GPTSimpleVectorIndex.from_documents(documents)

index.save_to_disk('test_index.json')

print("Build index completed!")
