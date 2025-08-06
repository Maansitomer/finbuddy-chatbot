
def retrieve_docs(vectorstore, query, k=3):
    return vectorstore.similarity_search(query, k=k)
