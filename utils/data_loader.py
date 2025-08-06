import os
from langchain.schema import Document

def load_text_files_from_folder(folder_path):
    """Load all .txt files from a folder and return them as Document objects."""
    docs = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            with open(os.path.join(folder_path, filename), "r", encoding="utf-8") as f:
                content = f.read()
                docs.append(Document(
                    page_content=content,
                    metadata={"source": filename}
                ))
    return docs

def load_all_datasets(internal_folder, external_folder):
    """Load internal and external datasets as Document objects."""
    return load_text_files_from_folder(internal_folder) + load_text_files_from_folder(external_folder)
