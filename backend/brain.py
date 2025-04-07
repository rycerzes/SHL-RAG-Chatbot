import os
from io import BytesIO, StringIO
from typing import Tuple, List
import time
import re
import pandas as pd

from llama_index.core.schema import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.astra_db import AstraDBVectorStore
from llama_index.embeddings.nvidia import NVIDIAEmbedding
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.llms.groq import Groq

from dotenv import load_dotenv

load_dotenv()

COLLECTION_NAME = os.getenv("ASTRA_DB_COLLECTION", "shl_solutions")
EMBEDDING_DIMENSION = 1024
MAX_VECTORIZE_LENGTH = 1000  # 1000 to stay under token limits
MAX_TOKENS = 475
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


def parse_csv(file: BytesIO, filename: str) -> Tuple[List[dict], str]:
    """Parse CSV file containing SHL solutions data

    The CSV has these headers: Name,Link,Type,Remote Testing,Adaptive/IRT,Test Type
    """
    content = file.read().decode("utf-8")
    csv_data = []

    df = pd.read_csv(StringIO(content))

    for _, row in df.iterrows():
        item = {
            "Name": row["Name"],
            "Link": row["Link"],
            "Type": row["Type"],
            "Remote Testing": row["Remote Testing"],
            "Adaptive/IRT": row["Adaptive/IRT"],
            "Test Type": row["Test Type"],
        }
        text_content = (
            f"Name: {item['Name']}\n"
            f"Link: {item['Link']}\n"
            f"Type: {item['Type']}\n"
            f"Remote Testing: {item['Remote Testing']}\n"
            f"Adaptive/IRT: {item['Adaptive/IRT']}\n"
            f"Test Type: {item['Test Type']}"
        )
        item["text_content"] = text_content
        csv_data.append(item)

    return csv_data, filename


def text_to_docs(items: List[dict], filename: str) -> List[Document]:
    """Convert parsed items to Document objects for indexing"""
    doc_chunks = []

    for i, item in enumerate(items):
        text = item["text_content"]

        splitter = SentenceSplitter(
            chunk_size=2000,
            chunk_overlap=200,
        )

        nodes = splitter.get_nodes_from_documents([Document(text=text)])

        for j, node in enumerate(nodes):
            chunk_text = node.text

            doc = Document(
                text=chunk_text,
                metadata={
                    "chunk": j,
                    "source": f"{filename}:item-{i}",
                    "filename": filename,
                    "item_index": i,
                    "name": item.get("Name", ""),
                    "link": item.get("Link", ""),
                    "type": item.get("Type", ""),
                    "remote_testing": item.get("Remote Testing", ""),
                    "adaptive_irt": item.get("Adaptive/IRT", ""),
                    "test_type": item.get("Test Type", ""),
                },
            )
            doc_chunks.append(doc)

    return doc_chunks


def connect_to_astra():
    """Connect to AstraDB using environment variables"""
    token = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
    api_endpoint = os.getenv("ASTRA_DB_API_ENDPOINT")

    if not token:
        raise ValueError("ASTRA_DB_APPLICATION_TOKEN environment variable is not set")

    if not api_endpoint:
        raise ValueError("ASTRA_DB_API_ENDPOINT environment variable is not set")

    embed_model = NVIDIAEmbedding(model_name="nvidia/embed-qa-4")

    astra_db_store = AstraDBVectorStore(
        token=token,
        api_endpoint=api_endpoint,
        collection_name=COLLECTION_NAME,
        embedding_dimension=EMBEDDING_DIMENSION,
    )

    print(f"Connected to Astra DB: {COLLECTION_NAME}")
    return astra_db_store, embed_model


def truncate_to_token_limit(text: str, max_tokens: int = MAX_TOKENS) -> str:
    """Truncate text to approximately stay under token limit.

    This is a simple estimation since we don't have access to the actual tokenizer.
    On average, 1 token is roughly 4 characters in English text.
    """
    words = re.findall(r"\b\w+\b|[^\w\s]|\s+", text)

    result = ""
    estimated_token_count = 0

    for word in words:
        word_token_estimate = max(1, round(len(word) / 4))

        if estimated_token_count + word_token_estimate > max_tokens:
            break

        result += word
        estimated_token_count += word_token_estimate

    return result


def store_documents(vector_store, embed_model, documents: List[Document]) -> None:
    """Store documents in AstraDB collection using LlamaIndex"""
    print(f"Preparing to store {len(documents)} documents in AstraDB...")

    try:
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        index = VectorStoreIndex.from_documents(
            documents, storage_context=storage_context, embed_model=embed_model
        )

        print(f"Successfully indexed {len(documents)} documents in AstraDB")
        return index
    except Exception as e:
        print(f"Error storing documents: {e}")
        raise


def get_index_for_csv(csv_files, csv_names):
    """Create or update index for CSV files using AstraDB"""
    print(f"Creating index for {len(csv_files)} CSV files...")

    vector_store, embed_model = connect_to_astra()

    documents = []
    for i, (csv_file, csv_name) in enumerate(zip(csv_files, csv_names)):
        print(f"Processing CSV file {i + 1}/{len(csv_files)}: {csv_name}")
        items, filename = parse_csv(BytesIO(csv_file), csv_name)
        doc_chunks = text_to_docs(items, filename)
        documents.extend(doc_chunks)
        print(f"  - Generated {len(doc_chunks)} chunks from {filename}")

    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(
        documents, storage_context=storage_context, embed_model=embed_model
    )

    print(f"Successfully indexed {len(documents)} documents in AstraDB")
    return index


def search_astra(query, limit=5):
    """Search AstraDB for documents matching the query"""
    vector_store, embed_model = connect_to_astra()

    # Create Groq LLM instance if API key is available
    llm = None
    if GROQ_API_KEY:
        llm = Groq(model="llama-3.3-70b-versatile", api_key=GROQ_API_KEY)

    StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)

    # Pass the LLM to the query engine or use None to disable LLM
    query_engine = index.as_query_engine(similarity_top_k=limit, llm=llm)

    response = query_engine.query(query)

    return response


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python brain.py [index|search]")
        sys.exit(1)

    command = sys.argv[1]

    if command == "index":
        if len(sys.argv) < 3:
            print("Usage: python brain.py index <file_or_directory> [--dry-run]")
            sys.exit(1)

        import glob

        file_path = sys.argv[2]
        dry_run = "--dry-run" in sys.argv

        start_time = time.time()
        csv_files = []
        csv_names = []
        skipped_files = []

        file_count = 0

        if file_path.endswith(".csv"):
            patterns = [file_path]
        else:
            patterns = [f"{file_path}/**/*.csv", f"{file_path}/*.csv"]

        for pattern in patterns:
            print(f"Searching for files matching: {pattern}")

            for matched_path in glob.glob(pattern, recursive=True):
                file_count += 1
                try:
                    with open(matched_path, "rb") as f:
                        csv_files.append(f.read())
                        csv_names.append(
                            matched_path
                        )  # Store full path for better identification
                except Exception as e:
                    skipped_files.append((matched_path, str(e)))

        print(f"Found {file_count} files, loaded {len(csv_files)} successfully")

        if skipped_files:
            print(f"Warning: Skipped {len(skipped_files)} files due to errors:")
            for file_path, error in skipped_files[:5]:  # Show first 5 errors
                print(f"  - {file_path}: {error}")
            if len(skipped_files) > 5:
                print(f"  ... and {len(skipped_files) - 5} more")

        if dry_run:
            print("Dry run completed. No documents were indexed.")
        else:
            get_index_for_csv(csv_files, csv_names)

        elapsed_time = time.time() - start_time
        print(f"Total processing time: {elapsed_time:.2f} seconds")

    elif command == "search":
        if len(sys.argv) < 3:
            print("Usage: python brain.py search <query> [--limit <number>]")
            sys.exit(1)

        query = sys.argv[2]
        limit = 5  # default limit

        if "--limit" in sys.argv and sys.argv.index("--limit") + 1 < len(sys.argv):
            try:
                limit = int(sys.argv[sys.argv.index("--limit") + 1])
            except ValueError:
                print("Error: limit must be a number")
                sys.exit(1)

        print(f"Searching for: '{query}' (limit: {limit})")

        try:
            results = search_astra(query, limit)

            if not results:
                print("No results found.")
            else:
                print(f"Found {len(results)} results:")

                for i, result in enumerate(results):
                    similarity = result.similarity * 100
                    metadata = result.metadata
                    name = metadata.get("name", "Unknown")
                    link = metadata.get("link", "No link available")
                    source = metadata.get("source", "Unknown source")

                    print(f"\n--- Result {i + 1} (Similarity: {similarity:.2f}%) ---")
                    print(f"Name: {name}")
                    print(f"Link: {link}")
                    print(f"Source: {source}")

                    content = result.text
                    content_preview = "\n".join(content.split("\n")[:3])
                    if content_preview != content:
                        content_preview += "\n..."
                    print(f"Content Preview: {content_preview}")

        except Exception as e:
            print(f"Error during search: {e}")

    else:
        print(f"Unknown command: {command}")
        print("Available commands: index, search")
        sys.exit(1)
