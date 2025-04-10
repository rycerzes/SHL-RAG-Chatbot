import os
from io import BytesIO, StringIO
from typing import Tuple, List
import time
import re
import pandas as pd
import tiktoken

from llama_index.core.schema import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.astra_db import AstraDBVectorStore
from llama_index.embeddings.nvidia import NVIDIAEmbedding
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.llms.groq import Groq
from llama_index.postprocessor.nvidia_rerank import NVIDIARerank
from dotenv import load_dotenv

load_dotenv()

COLLECTION_NAME = os.getenv("ASTRA_DB_COLLECTION", "shl_solutions")
EMBEDDING_DIMENSION = 1024
MAX_TOKENS = 2000
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Reranking configuration
RERANKER_MODEL = "nvidia/llama-3_2-nv-rerankqa-1b-v2"  # NVIDIA reranker model
RERANK_TOP_N = 10
USE_RERANKING = False  # Enable/disable reranking

# Max token limit for NVIDIA embedding model
# https://forums.developer.nvidia.com/t/discrepancy-in-maximum-token-length-for-nv-embed-qa-1b-v2-model/322768
NVIDIA_MODEL_MAX_TOKENS = 512
RAG_CHUNK_SIZE = 400
RAG_CHUNK_OVERLAP = 50

ENCODING = tiktoken.get_encoding("cl100k_base")


def count_tokens(text):
    """Count tokens in text using tiktoken or estimate based on characters"""
    if ENCODING:
        return len(ENCODING.encode(text))
    else:
        # Approximate token count (4 chars ≈ 1 token)
        return len(text) // 4


def parse_csv(file: BytesIO, filename: str) -> Tuple[List[dict], str]:
    """Parse CSV file containing SHL solutions data

    The CSV has headers: Name,Link,Type,Remote Testing,Adaptive/IRT,Test Type,Description,Languages,Job_levels,Completion_time
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
        if "Description" in df.columns:
            item["Description"] = row["Description"]
        if "Languages" in df.columns:
            item["Languages"] = row["Languages"]
        if "Job_levels" in df.columns:
            item["Job_levels"] = row["Job_levels"]
        if "Completion_time" in df.columns:
            item["Completion_time"] = row["Completion_time"]

        text_content = (
            f"Name: {item['Name']}\n"
            f"Link: {item['Link']}\n"
            f"Type: {item['Type']}\n"
            f"Remote Testing: {item['Remote Testing']}\n"
            f"Adaptive/IRT: {item['Adaptive/IRT']}\n"
            f"Test Type: {item['Test Type']}"
        )

        if "Description" in item:
            text_content += f"\nDescription: {item['Description']}"
        if "Languages" in item:
            text_content += f"\nLanguages: {item['Languages']}"
        if "Job_levels" in item:
            text_content += f"\nJob levels: {item['Job_levels']}"
        if "Completion_time" in item:
            text_content += f"\nCompletion time: {item['Completion_time']}"

        item["text_content"] = text_content
        csv_data.append(item)

    return csv_data, filename


def text_to_docs(items: List[dict], filename: str) -> List[Document]:
    """Convert parsed items to Document objects for indexing"""
    doc_chunks = []

    for i, item in enumerate(items):
        text = item["text_content"]

        # Use sentence splitter with RAG-optimized chunk parameters
        splitter = SentenceSplitter(
            chunk_size=RAG_CHUNK_SIZE,
            chunk_overlap=RAG_CHUNK_OVERLAP,
        )

        # Split into semantic chunks for better RAG performance
        chunks = splitter.split_text(text)

        for j, chunk_text in enumerate(chunks):
            # Count tokens accurately using tiktoken if available
            chunk_tokens = count_tokens(chunk_text)

            if chunk_tokens > NVIDIA_MODEL_MAX_TOKENS:
                # Further split if chunk exceeds token limit
                sub_chunks = []
                current_text = ""
                current_tokens = 0

                # Split by sentences for more semantic chunks
                sentences = re.split(r"(?<=[.!?])\s+", chunk_text)

                for sentence in sentences:
                    # Count tokens in this sentence
                    sentence_tokens = count_tokens(sentence)

                    # If adding this sentence would exceed the limit, create a new chunk
                    if current_tokens + sentence_tokens > NVIDIA_MODEL_MAX_TOKENS:
                        if current_text:
                            sub_chunks.append(current_text.strip())
                        current_text = sentence
                        current_tokens = sentence_tokens
                    else:
                        if current_text:
                            current_text += " "
                        current_text += sentence
                        current_tokens += sentence_tokens

                # Add the last sub-chunk if it exists
                if current_text:
                    sub_chunks.append(current_text.strip())

                # Handle sentences that are individually too long (rare case)
                final_sub_chunks = []
                for sub_chunk in sub_chunks:
                    sub_chunk_tokens = count_tokens(sub_chunk)

                    if sub_chunk_tokens > NVIDIA_MODEL_MAX_TOKENS:
                        # Hard truncate if a single sentence is too long
                        if ENCODING:
                            # Use tiktoken to truncate precisely at token boundaries
                            tokens = ENCODING.encode(sub_chunk)
                            truncated_tokens = tokens[
                                : NVIDIA_MODEL_MAX_TOKENS - 1
                            ]  # Leave room for potential continuation marker
                            truncated_text = ENCODING.decode(truncated_tokens) + "..."
                            final_sub_chunks.append(truncated_text)
                        else:
                            # Character-based truncation as fallback
                            char_limit = NVIDIA_MODEL_MAX_TOKENS * 4
                            final_sub_chunks.append(sub_chunk[: char_limit - 3] + "...")
                    else:
                        final_sub_chunks.append(sub_chunk)

                processed_chunks = final_sub_chunks
            else:
                processed_chunks = [chunk_text]

            for k, processed_chunk in enumerate(processed_chunks):
                # Final verification that chunk is within limits
                if (
                    ENCODING
                    and len(ENCODING.encode(processed_chunk)) > NVIDIA_MODEL_MAX_TOKENS
                ):
                    tokens = ENCODING.encode(processed_chunk)
                    processed_chunk = (
                        ENCODING.decode(tokens[: NVIDIA_MODEL_MAX_TOKENS - 1]) + "..."
                    )

                metadata = {
                    "chunk": j,
                    "sub_chunk": k if len(processed_chunks) > 1 else None,
                    "source": f"{filename}:item-{i}",
                    "filename": filename,
                    "item_index": i,
                    "name": item.get("Name", ""),
                    "link": item.get("Link", ""),
                    "type": item.get("Type", ""),
                    "remote_testing": item.get("Remote Testing", ""),
                    "adaptive_irt": item.get("Adaptive/IRT", ""),
                    "test_type": item.get("Test Type", ""),
                }

                if "Description" in item:
                    metadata["description"] = item["Description"]
                if "Languages" in item:
                    metadata["languages"] = item["Languages"]
                if "Job_levels" in item:
                    metadata["job_levels"] = item["Job_levels"]
                if "Completion_time" in item:
                    metadata["completion_time"] = item["Completion_time"]

                doc = Document(
                    text=processed_chunk,
                    metadata=metadata,
                )
                doc_chunks.append(doc)

    return doc_chunks


def connect_to_astra():
    """Connect to AstraDB using environment variables"""
    token = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
    api_endpoint = os.getenv("ASTRA_DB_API_ENDPOINT")

    # Add NVIDIA API configuration
    nvidia_api_key = os.getenv("NVIDIA_API_KEY")
    nvidia_base_url = "https://integrate.api.nvidia.com/v1"

    if not nvidia_api_key:
        raise ValueError("NVIDIA_API_KEY environment variable is not set")

    embed_model = NVIDIAEmbedding(
        model_name="nvidia/llama-3_2-nv-embedqa-1b-v2",
        api_key=nvidia_api_key,
        api_base=nvidia_base_url,
        max_tokens=NVIDIA_MODEL_MAX_TOKENS,  # Set to 512 token limit
        embed_batch_size=1,  # Process one input at a time
    )

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

        # Final verification and hard truncation if needed
        for doc in doc_chunks:
            # Ensure text is within token limit
            token_count = count_tokens(doc.text)
            if token_count > NVIDIA_MODEL_MAX_TOKENS:
                if ENCODING:
                    tokens = ENCODING.encode(doc.text)
                    doc.text = (
                        ENCODING.decode(tokens[: NVIDIA_MODEL_MAX_TOKENS - 3]) + "..."
                    )
                else:
                    # Fallback character-based truncation
                    doc.text = doc.text[: int(NVIDIA_MODEL_MAX_TOKENS * 3.5)] + "..."

        documents.extend(doc_chunks)
        print(f"  - Generated {len(doc_chunks)} chunks from {filename}")

    # Process documents in smaller batches to avoid batch size issues
    batch_size = 10
    all_documents = []

    for i in range(0, len(documents), batch_size):
        batch = documents[i : i + batch_size]
        print(
            f"Processing batch {i // batch_size + 1}/{(len(documents) - 1) // batch_size + 1}"
        )

        try:
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            batch_index = VectorStoreIndex.from_documents(
                batch, storage_context=storage_context, embed_model=embed_model
            )
            all_documents.extend(batch)
        except Exception as e:
            print(f"Error in batch {i // batch_size + 1}: {e}")
            # Process documents one by one to isolate problematic documents
            for j, doc in enumerate(batch):
                try:
                    single_doc_batch = [doc]
                    storage_context = StorageContext.from_defaults(
                        vector_store=vector_store
                    )
                    single_index = VectorStoreIndex.from_documents(
                        single_doc_batch,
                        storage_context=storage_context,
                        embed_model=embed_model,
                    )
                    all_documents.append(doc)
                    print(f"  - Successfully processed document {i + j + 1}")
                except Exception as e2:
                    token_count = count_tokens(doc.text)
                    print(
                        f"  - Failed to process document {i + j + 1} (token count: {token_count}): {e2}"
                    )
                    # Super aggressive truncation as last resort
                    try:
                        if token_count > NVIDIA_MODEL_MAX_TOKENS:
                            if ENCODING:
                                tokens = ENCODING.encode(doc.text)
                                doc.text = (
                                    ENCODING.decode(
                                        tokens[: NVIDIA_MODEL_MAX_TOKENS // 2]
                                    )
                                    + "..."
                                )
                            else:
                                doc.text = (
                                    doc.text[: NVIDIA_MODEL_MAX_TOKENS * 2] + "..."
                                )

                            storage_context = StorageContext.from_defaults(
                                vector_store=vector_store
                            )
                            single_index = VectorStoreIndex.from_documents(
                                [doc],
                                storage_context=storage_context,
                                embed_model=embed_model,
                            )
                            all_documents.append(doc)
                            print(
                                f"  - Salvaged document {i + j + 1} after aggressive truncation"
                            )
                    except Exception as e3:
                        print(
                            f"  - Could not salvage document {i + j + 1} even after truncation: {e3}"
                        )

    print(
        f"Successfully indexed {len(all_documents)} out of {len(documents)} documents in AstraDB"
    )

    # Return latest index for consistency with original function
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    return VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)


def search_astra(query, limit=5):
    """Search AstraDB for documents matching the query"""
    vector_store, embed_model = connect_to_astra()

    # Create Groq LLM instance if API key is available
    llm = None
    if GROQ_API_KEY:
        llm = Groq(model="llama-3.3-70b-versatile", api_key=GROQ_API_KEY)

    StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)

    # Apply reranking if enabled
    if USE_RERANKING:
        nvidia_api_key = os.getenv("NVIDIA_API_KEY")
        nvidia_base_url = "https://integrate.api.nvidia.com/v1"

        if not nvidia_api_key:
            raise ValueError("NVIDIA_API_KEY environment variable is not set")

        reranker = NVIDIARerank(
            model_name=RERANKER_MODEL,
            api_key=nvidia_api_key,
            api_base=nvidia_base_url,
            top_n=limit,  # Keep only top 'limit' results after reranking
        )

        # Create retriever with higher top_k for reranking
        retriever = index.as_retriever(similarity_top_k=RERANK_TOP_N)

        # Use the retriever with node postprocessors
        query_engine = retriever.as_query_engine(
            node_postprocessors=[reranker], llm=llm
        )
    else:
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
            response = search_astra(query, limit)

            # Handle different response types correctly
            if hasattr(response, "response"):
                # This is a response from the LLM
                print("\n=== LLM Response ===")
                print(response.response)

                if hasattr(response, "source_nodes") and response.source_nodes:
                    print("\n=== Sources ===")
                    for i, source_node in enumerate(response.source_nodes):
                        metadata = source_node.metadata
                        name = metadata.get("name", "Unknown")
                        link = metadata.get("link", "No link available")
                        source = metadata.get("source", "Unknown source")

                        print(f"\n--- Source {i + 1} ---")
                        print(f"Name: {name}")
                        print(f"Link: {link}")
                        print(f"Source: {source}")
            else:
                # This is just a list of results without LLM processing
                print(f"Found {len(response)} results:")

                for i, result in enumerate(response):
                    metadata = result.metadata
                    name = metadata.get("name", "Unknown")
                    link = metadata.get("link", "No link available")
                    source = metadata.get("source", "Unknown source")

                    print(f"\n--- Result {i + 1} ---")
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
            import traceback

            traceback.print_exc()

    else:
        print(f"Unknown command: {command}")
        print("Available commands: index, search")
        sys.exit(1)
