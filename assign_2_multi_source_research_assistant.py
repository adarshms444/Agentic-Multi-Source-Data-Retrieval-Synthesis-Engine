# Cell 1: Install Dependencies
!pip install -qU langchain langchain-community langchain-openai langchain-huggingface gradio chromadb rank_bm25 pypdf unstructured markdown sentence-transformers pandas

!pip install langchain-classic

import os
from langchain_openai import ChatOpenAI

os.environ["OPENROUTER_API_KEY"] = ""

MODEL_NAME = "nvidia/nemotron-3-nano-30b-a3b:free"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

def get_llm():
    return ChatOpenAI(
        model=MODEL_NAME,
        openai_api_key=os.environ["OPENROUTER_API_KEY"],
        openai_api_base=OPENROUTER_BASE_URL,
        temperature=0.1,
        max_tokens=1024
    )

print("LLM Client Ready!")

from langchain_community.document_loaders import PyPDFLoader, UnstructuredMarkdownLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

def process_documents(file_paths):
    documents = []

    for path in file_paths:
        ext = os.path.splitext(path)[1].lower()
        if ext == '.pdf':
            loader = PyPDFLoader(path)
        elif ext == '.md':
            loader = UnstructuredMarkdownLoader(path)
        elif ext == '.csv':
            loader = CSVLoader(path)
        else:
            print(f"Skipping unsupported file: {path}")
            continue

        docs = loader.load()
        for doc in docs:
            doc.metadata["source_file"] = os.path.basename(path)
        documents.extend(docs)

    # Chunking
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = text_splitter.split_documents(documents)

    # Embeddings
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # 1. Sparse Retriever
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 6

    # 2. Dense Retriever
    vectorstore = Chroma.from_documents(chunks, embeddings)
    dense_retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

    # 3. Hybrid Retriever
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, dense_retriever], weights=[0.5, 0.5]
    )

    print(f"Knowledge base built with {len(chunks)} chunks.")
    return ensemble_retriever

import json
import re

def execute_tool(query, retriever, system_prompt, format_instructions, k_value=6):
    """Generic execution block with dynamic formatting, averaging, and Dynamic K Retrieval."""


    bm25_retriever = retriever.retrievers[0]
    dense_retriever = retriever.retrievers[1]

    bm25_retriever.k = k_value
    dense_retriever.search_kwargs["k"] = k_value

    docs = retriever.invoke(query)
    retrieved_sources = list(set([d.metadata.get('source_file', 'Unknown') for d in docs]))
    context = "\n\n".join([f"Source: {d.metadata.get('source_file', 'Unknown')}\nContent: {d.page_content}" for d in docs])

    try:
        if not docs:
            raise ValueError("No documents retrieved.")

        tokenized_query = query.lower().split()
        bm25_model = bm25_retriever.vectorizer
        raw_bm25_scores = bm25_model.get_scores(tokenized_query)

        raw_dense_results = dense_retriever.vectorstore.similarity_search_with_score(query, k=15)

        total_hybrid_score = 0.0

        for chunk in docs:

            try:
                chunk_index = next(i for i, d in enumerate(bm25_retriever.docs) if d.page_content == chunk.page_content)
                chunk_bm25 = raw_bm25_scores[chunk_index]
            except StopIteration:
                chunk_bm25 = 0.0

            norm_keyword = min(1.0, chunk_bm25 / 15.0)

            chunk_distance = None
            for d, dist in raw_dense_results:
                if d.page_content == chunk.page_content:
                    chunk_distance = dist
                    break

            if chunk_distance is not None:
                norm_semantic = 1 / (1 + chunk_distance)
            else:
                norm_semantic = 0.0

            chunk_final_score = (0.5 * norm_semantic) + (0.5 * norm_keyword)
            total_hybrid_score += chunk_final_score

        # --- D. Average Context Confidence ---
        average_math_score = total_hybrid_score / len(docs)
        math_percentage = round(average_math_score * 100, 2)

        if math_percentage >= 65:
            ui_confidence = "High"
        elif math_percentage >= 35:
            ui_confidence = "Medium"
        else:
            ui_confidence = "Low"

    except Exception as e:
        ui_confidence = "Unknown"

    prompt = f"""{system_prompt}

    CRITICAL INSTRUCTION: You are strictly bound to the provided context. Answer using ONLY the information provided below.

    FORMATTING INSTRUCTIONS:
    {format_instructions}

    Context:
    {context}

    Question: {query}

    Answer:"""

    llm = get_llm()
    response = llm.invoke(prompt).content.strip()

    return {
        "answer": response,
        "confidence_score": ui_confidence,
        "citations": retrieved_sources
    }

def handle_factual(query, retriever):
    system_prompt = "You are a factual AI assistant. Answer the user's question directly and accurately using ONLY the provided context."
    format_instructions = "Provide a clear, direct, and concise answer. Do NOT use headers like 'Executive Summary' or 'Detailed Analysis'. Just state the facts using bullet points if necessary."
    return execute_tool(query, retriever, system_prompt, format_instructions, k_value=6)

def handle_comparative(query, retriever):
    system_prompt = "You are an analytical assistant. Compare, contrast, and correlate data across the provided context sources to answer the question."
    format_instructions = "Provide a structured comparison. Use Markdown tables, bold text, or side-by-side bullet points to clearly highlight differences and correlations between the data."

    return execute_tool(query, retriever, system_prompt, format_instructions, k_value=12)

def handle_summary(query, retriever):
    system_prompt = "You are a summarization assistant. Provide a comprehensive, structured summary of the requested topics based heavily on the context."
    format_instructions = """Always structure your answer exactly like this:
### Executive Summary
(Provide a direct, 1-2 sentence overview of the documents)

### Detailed Analysis
(Provide a deep dive using clear bullet points and bold text for emphasis.)"""
    return execute_tool(query, retriever, system_prompt, format_instructions, k_value=6)

def handle_clarification(query):
    return {
        "answer": f"Your query '{query}' is a bit ambiguous. Could you clarify if you are looking for specific facts, a summary of a document, or a comparison between data points?",
        "confidence_score": "N/A",
        "citations": []
    }

def classify_intent(query):
    """
    Acts as a semantic router, analyzing query intent to trigger the appropriate analytical tool.
    Utilizes highly structured zero-shot prompting to minimize token usage while enforcing deterministic output.
    """

    prompt = f"""You are a specialized routing agent within an enterprise Retrieval-Augmented Generation (RAG) pipeline.
Your sole function is to analyze the semantic intent of the user's query and map it to the correct downstream analytical tool.

# ROUTING CATEGORIES AND TRIGGERS:
1. FACTUAL:
   - Triggers: Questions asking for specific data points, definitions, mechanisms, or targeted information extraction (e.g., "What", "How", "Explain the metrics").
2. COMPARATIVE:
   - Triggers: Queries containing verbs like "compare", "contrast", "relate", "align", or explicit requests to evaluate differences between multiple entities, metrics, or documents.
3. SUMMARY:
   - Triggers: Requests for high-level overviews, document syntheses, executive summaries, or broad conceptual explanations of an entire file or architecture.
4. AMBIGUOUS:
   - Triggers: Queries lacking sufficient context, explicit subjects, or conversational filler that cannot be routed accurately.

# STRICT SYSTEM CONSTRAINTS:
- Evaluate the query's linguistic structure and explicit verbs against the triggers above.
- Output strictly the exact category name in uppercase (e.g., FACTUAL).
- Do NOT output any conversational text, reasoning, prefixes, or punctuation.

# INPUT TARGET:
Query: "{query}"
Category:"""

    llm = get_llm()
    response = llm.invoke(prompt).content.strip().upper()

    valid_intents = ["FACTUAL", "COMPARATIVE", "SUMMARY", "AMBIGUOUS"]
    for intent in valid_intents:
        if intent in response:
            return intent

    print(f"[Warning] Semantic Router Fallback. Unrecognized LLM output: {response}")
    return "FACTUAL"

def dispatch_query(query, retriever):
    intent = classify_intent(query)
    print(f"[System] Semantic Router mapped intent to: {intent}")

    if intent == "FACTUAL":
        result = handle_factual(query, retriever)
        tool_used = "Factual Q&A Tool (Hybrid Search)"
    elif intent == "COMPARATIVE":
        result = handle_comparative(query, retriever)
        tool_used = "Comparative/Analytical Tool (Hybrid Search)"
    elif intent == "SUMMARY":
        result = handle_summary(query, retriever)
        tool_used = "Summary Tool (Hybrid Search)"
    else:
        result = handle_clarification(query)
        tool_used = "Clarification Agent"

    return result, tool_used

import os
from google.colab import files

print("Please upload your files (PDF, CSV, Markdown)...")
uploaded = files.upload()

if not uploaded:
    print("No files uploaded. Please run the cell again to upload files.")
else:
    file_paths = list(uploaded.keys())
    print(f"\nUploaded files: {file_paths}")

    print("\nProcessing documents and building the ChromaDB Vector Store...")
    try:
        global_retriever = process_documents(file_paths)
        print("✅ Knowledge base successfully created!\n")

        print("="*60)
        print("🧠 Multi-Source RAG Assistant is Ready!")
        print("Type 'exit' or 'quit' to stop the chat.")
        print("="*60)

        while True:
            query = input("\nAsk a question: ")

            if query.lower() in ['exit', 'quit']:
                print("Exiting test mode.")
                break
            if not query.strip():
                continue

            print(f"\n[Thinking...] Routing query: '{query}'")
            try:
                result, tool_used = dispatch_query(query, global_retriever)


                answer = result.get("answer", "No answer generated.")

                raw_citations = result.get("citations", [])
                if isinstance(raw_citations, list):
                    citations = ", ".join(list(set(raw_citations)))
                else:
                    citations = str(raw_citations)

                confidence = result.get("confidence_score", "N/A")

                print("\n--- RESULTS ---")
                print(f"🛠️ Tool Triggered  : {tool_used}")
                print(f"📊 Confidence Score: {confidence}")
                print(f"📑 Citations       : {citations}")
                print(f"\n🤖 Answer:\n{answer}")
                print("-" * 60)

            except Exception as e:
                print(f"❌ Error processing query: {str(e)}")

    except Exception as e:
        print(f"❌ Error building knowledge base: {str(e)}")

import os
import re
import gradio as gr

def ui_process_documents(files, current_retriever_state):
    """Processes documents and updates the specific user's state."""
    if not files:
        return "⚠️ Please upload at least one file before processing.", current_retriever_state

    file_paths = [f.name for f in files]
    try:
        new_retriever = process_documents(file_paths)
        return "✅ Knowledge base successfully built! The Multi-Source RAG is ready for queries.", new_retriever
    except Exception as e:
        return f"❌ Error building knowledge base: {str(e)}", current_retriever_state

def clean_citations(citations_list):
    if not isinstance(citations_list, list):
        return str(citations_list)
    unique_cleansed_files = []
    for citation in citations_list:
        clean_name = re.sub(r'\s*\(\d+\)', '', citation)
        if clean_name not in unique_cleansed_files:
            unique_cleansed_files.append(clean_name)
    return ", ".join(unique_cleansed_files)

def ui_answer_query(query, current_retriever_state):
    """Answers queries using the specific user's isolated state."""
    if current_retriever_state is None:
        return "⚠️ **Error:** Please upload and process documents first.", "", "", ""

    if not query.strip():
        return "⚠️ **Error:** Please enter a question.", "", "", ""

    try:
        result, tool_used = dispatch_query(query, current_retriever_state)

        raw_answer = result.get("answer", "No answer generated.")
        formatted_answer = f"{raw_answer}"
        raw_citations = result.get("citations", [])
        citations = clean_citations(raw_citations)
        confidence = result.get("confidence_score", "N/A")

        return formatted_answer, tool_used, confidence, citations
    except Exception as e:
        return f"❌ **Error processing query:** {str(e)}", "Error", "Error", "Error"

# --- ENTERPRISE UI STYLING ---
custom_css = """
body { background-color: var(--background-fill-primary); }
.header-box { text-align: center; padding: 20px; margin-bottom: 20px; border-radius: 12px; background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%); color: white; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
.header-box h1 { color: white !important; margin-bottom: 5px; font-weight: 700; }
.header-box p { color: #e2e8f0 !important; font-size: 1.1em; }
.card { background: var(--background-fill-secondary); padding: 25px; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.05); border: 1px solid var(--border-color-primary); }
.answer-box {
    background: var(--background-fill-primary);
    padding: 25px;
    border-radius: 8px;
    border-left: 5px solid #3b82f6;
    font-size: 1.05em;
    line-height: 1.6;
    color: var(--body-text-color) !important;
}
"""

enterprise_theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="slate",
    neutral_hue="gray",
    font=[gr.themes.GoogleFont("Inter"), "sans-serif"]
)

with gr.Blocks(css=custom_css, title="Agentic RAG System", theme=enterprise_theme) as demo:

    user_retriever = gr.State(None)

    gr.HTML(
        """
        <div class="header-box">
            <h1>🧠 Nexus RAG Intelligence</h1>
            <p>Agentic Multi-Source Data Retrieval & Synthesis Engine</p>
        </div>
        """
    )

    with gr.Row():
        # LEFT COLUMN
        with gr.Column(scale=1):
            with gr.Group(elem_classes="card"):
                gr.Markdown("### 📂 1. System Ingestion")
                gr.Markdown("Upload standard operating procedures, CSV metrics, and research PDFs here.")
                file_upload = gr.File(
                    label="Drop Files Here",
                    file_count="multiple",
                    file_types=[".pdf", ".csv", ".md"]
                )
                process_btn = gr.Button("⚙️ Initialize VectorDB", variant="primary", size="lg")
                status_box = gr.Textbox(label="System Status", show_label=True, interactive=False)

        # RIGHT COLUMN
        with gr.Column(scale=2):
            with gr.Group(elem_classes="card"):
                gr.Markdown("### 💬 2. Agentic Query")

                with gr.Row(equal_height=True):
                    query_input = gr.Textbox(
                        label="Ask the AI",
                        placeholder="Type your query here...",
                        lines=2,
                        scale=4,
                        show_label=False
                    )
                    submit_btn = gr.Button("🔍 Search Docs", variant="primary", scale=1)

            gr.Markdown("### 🤖 Synthesized Analysis")
            answer_output = gr.Markdown(
                value="*The agent's synthesized research will appear here...*",
                elem_classes="answer-box"
            )

            with gr.Accordion("⚙️ View Agent Routing Diagnostics", open=True):
                with gr.Row():
                    tool_output = gr.Textbox(label="🛠️ Dispatcher Routing", interactive=False)
                    conf_output = gr.Textbox(label="📊 Hybrid Math Confidence", interactive=False)
                    cite_output = gr.Textbox(label="📑 Retrieved Sources", interactive=False)

    # --- EVENT LISTENERS WITH STATE INJECTION ---
    process_btn.click(
        fn=ui_process_documents,
        inputs=[file_upload, user_retriever],
        outputs=[status_box, user_retriever]
    )

    submit_btn.click(
        fn=ui_answer_query,
        inputs=[query_input, user_retriever],
        outputs=[answer_output, tool_output, conf_output, cite_output]
    )

    query_input.submit(
        fn=ui_answer_query,
        inputs=[query_input, user_retriever],
        outputs=[answer_output, tool_output, conf_output, cite_output]
    )

if __name__ == "__main__":
    demo.launch(debug=True, share=True)
