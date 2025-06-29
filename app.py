import streamlit as st
import os
import pandas as pd
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

# Configure page
st.set_page_config(page_title="Arnold Schönberg Letter Chatbot", page_icon="📜", layout="wide")

# Title
st.title("📜 Arnold Schönberg Letter Chatbot")
st.write("This Chatbot allows you to ask questions about the Correspondence between Arnold Schönberg and his publishers Universal-Edition and Verlag Dreililien. A digital edition of these letters is availabel at www.schoenberg-ua.at. The chatbot is based on a file that contains letter IDs and letter text, metadata is no included. In addition to natural language interaction, the bot provides IDs and quotes from up to three relevant letters. Most of the letters are written in German.")
st.write("Ask questions about the letters")

# Sidebar for configuration
with st.sidebar:
    st.header("Setup")
    openai_api_key = st.text_input("OpenAI API Key", type="password")
    
    # Model selection
    st.subheader("Model Selection")
    model_options = {
        "GPT-4o Mini": "gpt-4o-mini", 
        "GPT-4o": "gpt-4o",
        "GPT-3.5 Turbo": "gpt-3.5-turbo"
    }
    selected_model = st.selectbox("Choose OpenAI Model:", list(model_options.keys()))
    
    if openai_api_key:
        os.environ["OPENAI_API_KEY"] = openai_api_key
        st.success("API Key set!")
    else:
        st.warning("Please enter your OpenAI API Key")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "index" not in st.session_state:
    st.session_state.index = None

if "csv_data" not in st.session_state:
    st.session_state.csv_data = None

# Load and index CSV
@st.cache_resource
def load_csv_and_create_index(_openai_api_key, _selected_model):
    if not _openai_api_key:
        return None, None
    
    try:
        # Configure LlamaIndex settings
        Settings.llm = OpenAI(model=model_options[_selected_model], temperature=0.1)
        Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
        
        # Load the CSV file
        csv_file = "schoenberg_letters_chunks.csv"
        if not os.path.exists(csv_file):
            st.error(f"CSV file '{csv_file}' not found. Please make sure it's in the same directory as this script.")
            return None, None
            
        df = pd.read_csv(csv_file)
        
        # Create documents from CSV rows
        documents = []
        for idx, row in df.iterrows():
            # Combine all columns into a single text, you may want to adjust this based on your CSV structure
            text_content = str(row['text'])
            letter_id = str(row['letter_id'])
            
                       
            # Create document with metadata
            doc = Document(
        text=text_content,
        metadata={
            "letter_id": letter_id,
            "chunk_id": str(row['chunk_id']),
            "chunk_type": str(row['chunk_type']),
            "chunk_index": int(row['chunk_index']),
            "total_chunks": int(row['total_chunks']),
            "char_count": int(row['char_count']),
            "word_count": int(row['word_count']),
            "row_index": idx
        }
            )
            documents.append(doc)
        
        # Create index
        index = VectorStoreIndex.from_documents(documents)
        
        return index, df
        
    except Exception as e:
        st.error(f"Error loading CSV: {str(e)}")
        return None, None

# Load index if API key is provided
if openai_api_key and (st.session_state.index is None or st.button("Reload with Selected Model")):
    with st.spinner(f"Loading and indexing your CSV with {selected_model}... This may take a moment."):
        st.session_state.index, st.session_state.csv_data = load_csv_and_create_index(openai_api_key, selected_model)
        if st.session_state.index:
            st.success(f"CSV loaded successfully with {selected_model}! You can now ask questions.")
            
  
# Create two columns for the main interface
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Chat")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Stellen Sie eine Frage zu den Briefen... / Ask a question about the letters..."):
        if not openai_api_key:
            st.error("Please enter your OpenAI API Key in the sidebar.")
        elif st.session_state.index is None:
            st.error("Please wait for the CSV to finish loading.")
        else:
            # Add user message to chat
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Searching through the letters..."):
                    try:
                        # Create query engine with more sources for the second window
                        query_engine = st.session_state.index.as_query_engine(
                            similarity_top_k=5,
                            response_mode="compact"
                        )
                        
                        # Get response
                        response = query_engine.query(prompt)
                        
                        # Display response
                        st.markdown(str(response))
                        
                        # Add assistant response to chat
                        st.session_state.messages.append({"role": "assistant", "content": str(response)})
                        
                        # Store source information for the second window
                        if hasattr(response, 'source_nodes'):
                            st.session_state.last_sources = response.source_nodes[:3]  # Top 3 sources
                        
                    except Exception as e:
                        error_msg = f"Error: {str(e)}"
                        st.error(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})

if st.button("Clear Chat History"):
        st.session_state.messages = []
        if hasattr(st.session_state, 'last_sources'):
            delattr(st.session_state, 'last_sources')
        st.rerun()      
        
with col2:
    st.subheader("Source Letters")
    
    if hasattr(st.session_state, 'last_sources') and st.session_state.last_sources:
        st.write("**Top 3 relevant letters:**")
        
        for i, source_node in enumerate(st.session_state.last_sources, 1):
            with st.expander(f"Letter {i}: {source_node.metadata.get('letter_id', 'Unknown ID')}"):
                # Show letter ID
                st.write(f"**Letter ID:** {source_node.metadata.get('letter_id', 'Unknown')}")
                
                # Show similarity score if available
                if hasattr(source_node, 'score'):
                    st.write(f"**Relevance Score:** {source_node.score:.3f}")
                
                # Show excerpt from the letter
                st.write("**Excerpt:**")
                # Limit the text to avoid overwhelming the interface
                text_preview = source_node.text[:500] + "..." if len(source_node.text) > 500 else source_node.text
                st.write(text_preview)
    else:
        st.write("Ask a question to see relevant letter sources here.")

# Instructions in sidebar
with st.sidebar:
    st.header("Instructions")
    st.write("""
    1. Enter your OpenAI API Key above
    2. Choose your preferred OpenAI model
    3. Make sure 'schoenberg_letters_chunks.csv' is in the same folder as this script
    4. Wait for the CSV to load
    5. Ask questions in German or English!
    
    **Model Information:**
    - **GPT-4o**: Most capable, best for complex analysis
    - **GPT-4o Mini**: Good balance of capability and speed
    - **GPT-3.5 Turbo**: Fastest and most economical
    
    **Example questions (German):**
    - Was schreibt Schönberg über Notation?
    - Gibt es Polemik oder Humor in den Briefen?
    - Was steht in den Briefen über Pelleas?
    - Fasse die wichtigsten Passagen über Verträge zusammen
    
    **Example questions (English):**
    - Do the letters discuss performances of Pelleas?
    - Are there sarcastic passages in the letters?
    - Summarize the main legal issues
    """)
    
    
    # Show CSV info if loaded
    if st.session_state.csv_data is not None:
        st.subheader("CSV Information")
        st.write(f"**Rows:** {len(st.session_state.csv_data)}")
        st.write(f"**Columns:** {list(st.session_state.csv_data.columns)}")