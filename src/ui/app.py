"""
Streamlit UI for RAG Chatbot
Interactive chat interface for the RAG system
"""
import streamlit as st
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config.settings import Config
from src.config.logger import app_logger
from src.rag.pipeline import get_rag_pipeline
from src.vector_db.chroma_manager import get_vector_db

# Page configuration
st.set_page_config(
    page_title=Config.APP_TITLE,
    page_icon=Config.APP_ICON,
    layout=Config.PAGE_CONFIG,
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .source-card {
        background-color: #f0f2f6;
        padding: 0.5rem;
        border-radius: 0.3rem;
        margin: 0.3rem 0;
        font-size: 0.85rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        color: white;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "rag_pipeline" not in st.session_state:
    with st.spinner("🔄 Memuat model AI..."):
        try:
            st.session_state.rag_pipeline = get_rag_pipeline()
            app_logger.info("RAG pipeline loaded in Streamlit")
        except Exception as e:
            st.error(f"❌ Error loading RAG pipeline: {e}")
            app_logger.error(f"Failed to load RAG pipeline: {e}")
            st.stop()

# Sidebar
with st.sidebar:
    st.title(f"{Config.APP_ICON} {Config.APP_TITLE}")
    st.markdown("---")
    
    # Database statistics
    st.subheader("📊 Database Statistics")
    try:
        vector_db = get_vector_db()
        stats = vector_db.get_stats()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Dokumen", stats["total_documents"])
        with col2:
            st.metric("Sumber Unik", stats["unique_sources"])
    except Exception as e:
        st.warning("Tidak dapat memuat statistik database")
        app_logger.error(f"Error loading DB stats: {e}")
    
    st.markdown("---")
    
    # Settings
    st.subheader("⚙️ Pengaturan")
    
    # Filter by lembaga
    try:
        vector_db = get_vector_db()
        all_docs = vector_db.collection.get()
        unique_lembaga = sorted(set(meta.get("lembaga", "Unknown") for meta in all_docs["metadatas"]))
        
        filter_lembaga = st.selectbox(
            "🏛️ Filter berdasarkan Lembaga",
            options=["Semua Lembaga"] + unique_lembaga,
            help="Pilih lembaga tertentu atau semua lembaga"
        )
    except Exception as e:
        filter_lembaga = "Semua Lembaga"
        app_logger.error(f"Error loading lembaga filter: {e}")
    
    top_k = st.slider(
        "Jumlah dokumen relevan",
        min_value=1,
        max_value=10,
        value=Config.TOP_K_RETRIEVAL,
        help="Jumlah dokumen yang akan diambil untuk konteks"
    )
    
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.1,
        help="Mengontrol kreativitas response (0=faktual, 1=kreatif)"
    )
    
    show_sources = st.checkbox("Tampilkan sumber", value=True)
    
    st.markdown("---")
    
    # Clear chat button
    if st.button("🗑️ Hapus Riwayat Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    st.markdown("---")
    
    # Info
    with st.expander("ℹ️ Tentang Aplikasi"):
        st.markdown("""
        **RAG Chatbot Indonesia** menggunakan:
        - 🧠 IndoBERT untuk embedding
        - 🗄️ ChromaDB untuk vector database
        - 🤖 Ollama untuk generasi response
        - 🎨 Streamlit untuk UI
        
        Tanyakan apa saja tentang dokumen yang telah diupload!
        """)

# Main chat interface
st.title(f"{Config.APP_ICON} Chat dengan Dokumen Anda")
st.markdown("Tanyakan apa saja tentang dokumen yang telah diindeks ke dalam sistem.")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Display sources if available
        if message["role"] == "assistant" and "sources" in message and show_sources:
            if message["sources"]:
                with st.expander(f"📚 Sumber ({len(message['sources'])} dokumen)"):
                    for source in message["sources"]:
                        lembaga_info = f"<br>🏛️ Lembaga: {source['lembaga']}" if source.get('lembaga') else ""
                        st.markdown(f"""
                        <div class="source-card">
                            📄 <strong>{source['file']}</strong>{lembaga_info}<br>
                            🎯 Similarity: {source['similarity']:.2%}<br>
                            📊 Rank: #{source['rank']}
                        </div>
                        """, unsafe_allow_html=True)

# Chat input
if prompt := st.chat_input("Ketik pertanyaan Anda di sini..."):
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("🤔 Mencari informasi dan membuat response..."):
            try:
                # Prepare filter dictionary
                filter_dict = None
                if filter_lembaga != "Semua Lembaga":
                    filter_dict = {"lembaga": filter_lembaga}
                
                # Query RAG pipeline
                response = st.session_state.rag_pipeline.query(
                    query=prompt,
                    top_k=top_k,
                    temperature=temperature,
                    filter_dict=filter_dict
                )
                
                # Display response
                st.markdown(response["answer"])
                
                # Display sources
                if show_sources and response["sources"]:
                    with st.expander(f"📚 Sumber ({len(response['sources'])} dokumen)"):
                        for source in response["sources"]:
                            lembaga_info = f"<br>🏛️ Lembaga: {source['lembaga']}" if source.get('lembaga') else ""
                            st.markdown(f"""
                            <div class="source-card">
                                📄 <strong>{source['file']}</strong>{lembaga_info}<br>
                                🎯 Similarity: {source['similarity']:.2%}<br>
                                📊 Rank: #{source['rank']}
                            </div>
                            """, unsafe_allow_html=True)
                
                # Add assistant message to chat
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response["answer"],
                    "sources": response["sources"]
                })
                
                app_logger.info(f"Query processed successfully: {prompt[:50]}...")
                
            except Exception as e:
                error_msg = f"❌ Terjadi kesalahan: {str(e)}"
                st.error(error_msg)
                app_logger.error(f"Error processing query: {e}")
                
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                    "sources": []
                })

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "Dibuat dengan ❤️ menggunakan Streamlit, Ollama, ChromaDB, dan IndoBERT"
    "</div>",
    unsafe_allow_html=True
)
