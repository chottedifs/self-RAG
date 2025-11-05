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
from src.rag.self_rag_pipeline import create_self_rag_pipeline
from src.vector_db.chroma_manager import get_vector_db
from src.llm.ollama_client import get_ollama_llm

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

if "rag_mode" not in st.session_state:
    st.session_state.rag_mode = "Self-RAG"

if "rag_pipeline" not in st.session_state:
    with st.spinner("🔄 Memuat RAG Pipeline..."):
        try:
            st.session_state.rag_pipeline = get_rag_pipeline()
            app_logger.info("Standard RAG pipeline loaded in Streamlit")
        except Exception as e:
            st.error(f"❌ Error loading RAG pipeline: {e}")
            app_logger.error(f"Failed to load RAG pipeline: {e}")
            st.stop()

if "self_rag_pipeline" not in st.session_state:
    with st.spinner("🔄 Memuat Self-RAG Pipeline..."):
        try:
            st.session_state.self_rag_pipeline = create_self_rag_pipeline()
            app_logger.info("Self-RAG pipeline loaded in Streamlit")
        except Exception as e:
            st.error(f"❌ Error loading Self-RAG pipeline: {e}")
            app_logger.error(f"Failed to load Self-RAG pipeline: {e}")
            st.stop()

if "llm_only" not in st.session_state:
    with st.spinner("🔄 Memuat LLM..."):
        try:
            st.session_state.llm_only = get_ollama_llm()
            app_logger.info("LLM loaded in Streamlit")
        except Exception as e:
            st.error(f"❌ Error loading LLM: {e}")
            app_logger.error(f"Failed to load LLM: {e}")
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
    
    # RAG Mode Selection
    st.subheader("🤖 Mode Sistem")
    
    rag_mode = st.radio(
        "Pilih mode sistem:",
        options=["Self-RAG", "RAG Standar", "Tanpa RAG"],
        help="""
        - **Self-RAG**: Sistem dengan refleksi diri, memfilter dokumen relevan, dan verifikasi jawaban
        - **RAG Standar**: Retrieval-Augmented Generation standar
        - **Tanpa RAG**: Hanya menggunakan LLM tanpa dokumen (pengetahuan umum)
        """
    )
    
    # Update session state if mode changed
    if rag_mode != st.session_state.rag_mode:
        st.session_state.rag_mode = rag_mode
        st.session_state.messages = []  # Clear chat history on mode change
        st.rerun()
    
    # Show mode description
    if rag_mode == "Self-RAG":
        st.info("🔍 **Self-RAG Aktif**: Sistem akan melakukan refleksi diri untuk memvalidasi relevansi dokumen dan kualitas jawaban.")
    elif rag_mode == "RAG Standar":
        st.info("📚 **RAG Standar Aktif**: Sistem akan mengambil dokumen relevan dan menggunakannya untuk menjawab.")
    else:
        st.warning("💭 **Tanpa RAG Aktif**: Sistem hanya menggunakan pengetahuan umum LLM tanpa mengakses dokumen.")
    
    st.markdown("---")
    
    # Settings
    st.subheader("⚙️ Pengaturan")
    
    # Filter by lembaga (only for RAG modes)
    if rag_mode != "Tanpa RAG":
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
        
        show_sources = st.checkbox("Tampilkan sumber", value=True)
    else:
        filter_lembaga = "Semua Lembaga"
        top_k = 5
        show_sources = False
    
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.1,
        help="Mengontrol kreativitas response (0=faktual, 1=kreatif)"
    )
    
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
        
        **Mode Sistem:**
        
        **Self-RAG** 🔍
        - Memutuskan apakah perlu retrieval dokumen
        - Memfilter dokumen yang relevan
        - Memverifikasi jawaban didukung dokumen
        - Mengevaluasi kualitas jawaban
        - Melakukan iterasi untuk memperbaiki jawaban
        
        **RAG Standar** 📚
        - Mengambil dokumen relevan berdasarkan similarity
        - Menggunakan dokumen sebagai konteks
        - Menghasilkan jawaban dari konteks
        
        **Tanpa RAG** 💭
        - Hanya menggunakan pengetahuan umum LLM
        - Tidak mengakses dokumen
        - Untuk pertanyaan umum
        """)

# Main chat interface
mode_icons = {
    "Self-RAG": "🔍",
    "RAG Standar": "📚",
    "Tanpa RAG": "💭"
}
current_icon = mode_icons.get(st.session_state.rag_mode, "🤖")

st.title(f"{current_icon} Chat dengan Dokumen Anda - Mode: {st.session_state.rag_mode}")
if st.session_state.rag_mode == "Tanpa RAG":
    st.markdown("💭 **Mode Tanpa RAG:** Hanya menggunakan pengetahuan umum LLM.")
else:
    st.markdown("Tanyakan apa saja tentang dokumen yang telah diindeks ke dalam sistem.")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        # Show mode badge for assistant messages
        if message["role"] == "assistant" and "mode" in message:
            mode_badge = {
                "Self-RAG": "🔍 Self-RAG",
                "RAG Standar": "📚 RAG Standar",
                "Tanpa RAG": "💭 Tanpa RAG"
            }
            st.caption(mode_badge.get(message["mode"], ""))
        
        st.markdown(message["content"])
        
        # Display Self-RAG metadata if available
        if message["role"] == "assistant" and "metadata" in message and message.get("mode") == "Self-RAG":
            metadata = message["metadata"]
            if metadata:
                with st.expander("🔍 Metadata Self-RAG"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Iterasi", metadata.get("iterations", 0))
                    with col2:
                        st.metric("Utility Score", f"{metadata.get('utility_score', 0)}/5")
                    with col3:
                        st.metric("Dokumen Relevan", metadata.get("relevant_docs", 0))
                    
                    if metadata.get("support_level"):
                        support_emoji = {
                            "fully_supported": "✅",
                            "partially_supported": "⚠️",
                            "no_support": "❌"
                        }
                        support_label = {
                            "fully_supported": "Fully Supported",
                            "partially_supported": "Partially Supported",
                            "no_support": "No Support"
                        }
                        st.info(f"{support_emoji.get(metadata['support_level'], '❓')} Support Level: {support_label.get(metadata['support_level'], 'Unknown')}")
        
        # Display sources if available
        if message["role"] == "assistant" and "sources" in message and show_sources:
            if message["sources"] and message.get("mode") != "Tanpa RAG":
                with st.expander(f"📚 Sumber ({len(message['sources'])} dokumen)"):
                    for source in message["sources"]:
                        lembaga_info = f"<br>🏛️ Lembaga: {source.get('lembaga', 'Unknown')}"
                        
                        # Show relevance confidence for Self-RAG
                        relevance_info = ""
                        if message.get("mode") == "Self-RAG" and source.get('relevance_confidence'):
                            relevance_info = f"<br>🎯 Relevance: {source['relevance_confidence']:.2%}"
                        
                        st.markdown(f"""
                        <div class="source-card">
                            📄 <strong>{source.get('file', 'Unknown')}</strong>{lembaga_info}<br>
                            📊 Similarity: {source.get('similarity', 0):.2%}{relevance_info}<br>
                            🏆 Rank: #{source.get('rank', 0)}
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
        spinner_text = {
            "Self-RAG": "🔍 Self-RAG sedang menganalisis dan memverifikasi...",
            "RAG Standar": "📚 Mencari dokumen dan membuat response...",
            "Tanpa RAG": "💭 Menghasilkan response dari pengetahuan umum..."
        }
        
        with st.spinner(spinner_text.get(st.session_state.rag_mode, "🤔 Memproses...")):
            try:
                response_data = {"answer": "", "sources": [], "metadata": {}}
                
                # Prepare filter dictionary
                filter_dict = None
                if st.session_state.rag_mode != "Tanpa RAG" and filter_lembaga != "Semua Lembaga":
                    filter_dict = {"lembaga": filter_lembaga}
                
                # Process based on selected mode
                if st.session_state.rag_mode == "Self-RAG":
                    # Use Self-RAG pipeline
                    response = st.session_state.self_rag_pipeline.query(
                        query=prompt,
                        top_k=top_k,
                        filter_dict=filter_dict
                    )
                    response_data["answer"] = response["answer"]
                    
                    # Format sources from Self-RAG
                    if response.get("sources"):
                        response_data["sources"] = [
                            {
                                "file": doc.get("filename", "Unknown"),
                                "lembaga": doc.get("lembaga", "Unknown"),
                                "similarity": doc.get("similarity_score", 0),
                                "rank": idx + 1,
                                "relevance_confidence": doc.get("relevance_confidence", 0)
                            }
                            for idx, doc in enumerate(response["sources"])
                        ]
                    
                    # Store metadata
                    response_data["metadata"] = response.get("metadata", {})
                    
                elif st.session_state.rag_mode == "RAG Standar":
                    # Use standard RAG pipeline
                    response = st.session_state.rag_pipeline.query(
                        query=prompt,
                        top_k=top_k,
                        temperature=temperature,
                        filter_dict=filter_dict
                    )
                    response_data["answer"] = response["answer"]
                    response_data["sources"] = response.get("sources", [])
                    
                else:  # Tanpa RAG
                    # Direct LLM query without RAG
                    system_prompt = """Kamu adalah asisten AI yang membantu menjawab pertanyaan umum.
Berikan jawaban yang akurat, informatif, dan dalam Bahasa Indonesia yang baik.
PENTING: Kamu TIDAK memiliki akses ke dokumen khusus, jawab berdasarkan pengetahuan umum saja."""
                    
                    answer = st.session_state.llm_only.generate(
                        prompt=prompt,
                        system_prompt=system_prompt,
                        temperature=temperature
                    )
                    response_data["answer"] = answer
                    response_data["sources"] = []
                    response_data["metadata"] = {"mode": "no_rag"}
                
                # Display response
                st.markdown(response_data["answer"])
                
                # Display Self-RAG metadata if available
                if st.session_state.rag_mode == "Self-RAG" and response_data.get("metadata"):
                    metadata = response_data["metadata"]
                    with st.expander("🔍 Metadata Self-RAG"):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Iterasi", metadata.get("iterations", 0))
                        with col2:
                            st.metric("Utility Score", f"{metadata.get('utility_score', 0)}/5")
                        with col3:
                            st.metric("Dokumen Relevan", metadata.get("relevant_docs", 0))
                        
                        if metadata.get("support_level"):
                            support_emoji = {
                                "fully_supported": "✅",
                                "partially_supported": "⚠️",
                                "no_support": "❌"
                            }
                            support_label = {
                                "fully_supported": "Fully Supported",
                                "partially_supported": "Partially Supported",
                                "no_support": "No Support"
                            }
                            st.info(f"{support_emoji.get(metadata['support_level'], '❓')} Support Level: {support_label.get(metadata['support_level'], 'Unknown')}")
                
                # Display sources
                if show_sources and response_data["sources"] and st.session_state.rag_mode != "Tanpa RAG":
                    with st.expander(f"📚 Sumber ({len(response_data['sources'])} dokumen)"):
                        for source in response_data["sources"]:
                            lembaga_info = f"<br>🏛️ Lembaga: {source.get('lembaga', 'Unknown')}"
                            
                            # Show relevance confidence for Self-RAG
                            relevance_info = ""
                            if st.session_state.rag_mode == "Self-RAG" and source.get('relevance_confidence'):
                                relevance_info = f"<br>🎯 Relevance: {source['relevance_confidence']:.2%}"
                            
                            st.markdown(f"""
                            <div class="source-card">
                                📄 <strong>{source.get('file', 'Unknown')}</strong>{lembaga_info}<br>
                                📊 Similarity: {source.get('similarity', 0):.2%}{relevance_info}<br>
                                🏆 Rank: #{source.get('rank', 0)}
                            </div>
                            """, unsafe_allow_html=True)
                
                # Add assistant message to chat
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response_data["answer"],
                    "sources": response_data["sources"],
                    "metadata": response_data.get("metadata", {}),
                    "mode": st.session_state.rag_mode
                })
                
                app_logger.info(f"Query processed successfully ({st.session_state.rag_mode}): {prompt[:50]}...")
                
            except Exception as e:
                error_msg = f"❌ Terjadi kesalahan: {str(e)}"
                st.error(error_msg)
                app_logger.error(f"Error processing query ({st.session_state.rag_mode}): {e}")
                
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                    "sources": [],
                    "mode": st.session_state.rag_mode
                })

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "Dibuat dengan ❤️ menggunakan Streamlit, Ollama, ChromaDB, dan IndoBERT"
    "</div>",
    unsafe_allow_html=True
)
