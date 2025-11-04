# 🤖 RAG Chatbot dengan Self-RAG untuk Dokumen Institusional# 🤖 RAG Chatbot dengan Self-RAG untuk Dokumen Institusional# Sistem Klasifikasi Berita Hoax dengan LSTM dan GloVe



Sistem Retrieval-Augmented Generation (RAG) yang canggih untuk menjawab pertanyaan tentang dokumen institusional dengan dukungan **Self-Reflective RAG** yang dapat melakukan refleksi dan kritik terhadap jawabannya sendiri.



## ✨ Fitur UtamaSistem Retrieval-Augmented Generation (RAG) yang canggih untuk menjawab pertanyaan tentang dokumen institusional dengan dukungan **Self-Reflective RAG** yang dapat melakukan refleksi dan kritik terhadap jawabannya sendiri.Sistem ini dirancang untuk mengumpulkan data berita dari sumber publik dan mengklasifikasikannya sebagai berita hoax atau valid menggunakan deep learning dengan LSTM dan GloVe embeddings.



- 🧠 **Self-Reflective RAG** - Model dapat menilai relevansi, dukungan, dan utilitas jawabannya sendiri

- 🎓 **Fine-tuning Support** - Latih critic model dengan LoRA untuk meningkatkan kualitas refleksi

- 🇮🇩 **IndoBERT Embeddings** - Embeddings khusus untuk bahasa Indonesia## ✨ Fitur Utama## Struktur Proyek

- 🗄️ **ChromaDB Vector Store** - Penyimpanan vektor yang cepat dan efisien

- 🦙 **Ollama Integration** - Menggunakan LLM lokal (Llama, Mistral, dll)

- 🌐 **Streamlit UI** - Antarmuka web yang user-friendly

- 📄 **PDF Processing** - Ekstraksi dan chunking otomatis dari dokumen PDF- 🧠 **Self-Reflective RAG** - Model dapat menilai relevansi, dukungan, dan utilitas jawabannya sendiri```

- 🔍 **Multi-source Retrieval** - Mendukung filter berdasarkan lembaga/institusi

- 🎓 **Fine-tuning Support** - Latih critic model dengan LoRA untuk meningkatkan kualitas refleksitesis/

## 📁 Struktur Proyek

- 🇮🇩 **IndoBERT Embeddings** - Embeddings khusus untuk bahasa Indonesia├── data/

```

tesis/- 🗄️ **ChromaDB Vector Store** - Penyimpanan vektor yang cepat dan efisien│   ├── raw/              # Data mentah hasil scraping

├── cli.py                        # 🎯 Unified CLI tool (RECOMMENDED)

├── requirements.txt              # Dependencies- 🦙 **Ollama Integration** - Menggunakan LLM lokal (Llama, Mistral, dll)│   └── processed/        # Data yang sudah diproses

├── .gitignore                    # Git ignore rules

│- 🌐 **Streamlit UI** - Antarmuka web yang user-friendly├── embeddings/           # Pre-trained GloVe embeddings

├── data_resources/               # 📚 PDF documents

│   ├── Lembaga Kemahasiswaan dan Alumni/- 📄 **PDF Processing** - Ekstraksi dan chunking otomatis dari dokumen PDF├── models/               # Model yang sudah dilatih

│   ├── Lembaga Pengembangan Teknologi Informasi (LPTI)/

│   └── README.md- 🔍 **Multi-source Retrieval** - Mendukung filter berdasarkan lembaga/institusi├── notebooks/            # Jupyter notebooks untuk eksperimen

│

├── chroma_db/                    # 🗄️ Vector database (auto-generated)├── src/

├── models/                       # 🤖 Trained models (auto-generated)

├── fine_tuning_data/             # 📊 Training datasets (auto-generated)## 📁 Struktur Proyek│   ├── scraper/         # Modul web scraping

├── logs/                         # 📝 Application logs

││   ├── preprocessing/   # Modul preprocessing data

├── docs/                         # 📖 Documentation

│   ├── QUICK_START.md```│   └── model/           # Modul model LSTM

│   ├── TRAINING_QUICKSTART.md    # ⭐ Fine-tuning guide

│   ├── SELF_RAG_GUIDE.mdtesis/├── config.py            # Konfigurasi sistem

│   ├── CLI_USAGE.md

│   └── ...├── cli.py                        # 🎯 Unified CLI tool (RECOMMENDED)└── requirements.txt     # Dependencies

│

└── src/                          # 💻 Source code├── requirements.txt              # Dependencies```

    ├── config/                   # Configuration

    ├── data_processing/          # Data pipeline├── .gitignore                    # Git ignore rules

    ├── embeddings/               # Embedding models

    ├── vector_db/                # Vector storage│## Setup

    ├── llm/                      # LLM integration

    ├── rag/                      # RAG pipelines├── data_resources/               # 📚 PDF documents

    ├── fine_tuning/              # Model training

    ├── ui/                       # User interfaces│   ├── Lembaga Kemahasiswaan dan Alumni/1. Install dependencies:

    └── utils/                    # Utilities

```│   ├── Lembaga Pengembangan Teknologi Informasi (LPTI)/```bash



## 🚀 Quick Start│   └── README.mdpip install -r requirements.txt



### 1. Setup Environment│```



```bash├── chroma_db/                    # 🗄️ Vector database (auto-generated)

# Clone repository

git clone <repository-url>│2. Download NLTK data:

cd tesis

├── models/                       # 🤖 Trained models (auto-generated)```python

# Create virtual environment

python -m venv venv│   └── self_rag_critic/          # Fine-tuned critic modelimport nltk

```

│nltk.download('punkt')

### 2. ⚠️ ACTIVATE VIRTUAL ENVIRONMENT (IMPORTANT!)

├── fine_tuning_data/             # 📊 Training datasets (auto-generated)nltk.download('stopwords')

```powershell

# Windows PowerShell│   ├── train.jsonl```

.\venv\Scripts\Activate.ps1

│   ├── validation.jsonl

# Windows CMD

venv\Scripts\activate.bat│   └── test.jsonl3. Download GloVe embeddings:



# Linux/Mac│   - Download dari: https://nlp.stanford.edu/projects/glove/

source venv/bin/activate

```├── logs/                         # 📝 Application logs   - Extract file `glove.6B.100d.txt` ke folder `embeddings/`



**Ciri venv aktif**: Ada `(venv)` di prompt Anda:│

```

(venv) PS D:\DEVELOPMENT\tesis>├── docs/                         # 📖 Documentation## Penggunaan

```

│   ├── QUICK_START.md

### 3. Install Dependencies

│   ├── SELF_RAG_GUIDE.md### 1. Mengumpulkan Data

```bash

pip install -r requirements.txt│   ├── SYSTEM_OVERVIEW.md```python

```

│   └── ...from src.scraper.news_scraper import NewsScraperManager

### 4. Setup Ollama

│

```bash

# Install Ollama dari https://ollama.ai└── src/                          # 💻 Source codescraper = NewsScraperManager()

# Download model (pilih salah satu):

ollama pull llama2    ├── config/                   # Configurationscraper.scrape_all_sources()

ollama pull mistral

ollama pull deepseek-v3.1:671b-cloud    │   ├── settings.py```

```

    │   └── logger.py

### 5. Prepare Data

    ├── data_processing/          # Data pipeline### 2. Preprocessing Data

```bash

# Tempatkan file PDF di folder data_resources/    │   ├── pdf_extractor.py```python

# Jalankan data preparation

python cli.py prepare-data    │   ├── text_chunker.pyfrom src.preprocessing.text_preprocessor import TextPreprocessor

```

    │   └── prepare_data.py

### 6. Run Application

    ├── embeddings/               # Embedding modelspreprocessor = TextPreprocessor()

```bash

# Launch Streamlit UI    │   └── indobert_embeddings.pypreprocessor.process_dataset()

python cli.py run-ui

```    ├── vector_db/                # Vector storage```



Akses aplikasi di: `http://localhost:8501`    │   └── chroma_manager.py



## 🎯 Menggunakan CLI Tool (Recommended)    ├── llm/                      # LLM integration### 3. Melatih Model



CLI tool (`cli.py`) adalah cara terbaik untuk mengelola semua operasi:    │   └── ollama_client.py```python



### Data Preparation    ├── rag/                      # RAG pipelinesfrom src.model.lstm_classifier import HoaxClassifier



```bash    │   ├── pipeline.py           # Standard RAG

# Process semua PDFs

python cli.py prepare-data    │   └── self_rag_pipeline.py  # Self-RAG with criticclassifier = HoaxClassifier()



# Clear existing data dan reprocess    ├── fine_tuning/              # Model trainingclassifier.train()

python cli.py prepare-data --clear

    │   ├── prepare_dataset.py```

# Process single file

python cli.py prepare-data --file path/to/document.pdf    │   ├── train_critic.py

```

    │   └── evaluate_critic.py### 4. Prediksi

### Self-RAG Fine-tuning

    └── ui/                       # User interfaces```python

```bash

# 1. Generate training dataset (dari data REAL di ChromaDB)        └── app.pyfrom src.model.predictor import HoaxPredictor

python cli.py generate-dataset

```

# 2. Train critic model

python cli.py train-critic \predictor = HoaxPredictor()

  --base-model mistralai/Mistral-7B-v0.1 \

  --epochs 3 \## 🚀 Quick Startresult = predictor.predict("Teks berita yang akan diprediksi...")

  --batch-size 4

print(f"Prediksi: {result['label']} (Confidence: {result['confidence']:.2%})")

# 3. Evaluate model

python cli.py eval-critic \### 1. Setup Environment```

  --model-path ./models/self_rag_critic

```



**📖 Baca**: [docs/TRAINING_QUICKSTART.md](docs/TRAINING_QUICKSTART.md) untuk panduan lengkap fine-tuning```bash## Fitur



### Launch Services# Clone repository



```bashgit clone <repository-url>- ✅ Web scraping otomatis dari berbagai sumber berita

# Streamlit UI (Standard RAG)

python cli.py run-uicd tesis- ✅ Preprocessing teks bahasa Indonesia (tokenisasi, stemming, stopword removal)



# Streamlit UI (Self-RAG mode)- ✅ LSTM dengan GloVe pre-trained embeddings

python cli.py run-ui --self-rag

# Create virtual environment- ✅ Evaluasi model dengan berbagai metrik

# Custom port

python cli.py run-ui --port 8502python -m venv venv- ✅ Visualisasi hasil training

```

venv\Scripts\activate  # Windows- ✅ API untuk prediksi real-time

## 📚 Dokumentasi Lengkap

# source venv/bin/activate  # Linux/Mac

- 📖 [QUICK_START.md](docs/QUICK_START.md) - Panduan cepat memulai

- 🎓 [TRAINING_QUICKSTART.md](docs/TRAINING_QUICKSTART.md) - **Panduan fine-tuning (WAJIB BACA!)**## Arsitektur Model

- 🧠 [SELF_RAG_GUIDE.md](docs/SELF_RAG_GUIDE.md) - Panduan lengkap Self-RAG

- 🎯 [CLI_USAGE.md](docs/CLI_USAGE.md) - Referensi lengkap CLI# Install dependencies

- 🏗️ [SYSTEM_OVERVIEW.md](docs/SYSTEM_OVERVIEW.md) - Arsitektur sistem

- 📊 [DATASET_INFO.md](docs/DATASET_INFO.md) - Informasi datasetpip install -r requirements.txt- Input Layer: Sequence of words



## 🧠 Self-RAG: Apa itu?```- Embedding Layer: GloVe 100-dimensional embeddings



Self-RAG adalah teknik advanced RAG yang memungkinkan model untuk:- LSTM Layer: 128 units dengan dropout



1. **🤔 Retrieval Decision** - Memutuskan kapan perlu retrieve dokumen### 2. Setup Ollama- Dense Layer: 64 units dengan aktivasi ReLU

2. **✅ Relevance Check** - Menilai relevansi dokumen yang di-retrieve

3. **🔍 Support Verification** - Memverifikasi apakah jawaban didukung oleh dokumen- Output Layer: Sigmoid activation untuk klasifikasi biner

4. **⭐ Utility Evaluation** - Mengevaluasi utilitas jawaban untuk pertanyaan

```bash

Dengan fine-tuning critic model, sistem bisa belajar membuat refleksi yang lebih akurat sesuai dengan domain spesifik Anda!

# Install Ollama dari https://ollama.ai## Lisensi

## 🎓 Fine-tuning dengan Data Real

# Download model (pilih salah satu):

**Keunggulan**: Dataset generator sekarang menggunakan **data REAL dari dokumen Anda** (bukan random)!

ollama pull llama2Untuk keperluan penelitian/tesis.

```bash

# 1. Pastikan venv aktifollama pull mistral

.\venv\Scripts\Activate.ps1ollama pull deepseek-v3.1:671b-cloud

```

# 2. Prepare data dulu

python cli.py prepare-data### 3. Prepare Data



# 3. Generate dataset dari dokumen real```bash

python cli.py generate-dataset# Tempatkan file PDF di folder data_resources/

# Jalankan data preparation

# 4. Train (gunakan model open-source)python cli.py prepare-data

python cli.py train-critic \```

  --base-model mistralai/Mistral-7B-v0.1 \

  --epochs 3### 4. Run Application

```

```bash

**📖 PENTING**: Baca [docs/TRAINING_QUICKSTART.md](docs/TRAINING_QUICKSTART.md) sebelum fine-tuning!# Launch Streamlit UI

python cli.py run-ui

## ⚠️ Common Issues & Solutions```



### Issue 1: "ModuleNotFoundError: No module named 'peft'"Akses aplikasi di: `http://localhost:8501`



**Penyebab**: Virtual environment tidak aktif## 🎯 Menggunakan CLI Tool (Recommended)



**Solusi**:CLI tool (`cli.py`) adalah cara terbaik untuk mengelola semua operasi:

```bash

.\venv\Scripts\Activate.ps1### Data Preparation

python cli.py train-critic

``````bash

# Process semua PDFs

### Issue 2: "GatedRepoError: Cannot access gated repo"python cli.py prepare-data



**Penyebab**: Model Llama-2 memerlukan authentication# Clear existing data dan reprocess

python cli.py prepare-data --clear

**Solusi**: Gunakan model open-source

```bash# Process single file

python cli.py train-critic --base-model mistralai/Mistral-7B-v0.1python cli.py prepare-data --file path/to/document.pdf

``````



### Issue 3: "CUDA Out of Memory"### Self-RAG Fine-tuning



**Solusi**: Reduce batch size```bash

```bash# 1. Generate training dataset

python cli.py train-critic --batch-size 1python cli.py generate-dataset \

```  --output-dir ./fine_tuning_data \

  --num-retrieval 200 \

## 🔧 Configuration  --num-relevance 200 \

  --num-support 100 \

Edit `src/config/settings.py` untuk mengubah:  --num-utility 100



- Model Ollama yang digunakan# 2. Train critic model

- Chunk size dan overlappython cli.py train-critic \

- Top-k retrieval  --base-model meta-llama/Llama-2-7b-hf \

- Temperature dan parameter LLM  --output-dir ./models/self_rag_critic \

- Path ke data dan models  --epochs 3 \

  --batch-size 4 \

## 📊 Monitoring & Logging  --use-wandb



- Logs tersimpan di folder `logs/`# 3. Evaluate model

- Training metrics tersimpan di `models/*/`python cli.py eval-critic \

- Evaluation results di `models/evaluation_results.json`  --model-path ./models/self_rag_critic \

  --test-data ./fine_tuning_data/test.jsonl

## 🤝 Contributing```



Proyek ini untuk keperluan penelitian/tesis. Saran dan feedback sangat diterima!### Launch Services



## 📝 Lisensi```bash

# Streamlit UI (Standard RAG)

Untuk keperluan penelitian/tesis.python cli.py run-ui



---# Streamlit UI (Self-RAG mode)

python cli.py run-ui --self-rag

**⭐ Ingat**: SELALU aktifkan virtual environment sebelum menjalankan command! 

# Custom port

```bashpython cli.py run-ui --port 8502

.\venv\Scripts\Activate.ps1  # Windows

```# FastAPI server (jika sudah dibuat)

python cli.py run-api --port 8000
```

## 📚 Dokumentasi Lengkap

- 📖 [QUICK_START.md](docs/QUICK_START.md) - Panduan cepat memulai
- 🧠 [SELF_RAG_GUIDE.md](docs/SELF_RAG_GUIDE.md) - Panduan lengkap Self-RAG
- 🏗️ [SYSTEM_OVERVIEW.md](docs/SYSTEM_OVERVIEW.md) - Arsitektur sistem
- 📊 [DATASET_INFO.md](docs/DATASET_INFO.md) - Informasi dataset

## 🧠 Self-RAG: Apa itu?

Self-RAG adalah teknik advanced RAG yang memungkinkan model untuk:

1. **🤔 Retrieval Decision** - Memutuskan kapan perlu retrieve dokumen
2. **✅ Relevance Check** - Menilai relevansi dokumen yang di-retrieve
3. **🔍 Support Verification** - Memverifikasi apakah jawaban didukung oleh dokumen
4. **⭐ Utility Evaluation** - Mengevaluasi utilitas jawaban untuk pertanyaan

Dengan fine-tuning critic model, sistem bisa belajar membuat refleksi yang lebih akurat sesuai dengan domain spesifik Anda!

## 🔧 Configuration

Edit `src/config/settings.py` untuk mengubah:

- Model Ollama yang digunakan
- Chunk size dan overlap
- Top-k retrieval
- Temperature dan parameter LLM
- Path ke data dan models

## 📊 Monitoring & Logging

- Logs tersimpan di folder `logs/`
- Training metrics tersimpan di `models/*/`
- Evaluation results di `models/evaluation_results.json`

## 🤝 Contributing

Proyek ini untuk keperluan penelitian/tesis. Saran dan feedback sangat diterima!

## 📝 Lisensi

Untuk keperluan penelitian/tesis.
#   s e l f - R A G  
 