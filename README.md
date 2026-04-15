visual-search-project/
│
├── data/                       # All dataset-related files
│   ├── raw/                    # Original GPR1200 data (unchanged)
│   │   └── images/             # The 12,000 images from download
│   │       ├── 0001_xxx.jpg
│   │       └── ...
│   │
│   ├── processed/              # Preprocessed images & metadata
│   │   ├── embeddings/         # Saved feature vectors (numpy files)
│   │   ├── metadata.csv        # Image paths + category IDs
│   │   └── category_mapping.json  # Maps ID to category name
│   │
│   └── indexes/                # FAISS/vector DB indexes
│       └── gpr1200.index
│
├── src/                        # Source code
│   ├── preprocessing/          # Data loading & preparation
│   │   ├── __init__.py
│   │   ├── load_data.py        # Functions to read GPR1200
│   │   └── preprocess.py       # Resizing, normalization
│   │
│   ├── features/               # Feature extraction
│   │   ├── __init__.py
│   │   ├── extractor.py        # Model loading & embedding generation
│   │   └── model_factory.py    # Choose between ResNet/DINOv2/CLIP
│   │
│   ├── search/                 # Search logic
│   │   ├── __init__.py
│   │   ├── indexer.py          # Build FAISS index
│   │   ├── searcher.py         # Query & retrieve results
│   │   └── similarity.py       # Distance metrics
│   │
│   └── api/                    # FastAPI backend
│       ├── __init__.py
│       ├── main.py             # API endpoints
│       └── schemas.py          # Request/response models
│
├── notebooks/                  # Jupyter notebooks for exploration
│   ├── 01_explore_data.ipynb   # First look at GPR1200
│   ├── 02_generate_embeddings.ipynb
│   └── 03_evaluate_search.ipynb
│
├── frontend/                   # React/UI (optional)
│   └── ...
│
├── config/                     # Configuration files
│   ├── config.yaml             # Paths, model params, search settings
│   └── logging.conf
│
├── scripts/                    # Utility scripts
│   ├── download_data.sh        # Download GPR1200
│   ├── run_indexing.py         # One-click indexing pipeline
│   └── run_api.sh              # Start the server
│
├── tests/                      # Unit tests
│   ├── test_preprocessing.py
│   ├── test_search.py
│   └── test_api.py
│
├── requirements.txt            # Python dependencies
├── Dockerfile                  # For containerization
├── docker-compose.yml          # For running with vector DB
├── README.md                   # Project documentation
└── .gitignore                  # Ignore data/, indexes/, __pycache__/