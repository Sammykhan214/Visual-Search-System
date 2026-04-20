visual-search-project/
├── data/
│ ├── raw/
│ │ └── images/ # 12,000 original images
│ ├── processed/
│ │ ├── embeddings/ # .npy files (one per image)
│ │ ├── metadata.csv # image_name, category_id, category_name
│ │ └── category_mapping.json
│ └── indexes/
│ └── faiss_index.bin # FAISS index file
│
├── src/
│ ├── preprocessing/
│ │ ├── load_data.py
│ │ └── preprocess.py
│ ├── features/
│ │ ├── extractor.py
│ │ └── model_factory.py
│ ├── search/
│ │ ├── indexer.py
│ │ ├── searcher.py
│ │ └── similarity.py
│ ├── pipeline/ # NEW: Orchestration for offline jobs
│ │ ├── build_index_pipeline.py
│ │ └── update_index_pipeline.py
│ ├── api/
│ │ ├── main.py
│ │ └── schemas.py
│ └── monitoring/ # NEW: Placeholder for logs/metrics
│ └── logger.py
│
├── notebooks/ # Exploration and evaluation
├── frontend/ # Optional Streamlit/React UI
├── config/
│ ├── config.yaml # Central configuration
│ └── logging.conf
├── scripts/
│ ├── run_indexing.sh
│ └── run_api.sh
├── tests/
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── README.md
└── .gitignore
