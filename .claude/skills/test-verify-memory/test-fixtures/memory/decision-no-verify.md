# Storage Choice

We chose SQLite over PostgreSQL because the dataset is small (<10K rows) and the project runs on a single server. No need for connection pooling or concurrent writes.
