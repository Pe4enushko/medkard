-- 001_extensions.sql
-- Prerequisites: run as a superuser or a role with CREATE EXTENSION privilege.

CREATE EXTENSION IF NOT EXISTS vector;       -- pgvector
CREATE EXTENSION IF NOT EXISTS "uuid-ossp"; -- uuid_generate_v4()
