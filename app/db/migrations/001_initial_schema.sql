-- ### FILE: app/db/migrations/001_initial_schema.sql
-- ==============================================================================
-- SMART NOTES - INITIAL DATABASE SCHEMA MIGRATION
-- ==============================================================================

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    username VARCHAR(64) NOT NULL UNIQUE,
    email VARCHAR(255) NOT NULL UNIQUE,
    hashed_password VARCHAR(255) NOT NULL,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT chk_username_length CHECK (char_length(username) >= 3),
    CONSTRAINT chk_email_format CHECK (email ~* '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$')
);

CREATE INDEX IF NOT EXISTS idx_users_username ON users (username);
CREATE INDEX IF NOT EXISTS idx_users_email ON users (email);

CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NULL REFERENCES users(id) ON DELETE CASCADE,
    title VARCHAR(255) NOT NULL,
    description TEXT NULL,
    original_filename VARCHAR(255) NOT NULL,
    file_path VARCHAR(1024) NOT NULL,
    file_size_bytes BIGINT NOT NULL DEFAULT 0,
    mime_type VARCHAR(100) NOT NULL DEFAULT 'image/jpeg',
    status VARCHAR(32) NOT NULL DEFAULT 'PENDING',
    error_message TEXT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT chk_document_status CHECK (status IN ('PENDING', 'PROCESSING', 'COMPLETED', 'FAILED')),
    CONSTRAINT chk_file_size_positive CHECK (file_size_bytes >= 0)
);

CREATE INDEX IF NOT EXISTS idx_documents_user_id ON documents (user_id);
CREATE INDEX IF NOT EXISTS idx_documents_status ON documents (status);
CREATE INDEX IF NOT EXISTS idx_documents_created_at ON documents (created_at DESC);

CREATE TABLE IF NOT EXISTS pages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    page_number INTEGER NOT NULL DEFAULT 1,
    raw_image_path VARCHAR(1024) NOT NULL,
    processed_image_path VARCHAR(1024) NULL,
    deskewed_image_path VARCHAR(1024) NULL,
    debug_dir_path VARCHAR(1024) NULL,
    width INTEGER NOT NULL DEFAULT 0,
    height INTEGER NOT NULL DEFAULT 0,
    skew_angle REAL NOT NULL DEFAULT 0.0,
    status VARCHAR(32) NOT NULL DEFAULT 'PENDING',
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT chk_page_number_positive CHECK (page_number >= 1),
    CONSTRAINT chk_page_dimensions CHECK (width >= 0 AND height >= 0),
    CONSTRAINT chk_page_status CHECK (status IN ('PENDING', 'PREPROCESSED', 'SEGMENTED', 'COMPLETED', 'FAILED')),
    CONSTRAINT uq_document_page UNIQUE (document_id, page_number)
);

CREATE INDEX IF NOT EXISTS idx_pages_document_id ON pages (document_id);
CREATE INDEX IF NOT EXISTS idx_pages_status ON pages (status);

CREATE TABLE IF NOT EXISTS text_lines (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    page_id UUID NOT NULL REFERENCES pages(id) ON DELETE CASCADE,
    line_index INTEGER NOT NULL DEFAULT 0,
    bbox_x INTEGER NOT NULL,
    bbox_y INTEGER NOT NULL,
    bbox_w INTEGER NOT NULL,
    bbox_h INTEGER NOT NULL,
    cropped_image_path VARCHAR(1024) NULL,
    recognized_text TEXT NOT NULL DEFAULT '',
    confidence REAL NOT NULL DEFAULT 0.0,
    is_header BOOLEAN NOT NULL DEFAULT FALSE,
    is_math BOOLEAN NOT NULL DEFAULT FALSE,
    indent_level INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT chk_bbox_values CHECK (bbox_x >= 0 AND bbox_y >= 0 AND bbox_w > 0 AND bbox_h > 0),
    CONSTRAINT chk_confidence_range CHECK (confidence >= 0.0 AND confidence <= 1.0),
    CONSTRAINT chk_indent_level CHECK (indent_level >= 0),
    CONSTRAINT uq_page_line_index UNIQUE (page_id, line_index)
);

CREATE INDEX IF NOT EXISTS idx_text_lines_page_id ON text_lines (page_id);
CREATE INDEX IF NOT EXISTS idx_text_lines_page_index ON text_lines (page_id, line_index);

CREATE TABLE IF NOT EXISTS export_documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    export_format VARCHAR(32) NOT NULL DEFAULT 'MARKDOWN',
    file_path VARCHAR(1024) NOT NULL,
    content TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT chk_export_format CHECK (export_format IN ('MARKDOWN', 'TXT', 'LATEX', 'JSON'))
);

CREATE INDEX IF NOT EXISTS idx_export_documents_doc_id ON export_documents (document_id);
CREATE INDEX IF NOT EXISTS idx_export_documents_format ON export_documents (document_id, export_format);
