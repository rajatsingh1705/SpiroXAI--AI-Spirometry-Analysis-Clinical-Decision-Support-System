-- ============================================================
-- SpiroXAI — Supabase Database Schema
-- Run this in Supabase SQL Editor to create the predictions table
-- ============================================================

CREATE TABLE IF NOT EXISTS predictions (
    id              UUID        DEFAULT gen_random_uuid() PRIMARY KEY,
    patient_name    TEXT        DEFAULT 'Unknown',   -- ADDED
    patient_age     INTEGER,
    patient_sex     INTEGER,
    race            TEXT,
    weight          FLOAT,
    height          FLOAT,
    bmi             FLOAT,
    pef             FLOAT,
    fef2575         FLOAT,
    extrapolated_volume      FLOAT,
    forced_expiratory_time   FLOAT,
    acceptable_curves        INTEGER,
    predicted_class TEXT        NOT NULL,
    confidence_normal       FLOAT,
    confidence_obstruction  FLOAT,
    confidence_restriction  FLOAT,
    top_features    JSONB,
    created_at      TIMESTAMPTZ DEFAULT now()
);

-- Index for fast retrieval sorted by time
CREATE INDEX IF NOT EXISTS idx_predictions_created_at
    ON predictions (created_at DESC);

-- Row Level Security (optional — disabled for demo)
ALTER TABLE predictions ENABLE ROW LEVEL SECURITY;

-- Allow all operations for anon key (demo mode)
CREATE POLICY "Allow all for demo"
    ON predictions
    FOR ALL
    USING (true)
    WITH CHECK (true);
