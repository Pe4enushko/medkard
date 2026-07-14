-- Migration 021: FK docs.file_id → guidelines.file_id.
--
-- Требует, чтобы guidelines была заполнена seed-скриптом
-- (scripts/seed-guidelines.py) ДО этой миграции. Если seed не выполнен,
-- docs.file_id окажутся сиротами и миграция ЯВНО падает — migrate.sh
-- (ON_ERROR_STOP=1) остановится здесь. Это штатная защита порядка:
-- выполните seed и повторите migrate.sh.
--
-- Guard: constraint добавляется, только если его ещё нет (идемпотентно).

DO $$
DECLARE
    orphan_count integer;
BEGIN
    SELECT count(DISTINCT file_id) INTO orphan_count
    FROM docs
    WHERE file_id IS NOT NULL
      AND file_id NOT IN (SELECT file_id FROM guidelines);

    IF orphan_count > 0 THEN
        RAISE EXCEPTION
            'docs содержит % file_id без строки в guidelines — выполните scripts/seed-guidelines.py перед этой миграцией',
            orphan_count;
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'docs_file_id_fkey'
    ) THEN
        ALTER TABLE docs
            ADD CONSTRAINT docs_file_id_fkey
            FOREIGN KEY (file_id) REFERENCES guidelines(file_id);
    END IF;
END$$;
