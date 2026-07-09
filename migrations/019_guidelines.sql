-- Migration 019: таблица guidelines — справочник клинреков (зеркало manifest.csv).
--
-- Единственный канонический источник справочника. Связывается с docs
-- внешним ключом по file_id (см. 021_docs_guideline_fk.sql).
--
-- mkb / age_category — массивы: один клинрек покрывает несколько кодов МКБ-10
-- и может относиться к нескольким возрастным категориям.
-- age_category хранится дословно как в CSV ('Взрослые', 'Дети').

CREATE TABLE IF NOT EXISTS guidelines (
    file_id       TEXT   PRIMARY KEY,               -- манифестный ID (= docs.file_id)
    name          TEXT,                             -- Наименование
    mkb           TEXT[] NOT NULL DEFAULT '{}',     -- МКБ-10: ['J20.0','J20.1']
    age_category  TEXT[] NOT NULL DEFAULT '{}',     -- ['Взрослые','Дети']
    developer     TEXT,                             -- Разработчик
    nps_status    TEXT,                             -- Статус одобрения НПС
    published_at  TEXT,                             -- Дата размещения (строка как в CSV)
    usage_status  TEXT                              -- Статус применения
);

-- GIN — матчинг по коду МКБ (code = ANY(mkb)) это горячий путь аудита.
CREATE INDEX IF NOT EXISTS guidelines_mkb_idx ON guidelines USING GIN (mkb);
