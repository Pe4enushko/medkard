-- 029_grls_normalized_columns.sql
-- Хранимые нормализованные названия вместо индексов по выражению.
-- Идемпотентно, вперёд-только.
--
-- Замер: docs/grls-search-cost-2026-08-23.md. Индексы 028 построены по
-- выражению grls_norm(...), поэтому любой запрос, который планировщик не
-- смог взять из индекса, пересчитывал translate+regexp_replace на всех
-- 39 тыс. строк: 98 % стоимости такого запроса — сама нормализация, а не
-- поиск. Каскад промаха 311 → 15,8 мс, объём таблицы с индексами +10 %.

-- Зеркало grls.normalize.normalize_query (править вместе). Два отличия от 028:
--   * неразрывный пробел: '\s+' в Postgres его НЕ схлопывает, а str.split() в
--     питоне схлопывает — две половины системы расходились, что и ловит тест
--     test_grls_norm_parity_with_python;
--   * пробелы вокруг «+»: врач пишет «Амоксициллин + Клавулановая кислота»,
--     реестр хранит «Амоксициллин+клавулановая кислота». Без общей формы это
--     разные строки, и составное МНН опознавалось как «похожее».
-- Функция переопределяется ДО создания колонок, чтобы они считались по новой.
CREATE OR REPLACE FUNCTION grls_norm(t TEXT) RETURNS TEXT
LANGUAGE sql IMMUTABLE PARALLEL SAFE AS $$
    SELECT NULLIF(
        regexp_replace(
            btrim(regexp_replace(
                translate(lower(coalesce(t, '')), E'\u00a0ё"«»„“”‘’''®™©~', E' е'),
                '\s+', ' ', 'g')),
            '\s*\+\s*', '+', 'g'),
        '')
$$;

-- ВНИМАНИЕ: значения колонок вычисляются при записи. Любая правка grls_norm()
-- после этой миграции оставит в таблице старые значения — Postgres генерируемые
-- колонки не пересчитывает. Менять функцию можно только вместе с
-- принудительным пересчётом (ALTER ... DROP COLUMN / ADD COLUMN или полный
-- переимпорт реестра).

ALTER TABLE grls_registry
    ADD COLUMN IF NOT EXISTS inn_norm TEXT
        GENERATED ALWAYS AS (grls_norm(inn_name)) STORED,
    ADD COLUMN IF NOT EXISTS trade_norm TEXT
        GENERATED ALWAYS AS (grls_norm(trade_name)) STORED;

-- Точное совпадение — обычный b-tree, а не поиск по GIN триграмм.
CREATE INDEX IF NOT EXISTS grls_registry_inn_norm_idx   ON grls_registry (inn_norm);
CREATE INDEX IF NOT EXISTS grls_registry_trade_norm_idx ON grls_registry (trade_norm);

CREATE INDEX IF NOT EXISTS grls_registry_inn_norm_trgm_idx
    ON grls_registry USING GIN (inn_norm gin_trgm_ops);
CREATE INDEX IF NOT EXISTS grls_registry_trade_norm_trgm_idx
    ON grls_registry USING GIN (trade_norm gin_trgm_ops);

-- Индексы по выражению больше не используются: код читает колонки.
DROP INDEX IF EXISTS grls_registry_inn_trgm_idx;
DROP INDEX IF EXISTS grls_registry_trade_name_trgm_idx;

ANALYZE grls_registry;
