-- 028_grls_registry.sql
-- GRLS (State Register of Medicines) with registration-certificate statuses.
-- Replaces the ЕСКЛП `drugs` table. Idempotent, forward-only.
-- Design: docs/superpowers/specs/2026-08-17-grls-registry-design.md

CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Mirror of grls.normalize.normalize_query (keep in sync):
-- lower, drop quotes/®™©/~, ё→е, collapse whitespace, trim; empty → NULL.
CREATE OR REPLACE FUNCTION grls_norm(t TEXT) RETURNS TEXT
LANGUAGE sql IMMUTABLE PARALLEL SAFE AS $$
    SELECT NULLIF(
        btrim(regexp_replace(
            translate(lower(coalesce(t, '')), 'ё"«»„“”‘’''®™©~', 'е'),
            '\s+', ' ', 'g')),
        '')
$$;

CREATE TABLE IF NOT EXISTS grls_registry (
    id                BIGSERIAL PRIMARY KEY,
    status            TEXT NOT NULL CHECK (status IN (
        'Действующий',
        'Выдано по правилам ЕАЭС',
        'Действует, на подтверждении государственной регистрации',
        'Действует, в иностранных упаковках',
        'Приостановлено применение',
        'Истёкший',
        'Исключённый'
    )),
    reg_number        TEXT NOT NULL,
    registered_at     DATE,
    expires_at        DATE,
    annulled_at       DATE,
    holder            TEXT,
    holder_country    TEXT,
    trade_name        TEXT NOT NULL,
    inn_name          TEXT,
    forms             TEXT[] NOT NULL DEFAULT '{}',
    forms_raw         TEXT,
    dosage_forms      TEXT[] NOT NULL DEFAULT '{}',
    dispensing        TEXT[] NOT NULL DEFAULT '{}',
    is_substance      BOOLEAN NOT NULL DEFAULT false,
    production_stages TEXT,
    normative_docs    TEXT,
    pharm_group       TEXT,
    is_vital          BOOLEAN,
    narcotic_list     TEXT,
    is_orphan         BOOLEAN,
    row_hash          TEXT NOT NULL UNIQUE,
    imported_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);

COMMENT ON TABLE  grls_registry IS 'ГРЛС: одна строка = одна строка выгрузки xlsx (статус-файл). Номер РУ не уникален.';
COMMENT ON COLUMN grls_registry.id IS 'Суррогатный ключ.';
COMMENT ON COLUMN grls_registry.status IS 'Состояние записи реестра — маркер из строки 6 листа выгрузки, дословно (7 значений). Истина о статусе; из дат не выводится.';
COMMENT ON COLUMN grls_registry.reg_number IS 'Номер регистрационного удостоверения (xlsx: «Номер регистрационного удостоверения»). Не уникален: перерегистрация сохраняет номер, советские номера общие для нескольких фабрик.';
COMMENT ON COLUMN grls_registry.registered_at IS 'Дата выдачи РУ (xlsx: «Дата регистрации»).';
COMMENT ON COLUMN grls_registry.expires_at IS 'Срок действия РУ (xlsx: «Дата окончания действия регистрационного удостоверения»); NULL = бессрочно только для живых статусов.';
COMMENT ON COLUMN grls_registry.annulled_at IS 'Дата аннулирования (xlsx: «Дата аннулирования регистрационного удостоверения»); заполнена в основном у «Исключённый».';
COMMENT ON COLUMN grls_registry.holder IS 'Держатель РУ (xlsx: «Юридическое лицо, на имя которого выдано регистрационное удостоверение»).';
COMMENT ON COLUMN grls_registry.holder_country IS 'Страна держателя (xlsx: безымянная колонка H сразу после ЮЛ).';
COMMENT ON COLUMN grls_registry.trade_name IS 'Торговое наименование (xlsx: «Торговое наименование лекарственного препарата»). Хранится как есть; поиск через grls_norm().';
COMMENT ON COLUMN grls_registry.inn_name IS 'МНН / группировочное / химическое наименование (xlsx: «Международное непатентованное или химическое наименование»); «~» → NULL.';
COMMENT ON COLUMN grls_registry.forms IS 'Формы выпуска, split по «;» (xlsx: «Формы выпуска»): «форма, дозировка, фасовка - упаковка - … - условия отпуска».';
COMMENT ON COLUMN grls_registry.forms_raw IS 'xlsx: «Формы выпуска» как есть — для отладки парсера и пересчёта row_hash.';
COMMENT ON COLUMN grls_registry.dosage_forms IS 'Производное от forms: уникальные лекарственные формы (первый сегмент элемента до запятой).';
COMMENT ON COLUMN grls_registry.dispensing IS 'Производное от forms: уникальные условия отпуска (последний сегмент после « - »): По рецепту / Без рецепта / для стационаров / Не указано / In-Bulk / …';
COMMENT ON COLUMN grls_registry.is_substance IS 'Производное: фармацевтическая субстанция (номер ФС-… или форма «субстанция…»), не препарат; поиск по умолчанию исключает.';
COMMENT ON COLUMN grls_registry.production_stages IS 'Производители по стадиям (xlsx: «Сведения о стадиях производства»).';
COMMENT ON COLUMN grls_registry.normative_docs IS 'Реквизиты НД/ФС и редакций РУ (xlsx: «Нормативная документация»).';
COMMENT ON COLUMN grls_registry.pharm_group IS 'Фармако-терапевтическая группа (xlsx: «Фармако-терапевтическая группа»).';
COMMENT ON COLUMN grls_registry.is_vital IS 'Входит в перечень ЖНВЛП (xlsx: «Наличие лекарственного препарата в перечне ЖНВЛП», Да/Нет).';
COMMENT ON COLUMN grls_registry.narcotic_list IS 'Список ПКУ по ПП РФ № 681 (xlsx: «Наличие в лекарственном препарате наркотических средств, психотропных веществ…»): НII/ПII/ПIII/ПК; «~»/«Нет» → NULL.';
COMMENT ON COLUMN grls_registry.is_orphan IS 'Орфанный препарат (xlsx: «Орфанный»); в выгрузке 2026-08 пусто → NULL.';
COMMENT ON COLUMN grls_registry.row_hash IS 'sha256 нормализованного кортежа исходных полей (см. спеку §4.3) — ключ дедупликации и синка в engine.';
COMMENT ON COLUMN grls_registry.imported_at IS 'Момент загрузки строки.';

CREATE INDEX IF NOT EXISTS grls_registry_trade_name_trgm_idx
    ON grls_registry USING GIN (grls_norm(trade_name) gin_trgm_ops);
CREATE INDEX IF NOT EXISTS grls_registry_inn_trgm_idx
    ON grls_registry USING GIN (grls_norm(inn_name) gin_trgm_ops);
CREATE INDEX IF NOT EXISTS grls_registry_status_idx ON grls_registry (status);
CREATE INDEX IF NOT EXISTS grls_registry_reg_number_idx ON grls_registry (reg_number);
CREATE INDEX IF NOT EXISTS grls_registry_is_substance_idx ON grls_registry (is_substance);

CREATE TABLE IF NOT EXISTS grls_imports (
    id             BIGSERIAL PRIMARY KEY,
    archive_name   TEXT NOT NULL,
    registry_date  DATE NOT NULL,
    status_counts  JSONB NOT NULL,
    skipped_files  JSONB NOT NULL DEFAULT '[]',
    imported_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

COMMENT ON TABLE  grls_imports IS 'Журнал загрузок ГРЛС; последняя строка = текущая версия реестра.';
COMMENT ON COLUMN grls_imports.id IS 'Суррогатный ключ.';
COMMENT ON COLUMN grls_imports.archive_name IS 'Имя архива/каталога выгрузки.';
COMMENT ON COLUMN grls_imports.registry_date IS 'Дата выгрузки ГРЛС из строки 3 листа («по состоянию на DD.MM.YYYY») — версия реестра.';
COMMENT ON COLUMN grls_imports.status_counts IS 'Строк по статусам после дедупликации: {"Действующий": N, …}.';
COMMENT ON COLUMN grls_imports.skipped_files IS 'Пропущенные файлы (напр. «Изменённый»).';
COMMENT ON COLUMN grls_imports.imported_at IS 'Момент загрузки.';

-- ЕСКЛП table replaced by GRLS.
DROP TABLE IF EXISTS drugs;
