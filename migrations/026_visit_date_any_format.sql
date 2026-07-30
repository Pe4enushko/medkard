-- Прием.DATE arrives in more than one shape.
--
-- 1C sends DD.MM.YYYY ("29.07.2026"), which every date query here was written
-- against. But real cards also carry ISO timestamps ("2026-07-30T13:58:50"),
-- and to_date(iso, 'DD.MM.YYYY') does not merely mis-parse them — it reads
-- "2026" as the day and raises DatetimeFieldOverflow, so a single such card
-- fails the whole query with a 500. Providers differ and we do not control
-- what they send, so parsing is made tolerant here rather than at every call
-- site.
--
-- IMMUTABLE (the parse is deterministic) so it can back an expression index;
-- STRICT so a NULL date yields NULL instead of an exception.
CREATE OR REPLACE FUNCTION medkard_visit_date(raw text) RETURNS date
    LANGUAGE plpgsql IMMUTABLE STRICT AS
$fn$
BEGIN
    -- DD.MM.YYYY first: it is the 1C format and the overwhelming majority.
    IF raw ~ '^\d{2}\.\d{2}\.\d{4}' THEN
        RETURN to_date(left(raw, 10), 'DD.MM.YYYY');
    END IF;

    -- ISO date or timestamp, with or without a time part and timezone.
    IF raw ~ '^\d{4}-\d{2}-\d{2}' THEN
        RETURN left(raw, 10)::date;
    END IF;

    -- Anything else is not a date we can read. NULL keeps it out of every
    -- date-filtered result instead of failing the query for all other cards.
    RETURN NULL;
EXCEPTION
    WHEN OTHERS THEN
        -- Shape matched but values didn't (e.g. "31.02.2026"): same reasoning.
        RETURN NULL;
END;
$fn$;

COMMENT ON FUNCTION medkard_visit_date(text) IS
    'Прием.DATE -> date, accepting both DD.MM.YYYY (1C) and ISO. NULL when unparseable.';
