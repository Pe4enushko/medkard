-- Migration 006: view to detect done_cards rows with Chinese characters in
-- formal_result or diag_result, and a function to delete them.

CREATE OR REPLACE VIEW done_cards_chinese AS
SELECT id, card_guid, formal_result, diag_result
FROM done_cards
WHERE formal_result::text ~ '[\x4E00-\x9FFF\x3400-\x4DBF\xF900-\xFAFF]'
   OR diag_result::text  ~ '[\x4E00-\x9FFF\x3400-\x4DBF\xF900-\xFAFF]';


CREATE OR REPLACE FUNCTION delete_chinese_done_cards()
RETURNS integer
LANGUAGE plpgsql AS $$
DECLARE
    deleted integer;
BEGIN
    DELETE FROM done_cards
    WHERE id IN (SELECT id FROM done_cards_chinese);
    GET DIAGNOSTICS deleted = ROW_COUNT;
    RETURN deleted;
END;
$$;
