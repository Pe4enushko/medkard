-- A card cannot be both ignored and broken at the same time.
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'done_cards_not_both_ignored_and_broken'
    ) THEN
        ALTER TABLE done_cards
            ADD CONSTRAINT done_cards_not_both_ignored_and_broken
            CHECK (NOT (ignored AND broken));
    END IF;
END$$;
