-- A card cannot be both ignored and broken at the same time.
ALTER TABLE done_cards
    ADD CONSTRAINT done_cards_not_both_ignored_and_broken
    CHECK (NOT (ignored AND broken));
