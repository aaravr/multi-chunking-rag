## Work Order: Harden "Items of Note" Numeric Coverage Queries

### Problem
Previous implementations incorrectly anchored "items of note" queries to:
- adjusted measures definitions
- LCR explanatory text
- financial statement notes

### Required Behavior
For the query:
> "List the items of note affecting 2024 net income and the aggregate impact"

The system MUST:
- Anchor in MD&A / Financial Results Review
- Extract only explicitly enumerated items
- Use after-tax figures for aggregate impact
- Never treat adjusted-measure definitions as items of note

### Acceptance Criteria
- Unit test `test_items_of_note_numeric_list.py` must pass
- Aggregate impact must equal sum(after-tax items)
- Verifier must return YES