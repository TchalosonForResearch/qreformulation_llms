# Manual validation of the anchor proxy

Review each unique query without looking at the retrieval delta.

For each row:

1. Read `original_text` and `file_paths`.
2. Write the source-identifying tokens or short phrases that are genuinely
   present in the original query in `manual_anchor_tokens`.
3. Mark `proxy_correct_yes_no` as `yes` only when the extracted combined
   anchors are an acceptable token-level approximation.
4. Record omitted anchors in `missing_anchor_tokens`.
5. Record generic or spurious extracted anchors in
   `spurious_anchor_tokens`.
6. Use `reviewer_notes` for ambiguous cases.

The worksheet is stratified by task with a deterministic seed. It must be
reviewed before the manuscript calls the score a validated APR. Until then,
use the expression “lexical anchor-preservation proxy”.
