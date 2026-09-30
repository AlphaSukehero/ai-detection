# EEG: Peak Window, Representation Selector, CSV Input, JSON API

Date: 2026-09-30
Status: Approved design

## Context

A requested change list asked for sliding-window inference, "any window >
0.5 = Abnormal", two models (line plot / scalogram), placeholder functions and
a fetch-based UI. Verified against the code on 2026-09-30:

- Sliding windows already cover the whole recording (2 s, 50% overlap — the
  window the models were trained on). 10 s windows would break the contract.
- "Any window > 0.5" was rejected: per-window thresholding flagged 8–22% of
  background windows on held-out CHB-MIT (precision 0.20).
- No line-plot model exists; the repo's real image functions exist, so
  placeholders are not added.

## Decisions

1. Verdict stays episode-based. Add the peak window (score, start, stop). If
   the peak crossed the threshold without forming an episode, show an
   "isolated high-scoring window" note. Store `peak_score` as a measurement.
2. Representation selector: scalogram | lineplot. Models are looked up as
   `model/eeg_<task>.keras` (scalogram) and `model/eeg_<task>_lineplot.keras`.
   With no validated model for the chosen representation the result is
   NOT ASSESSED; signal measurements still show.
3. `.csv` upload: one column per channel, optional header, a monotonic time
   column is dropped. A sampling rate (Hz) is required for CSV.
4. `POST /api/eeg/analyze` returns JSON; `static/eeg.js` posts FormData and
   renders without reload; the server-rendered form keeps working without JS.
   One shared analysis function serves both.

## Testing

Isolated spike vs real episode peak reporting; line-plot NOT ASSESSED; CSV
with / without sampling rate; API JSON shape; API save to patient record.
