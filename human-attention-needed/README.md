# Human attention needed

This folder is the **async decision channel** for the autonomous (ralph-loop) build.

Whenever the build hits something it cannot resolve from the specs, the code, or
sensible defaults (a genuine product/infra decision, a missing credential, a
blocked external dependency), it drops a numbered markdown file here:

```
human-attention-needed/NNN-short-slug.md
```

Each file contains: the question, the options with a recommended default, and
exactly what task is blocked.

## How to respond

1. Open the file, read the question and options.
2. Write your answer under the `## Answer` heading (edit the file in place).
3. Move the file into `human-attention-needed/done/` (or leave it; the loop
   treats a file with a filled-in `## Answer` as resolved).

The loop re-scans this folder every iteration. An answered/moved file unblocks
its dependent task. The loop never silently guesses on a flagged decision.
