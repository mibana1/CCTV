# Architecture decision records

This directory stores accepted or superseded architecture decisions. Open
questions remain only in `docs/OPEN_QUESTIONS.md` until evidence supports a
decision.

Use the naming convention:

```text
001-stream-source.md
002-hiperwall-auth.md
003-display-stream-mode.md
```

Copy `000-template.md`, assign the next number, and record the related open
question ID. A decision is complete only when its verification evidence is
included or referenced.

Never rewrite historical decisions silently. Mark an obsolete ADR as
`Superseded` and link the replacement.
