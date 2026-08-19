# Database migrations

SQLite schema migrations belong here. Migrations must be repeatable and kept
separate from runtime database files.

`cctv.db.database.initialize_database` applies migrations in version order and
records each successful migration in `schema_migrations`. Application startup
is safe to repeat: an already recorded version is not applied again.

The initial migration creates the camera registry. Version 2 adds normalized
analysis-run, analyzed-frame, and detection tables with query indexes. Version
3 adds the nullable per-run `track_id` used by object tracking while preserving
historical detection rows. Version 4 adds aggregate `tracks` and normalized
`track_observations`, including a backfill from existing tracked detections. Add
Version 5 adds the tracker-owned active state and a class/time/state query index.
Version 11 adds a unique MediaMTX stream path to camera registrations for local
browser previews. Version 12 adds an encrypted RTSP source column, a redacted
endpoint, on-demand behavior, and MediaMTX provisioning state. Version 13 adds
session-scoped `person_instances`, their `track_identity_links`, and the optional
face-event association used to stitch fragmented tracker IDs. Add later domain
tables as new versioned migration modules; never edit a migration that has been
deployed.
