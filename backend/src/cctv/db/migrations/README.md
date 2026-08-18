# Database migrations

SQLite schema migrations belong here. Migrations must be repeatable and kept
separate from runtime database files.

`cctv.db.database.initialize_database` applies migrations in version order and
records each successful migration in `schema_migrations`. Application startup
is safe to repeat: an already recorded version is not applied again.

The initial migration creates the camera registry. Version 2 adds normalized
analysis-run, analyzed-frame, and detection tables with query indexes. Version
3 adds the nullable per-run `track_id` used by object tracking while preserving
historical detection rows. Add later domain tables as new versioned migration
modules; never edit a migration that has been deployed.
