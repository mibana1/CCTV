"""Keep registered camera sources connected instead of pulling them on demand."""

from sqlite3 import Connection


def apply(connection: Connection) -> None:
    connection.execute(
        "UPDATE cameras SET source_on_demand = 0 WHERE source_on_demand <> 0"
    )
