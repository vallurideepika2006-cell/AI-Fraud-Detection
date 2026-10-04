
import sqlite3
from pathlib import Path

# Find the main project folder
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Store the database inside the database folder
DATABASE_PATH = PROJECT_ROOT / "database" / "fraud_detection.db"


def get_connection():
    """Create a connection to the SQLite database."""
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database():
    """Create the transaction history table if it doesn't exist."""
    connection = get_connection()

    try:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS transaction_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                transaction_time REAL NOT NULL,
                amount REAL NOT NULL,
                prediction TEXT NOT NULL,
                risk_score REAL NOT NULL,
                risk_level TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)

        connection.commit()
        print("Database initialized successfully.")

    finally:
        connection.close()


if __name__ == "__main__":
    initialize_database()
