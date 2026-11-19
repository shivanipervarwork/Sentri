
import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL

# Find the main project folder
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load configuration from the .env file
load_dotenv(PROJECT_ROOT / ".env")

# Read connection settings
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_NAME = os.getenv("DB_NAME", "sentri_db")

# Create the database connection URL
DATABASE_URL = URL.create(
    drivername="postgresql+psycopg",
    username=DB_USER,
    password=DB_PASSWORD,
    host=DB_HOST,
    port=DB_PORT,
    database=DB_NAME,
)

# Create the SQLAlchemy engine
engine = create_engine(DATABASE_URL, pool_pre_ping=True)


def test_connection():
    """Test whether Python can connect to PostgreSQL."""
    with engine.connect() as connection:
        result = connection.execute(
            text("SELECT current_database()")
        )
        print("Connected to database:", result.scalar())


if __name__ == "__main__":
    test_connection()
