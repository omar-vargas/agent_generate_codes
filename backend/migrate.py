import os
import json
from pathlib import Path
from models import User, Session, Document, get_db, create_tables
from sqlalchemy.orm import Session as DBSession
from sqlalchemy import text
import shutil

def add_role_column(db: DBSession):
    """Add role column to users table if it doesn't exist"""
    try:
        # Check if role column exists
        result = db.execute(text("PRAGMA table_info(users)")).fetchall()
        columns = [row[1] for row in result]
        if 'role' not in columns:
            # Add role column with default 'user'
            db.execute(text("ALTER TABLE users ADD COLUMN role VARCHAR(20) DEFAULT 'user' NOT NULL"))
            print("Added role column to users table")
        else:
            print("Role column already exists")
    except Exception as e:
        print(f"Error adding role column: {e}")

def migrate_users(db: DBSession):
    """Migrate hardcoded users to database"""
    users_data = {
        "omar": {"password": "1234", "role": "user"},
        "elsa": {"password": "abcd", "role": "user"},
        "admin": {"password": "admin", "role": "admin"},
        "admin1": {"password": "admin", "role": "admin"}
    }

    for username, data in users_data.items():
        # Check if user already exists
        existing_user = db.query(User).filter(User.username == username).first()
        if not existing_user:
            user = User(username=username, password=data["password"], role=data["role"])
            db.add(user)
            print(f"Migrated user: {username} with role {data['role']}")
        else:
            # Update role
            existing_user.role = data["role"]
            print(f"Updated role for user: {username} to {data['role']}")

    db.commit()

def migrate_sessions(db: DBSession):
    """Migrate session data from JSON files to database"""
    sessions_dir = Path("sessions")
    if not sessions_dir.exists():
        print("Sessions directory not found")
        return

    for session_file in sessions_dir.glob("session_*.json"):
        try:
            with open(session_file, "r", encoding="utf-8") as f:
                session_data = json.load(f)

            session_id = session_file.stem.replace("session_", "")

            # Extract user from session_id (format: username--uuid)
            if "--" in session_id:
                username = session_id.split("--")[0]
                user = db.query(User).filter(User.username == username).first()
                if not user:
                    print(f"User {username} not found for session {session_id}")
                    continue
                user_id = user.id
            else:
                # For sessions without user prefix, assign to first admin user
                admin_user = db.query(User).filter(User.username == "admin").first()
                if admin_user:
                    user_id = admin_user.id
                else:
                    print(f"No admin user found for session {session_id}")
                    continue

            # Check if session already exists
            existing_session = db.query(Session).filter(Session.id == session_id).first()
            if existing_session:
                continue

            # Create session
            session = Session(
                id=session_id,
                user_id=user_id,
                codes=json.dumps(session_data.get("codes", [])),
                data=json.dumps(session_data.get("data", [])),
                questions=json.dumps(session_data.get("questions", [])),
                feedback=session_data.get("feedback", "")
            )
            db.add(session)
            print(f"Migrated session: {session_id}")

        except Exception as e:
            print(f"Error migrating session {session_file}: {e}")

    db.commit()

def add_document_columns(db: DBSession):
    """Add filename and file_type columns to documents table if they don't exist"""
    try:
        # Check if columns exist
        result = db.execute(text("PRAGMA table_info(documents)")).fetchall()
        columns = [row[1] for row in result]
        if 'filename' not in columns:
            db.execute(text("ALTER TABLE documents ADD COLUMN filename VARCHAR(255)"))
            print("Added filename column to documents table")
        if 'file_type' not in columns:
            db.execute(text("ALTER TABLE documents ADD COLUMN file_type VARCHAR(50)"))
            print("Added file_type column to documents table")
    except Exception as e:
        print(f"Error adding document columns: {e}")

def migrate_documents(db: DBSession):
    """Migrate consolidated and output documents to database"""
    # Migrate consolidated files
    consolidated_dir = Path(".")
    for consolidated_file in consolidated_dir.glob("consolidado_*.txt"):
        try:
            session_id = consolidated_file.stem.replace("consolidado_", "")

            with open(consolidated_file, "r", encoding="utf-8") as f:
                content = f.read()

            # Check if document already exists
            existing_doc = db.query(Document).filter(
                Document.session_id == session_id,
                Document.type == "consolidated"
            ).first()
            if existing_doc:
                continue

            document = Document(
                session_id=session_id,
                type="consolidated",
                content=content
            )
            db.add(document)
            print(f"Migrated consolidated document: {session_id}")

        except Exception as e:
            print(f"Error migrating consolidated file {consolidated_file}: {e}")

    # Migrate output files
    output_dir = Path("archivos_guardados")
    if output_dir.exists():
        for output_file in output_dir.glob("output_*.txt"):
            try:
                session_id = output_file.stem.replace("output_", "")

                with open(output_file, "r", encoding="utf-8") as f:
                    content = f.read()

                # Check if document already exists
                existing_doc = db.query(Document).filter(
                    Document.session_id == session_id,
                    Document.type == "output"
                ).first()
                if existing_doc:
                    continue

                document = Document(
                    session_id=session_id,
                    type="output",
                    content=content
                )
                db.add(document)
                print(f"Migrated output document: {session_id}")

            except Exception as e:
                print(f"Error migrating output file {output_file}: {e}")

    db.commit()

def add_embedding_column(db: DBSession):
    """Add embedding column to codes table if it doesn't exist"""
    try:
        # Check if embedding column exists
        result = db.execute(text("PRAGMA table_info(codes)")).fetchall()
        columns = [row[1] for row in result]
        if 'embedding' not in columns:
            # Add embedding column
            db.execute(text("ALTER TABLE codes ADD COLUMN embedding TEXT"))
            print("Added embedding column to codes table")
        else:
            print("Embedding column already exists")
    except Exception as e:
        print(f"Error adding embedding column: {e}")

def add_ontology_columns(db: DBSession):
    """Add ontology-related columns to codes table if they don't exist"""
    try:
        # Check if ontology_concept_id column exists
        result = db.execute(text("PRAGMA table_info(codes)")).fetchall()
        columns = [row[1] for row in result]
        if 'ontology_concept_id' not in columns:
            # Add ontology_concept_id column
            db.execute(text("ALTER TABLE codes ADD COLUMN ontology_concept_id INTEGER REFERENCES ontology_concepts(id)"))
            print("Added ontology_concept_id column to codes table")
        else:
            print("Ontology concept column already exists")
    except Exception as e:
        print(f"Error adding ontology columns: {e}")

def add_session_name_column(db: DBSession):
    """Add name column to sessions table if it doesn't exist"""
    try:
        # Check if name column exists
        result = db.execute(text("PRAGMA table_info(sessions)")).fetchall()
        columns = [row[1] for row in result]
        if 'name' not in columns:
            # Add name column
            db.execute(text("ALTER TABLE sessions ADD COLUMN name VARCHAR(255)"))
            print("Added name column to sessions table")
        else:
            print("Name column already exists")
    except Exception as e:
        print(f"Error adding name column: {e}")

def add_search_indexes(db: DBSession):
    """Add database indexes to optimize search performance"""
    try:
        # Indexes for search optimization
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_codes_name ON codes (name)",
            "CREATE INDEX IF NOT EXISTS idx_codes_user_id ON codes (user_id)",
            "CREATE INDEX IF NOT EXISTS idx_codes_session_id ON codes (session_id)",
            "CREATE INDEX IF NOT EXISTS idx_codes_category_id ON codes (category_id)",
            "CREATE INDEX IF NOT EXISTS idx_codes_created_at ON codes (created_at)",
            "CREATE INDEX IF NOT EXISTS idx_codes_embedding ON codes (embedding) WHERE embedding IS NOT NULL",

            "CREATE INDEX IF NOT EXISTS idx_documents_session_id ON documents (session_id)",
            "CREATE INDEX IF NOT EXISTS idx_documents_type ON documents (type)",
            "CREATE INDEX IF NOT EXISTS idx_documents_timestamp ON documents (timestamp)",

            "CREATE INDEX IF NOT EXISTS idx_categories_user_id ON categories (user_id)",
            "CREATE INDEX IF NOT EXISTS idx_categories_name ON categories (name)",
            "CREATE INDEX IF NOT EXISTS idx_categories_created_at ON categories (created_at)",

            "CREATE INDEX IF NOT EXISTS idx_ontology_concepts_ontology_id ON ontology_concepts (ontology_id)",
            "CREATE INDEX IF NOT EXISTS idx_ontology_concepts_name ON ontology_concepts (name)",
            "CREATE INDEX IF NOT EXISTS idx_ontology_concepts_embedding ON ontology_concepts (embedding) WHERE embedding IS NOT NULL",

            "CREATE INDEX IF NOT EXISTS idx_ontologies_user_id ON ontologies (user_id)",
            "CREATE INDEX IF NOT EXISTS idx_ontologies_name ON ontologies (name)",

            "CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions (user_id)",
            "CREATE INDEX IF NOT EXISTS idx_sessions_timestamp ON sessions (timestamp)",

            "CREATE INDEX IF NOT EXISTS idx_users_username ON users (username)",
            "CREATE INDEX IF NOT EXISTS idx_users_role ON users (role)"
        ]

        for index_sql in indexes:
            try:
                db.execute(text(index_sql))
                print(f"Created index: {index_sql.split(' ON ')[1].split(' (')[0]}")
            except Exception as e:
                print(f"Error creating index: {e}")

        print("Search indexes added successfully")
    except Exception as e:
        print(f"Error adding search indexes: {e}")

def create_ontology_tables(db: DBSession):
    """Create ontology tables if they don't exist"""
    try:
        # Create ontologies table
        db.execute(text("""
            CREATE TABLE IF NOT EXISTS ontologies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name VARCHAR(255) NOT NULL,
                description TEXT,
                user_id INTEGER,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """))

        # Create ontology_concepts table
        db.execute(text("""
            CREATE TABLE IF NOT EXISTS ontology_concepts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name VARCHAR(255) NOT NULL,
                description TEXT,
                ontology_id INTEGER,
                parent_id INTEGER,
                embedding TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (ontology_id) REFERENCES ontologies(id),
                FOREIGN KEY (parent_id) REFERENCES ontology_concepts(id)
            )
        """))

        print("Created ontology tables")
    except Exception as e:
        print(f"Error creating ontology tables: {e}")

def create_new_tables(db: DBSession):
    """Create new tables for categories and codes"""
    try:
        # Create categories table
        db.execute(text("""
            CREATE TABLE IF NOT EXISTS categories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name VARCHAR(100) NOT NULL,
                description TEXT,
                user_id INTEGER,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """))

        # Create codes table
        db.execute(text("""
            CREATE TABLE IF NOT EXISTS codes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name VARCHAR(255) NOT NULL,
                category_id INTEGER,
                session_id VARCHAR(100),
                parent_id INTEGER,
                user_id INTEGER,
                embedding TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (category_id) REFERENCES categories(id),
                FOREIGN KEY (session_id) REFERENCES sessions(id),
                FOREIGN KEY (parent_id) REFERENCES codes(id),
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """))

        print("Created new tables: categories and codes")
    except Exception as e:
        print(f"Error creating new tables: {e}")

def main():
    """Run the migration"""
    print("Starting database migration...")

    # Create tables
    create_tables()

    # Get database session
    db = get_db()

    try:
        # Add role column
        print("Adding role column...")
        add_role_column(db)

        # Migrate users
        print("Migrating users...")
        migrate_users(db)

        # Migrate sessions
        print("Migrating sessions...")
        migrate_sessions(db)

        # Add document columns
        print("Adding document columns...")
        add_document_columns(db)

        # Migrate documents
        print("Migrating documents...")
        migrate_documents(db)

        # Create new tables
        print("Creating new tables...")
        create_new_tables(db)

        # Add embedding column
        print("Adding embedding column...")
        add_embedding_column(db)

        # Create ontology tables
        print("Creating ontology tables...")
        create_ontology_tables(db)

        # Add ontology columns
        print("Adding ontology columns...")
        add_ontology_columns(db)

        # Add session name column
        print("Adding session name column...")
        add_session_name_column(db)

        # Add search indexes
        print("Adding search indexes...")
        add_search_indexes(db)

        print("Migration completed successfully!")

    except Exception as e:
        print(f"Migration failed: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    main()