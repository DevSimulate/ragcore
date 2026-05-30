"""initial schema

Revision ID: 0001
Revises:
Create Date: 2026-05-30

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "documents",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("filename", sa.String(512), nullable=False),
        sa.Column("tenant_id", sa.String(64), nullable=False),
        sa.Column("size_bytes", sa.Integer(), nullable=True),
        sa.Column("chunk_count", sa.Integer(), nullable=True),
        sa.Column("status", sa.String(32), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("now()"), nullable=True),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.text("now()"), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_documents_tenant_id", "documents", ["tenant_id"])

    op.create_table(
        "query_logs",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("tenant_id", sa.String(64), nullable=False),
        sa.Column("session_id", sa.String(64), nullable=False),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("answer", sa.Text(), nullable=False),
        sa.Column("source_count", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_query_logs_tenant_id", "query_logs", ["tenant_id"])
    op.create_index("ix_query_logs_session_id", "query_logs", ["session_id"])


def downgrade() -> None:
    op.drop_table("query_logs")
    op.drop_table("documents")
