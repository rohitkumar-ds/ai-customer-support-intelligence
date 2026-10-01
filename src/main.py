import argparse
import sys
from pathlib import Path
from typing import Dict, Any

from src.config import get_settings
from src.database.connection import get_engine, get_db, init_db
from src.database.models import Base, Ticket
from src.database.repository import TicketRepository
from src.utils.logger import setup_logging, get_logger

logger = get_logger(__name__)


def run_system_diagnostics() -> Dict[str, Any]:
    """Runs a complete diagnostic check across configuration, directories, and database.

    Returns:
        A dictionary containing diagnostic status for each core subsystem.
    """
    logger.info("Initiating platform diagnostic self-test...")
    settings = get_settings()

    status: Dict[str, Any] = {
        "platform_name": settings.app.name,
        "version": settings.app.version,
        "environment": settings.app.environment,
        "directories_ready": False,
        "database_connected": False,
        "tables_initialized": False,
        "intents_configured": len(settings.models.intent_classes),
    }

    # 1. Directory verification
    try:
        settings.ensure_directories()
        status["directories_ready"] = True
        logger.info("All application directories verified and ready.")
    except Exception as e:
        logger.error(f"Directory initialization failed: {e}")
        status["directories_error"] = str(e)

    # 2. Database connectivity and table provisioning
    try:
        init_db()
        status["database_connected"] = True
        status["tables_initialized"] = True
        logger.info("Database connection active and schemas verified.")
    except Exception as e:
        logger.error(f"Database setup check failed: {e}")
        status["database_error"] = str(e)

    # 3. Database CRUD round-trip sanity check
    try:
        with get_db() as session:
            repo = TicketRepository(session)
            analytics = repo.get_analytics_summary()
            status["total_persisted_tickets"] = analytics["total_tickets"]
            logger.info(
                f"Persistence verified. Current stored tickets: {analytics['total_tickets']}"
            )
    except Exception as e:
        logger.error(f"CRUD sanity check failed: {e}")
        status["crud_error"] = str(e)

    return status


def print_diagnostic_banner(diagnostics: Dict[str, Any]) -> None:
    """Renders a readable summary banner in the terminal."""
    print("=" * 70)
    print(f" {diagnostics.get('platform_name')} (v{diagnostics.get('version')})")
    print("=" * 70)
    print(f" Environment        : {diagnostics.get('environment')}")
    print(f" Intent Classes     : {diagnostics.get('intents_configured')} configured")
    print(f" Directories Ready  : {'[YES]' if diagnostics.get('directories_ready') else '[FAIL]'}")
    print(f" Database Connected : {'[YES]' if diagnostics.get('database_connected') else '[FAIL]'}")
    print(f" Tables Provisioned : {'[YES]' if diagnostics.get('tables_initialized') else '[FAIL]'}")
    print(f" Current Tickets    : {diagnostics.get('total_persisted_tickets', 0)}")
    print("=" * 70)
    
    if diagnostics.get("database_connected") and diagnostics.get("directories_ready"):
        print(" [SUCCESS] Phase 1 Foundation is healthy and ready for ML/NLP components.")
    else:
        print(" [WARNING] Some diagnostics reported issues. Check logs/app.log.")
    print("=" * 70)


def main() -> None:
    """Main CLI entrypoint for Phase 1 verification."""
    parser = argparse.ArgumentParser(
        description="AI Customer Support Intelligence Platform Manager"
    )
    parser.add_argument(
        "--init-db", action="store_true", help="Initialize and provision database tables"
    )
    parser.add_argument(
        "--status", action="store_true", help="Run diagnostics and print system status"
    )

    args = parser.parse_args()

    # Initialize structured logging first
    setup_logging()

    if args.init_db:
        logger.info("Explicit database initialization requested via CLI.")
        init_db()
        print("Database initialized successfully.")
        return

    # Default action: run full diagnostics
    diagnostics = run_system_diagnostics()
    print_diagnostic_banner(diagnostics)


if __name__ == "__main__":
    main()
