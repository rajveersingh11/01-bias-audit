from src.components.data_ingestion import DataIngestion, DataIngestionArtifact
from src.constants import STAGE_01
from src import get_logger

logger = get_logger(__name__)


class DataIngestionPipeline:
    def __init__(self):
        self.component = DataIngestion()

    def run(self) -> DataIngestionArtifact:
        logger.info(f">>>>>> {STAGE_01} started <<<<<<")
        try:
            artifact = self.component.run()
            logger.info(f">>>>>> {STAGE_01} completed <<<<<<\n\nx==========x")
            return artifact
        except FileNotFoundError as e:
            logger.error(f"{STAGE_01} — file not found: {e}")
            raise
        except ValueError as e:
            logger.error(f"{STAGE_01} — data validation failed: {e}")
            raise
        except Exception as e:
            logger.exception(f"{STAGE_01} — unexpected error: {e}")
            raise


if __name__ == "__main__":
    pipeline = DataIngestionPipeline()
    artifact = pipeline.run()
    print(f"Raw data path       : {artifact.raw_data_path}")
    print(f"Processed data path : {artifact.processed_data_path}")
    print(f"Rows                : {artifact.row_count:,}")
    print(f"Columns             : {artifact.column_count}")