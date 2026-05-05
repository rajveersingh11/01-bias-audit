from src import get_logger
logger = get_logger(__name__)
from src.pipeline.stage_01_data_ingestion import DataIngestionPipeline


STAGE_NAME = "Data Ingestion"
try:
    logger.info(f">>>>>> Stage: {STAGE_NAME} started <<<<<<")
    pipeline = DataIngestionPipeline()
    artifact = pipeline.run()
    logger.info(f">>>>>> Stage: {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(f"Error in stage {STAGE_NAME}: {e}")
    raise e


STAGE_NAME = "Prepare Base Model"
try:
    logger.info(f">>>>>> Stage: {STAGE_NAME} started <<<<<<")
    from src.pipeline.stage_02_prepare_base_model import PrepareBaseModelPipeline
    pipeline = PrepareBaseModelPipeline()
    artifact = pipeline.run()
    logger.info(f">>>>>> Stage: {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(f"Error in stage {STAGE_NAME}: {e}")
    raise e