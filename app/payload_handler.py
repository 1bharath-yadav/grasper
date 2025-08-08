from app.agent.executor import create_temp_dir, save_multiple_uploaded_files
from app.models.schemas import AnalysisRequest
from app.agent.data_agent import handle_data
from app.agent.analyzer_agent import analyze_temp_directory
from app.agent.answer_agent import answer_questions
import logfire

# Configure Logfire
logfire.configure()


async def handle_attachments(data_analyst_input: str, uploaded_files: list):
    tmp_dir = await create_temp_dir(data_analyst_input)
    saved_files = await save_multiple_uploaded_files(uploaded_files, tmp_dir)
    return tmp_dir, saved_files


async def handle_payload(payload: dict):
    data_analyst_input = payload["data_analyst_input"]
    attachments = payload.get("attachments", [])

    with logfire.span("handle_payload") as span:
        span.set_attribute("input_length", len(data_analyst_input))
        span.set_attribute("attachment_count", len(attachments))

        logfire.info("Processing payload",
                     input_preview=data_analyst_input[:100] + "..." if len(
                         data_analyst_input) > 100 else data_analyst_input,
                     attachment_count=len(attachments))

        tmp_dir = None
        if attachments:
            tmp_dir, saved_files = await handle_attachments(data_analyst_input, attachments)
            saved_file_paths = [str(f) for f in saved_files]
            logfire.info("Files saved", temp_dir=str(tmp_dir),
                         file_count=len(saved_file_paths))
        else:
            saved_file_paths = []
            logfire.info("No attachments provided")

        # ✅ Call the data agent to handle data sources and download files
        with logfire.span("data_processing"):
            data_result = await handle_data(data_analyst_input, saved_file_paths)
            logfire.info("Data processing completed",
                         sources=data_result.get("total_sources", 0),
                         successful=data_result.get("successful_sources", 0))

        # Get the temp directory from data agent result or use the one from attachments
        if tmp_dir is None:
            tmp_dir = data_result.get("temp_directory")
            if tmp_dir is None:
                tmp_dir = await create_temp_dir(data_analyst_input)

        # ✅ Analyze all files in the temp directory
        with logfire.span("file_analysis"):
            logfire.info("Starting file analysis", temp_dir=str(tmp_dir))
            analysis_result = await analyze_temp_directory(str(tmp_dir))
            logfire.info("File analysis completed",
                         total_files=analysis_result.get('total_files', 0),
                         cache_hits=analysis_result.get('cache_size', 0))

        # ✅ Generate answers using the answer agent
        with logfire.span("answer_generation"):
            logfire.info("Generating answers to questions")
            answer_result = await answer_questions(
                data_analyst_input,
                str(tmp_dir),
                analysis_result
            )
            logfire.info("Answer generation completed",
                         status=answer_result.get("status"),
                         execution_time=answer_result.get("execution_time", 0))

        final_result = {
            "status": "success",
            "message": "Complete analysis pipeline executed successfully",
            "temp_directory": str(tmp_dir),
            "data_processing": data_result,
            "file_analysis": analysis_result,
            "answers": answer_result
        }

        logfire.info("Pipeline completed successfully",
                     total_files=analysis_result.get('total_files', 0),
                     answer_status=answer_result.get("status"))

        return final_result
