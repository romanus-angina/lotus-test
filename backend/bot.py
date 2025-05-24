import datetime
import io
import asyncio
import os
import sys
import wave
import aiofiles
from typing import Dict, Any, Optional
from dotenv import load_dotenv
from fastapi import WebSocket
from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.audio.audio_buffer_processor import AudioBufferProcessor
from pipecat.serializers.twilio import TwilioFrameSerializer
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.network.fastapi_websocket import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)
from db_service import MongoDBService
import call_state
from openai import AsyncOpenAI

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG", format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}", colorize=True)

db_service = MongoDBService()
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

async def save_audio(server_name: str, audio: bytes, sample_rate: int, num_channels: int, stream_sid: str):
    from_number, to_number = call_state.get_phone_numbers(stream_sid)
    if len(audio) > 0:
        try:
            # Get participant info from call state
            call_data = call_state.get_call_data(stream_sid)
            participant_name = call_data.get("participant_name", "Unknown") if call_data else "Unknown"
            
            metadata = {
                "server_name": server_name,
                "sample_rate": sample_rate,
                "num_channels": num_channels,
                "participant_name": participant_name,
                "audio_length": len(audio),
                "processing_timestamp": datetime.datetime.utcnow().isoformat()
            }
            
            await db_service.save_call(
                stream_sid=stream_sid, 
                from_number=from_number, 
                to_number=to_number, 
                audio_data=audio, 
                metadata=metadata
            )
            logger.info(f"saved audio to database with stream_sid: {stream_sid}")
        except Exception as e:
            logger.error(f"failed to save audio to database: {e}")
    else:
        logger.warning("no audio data to save!")

async def generate_clinical_summary(transcript: str) -> str:
    """Generate clinical summary using direct OpenAI call"""
    try:
        response = await openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": """You are generating a clinical summary for a diabetes study check-in call.
                    
                    Based on the conversation, create a brief clinical note covering:
                    - Medication compliance status
                    - Reported symptoms or side effects  
                    - Patient's overall condition
                    - Any concerns or follow-up needed
                    - Recommended next steps
                    
                    Format as a professional clinical note, 3-4 sentences maximum.
                    Use clinical terminology where appropriate but keep it clear."""
                },
                {
                    "role": "user", 
                    "content": f"Generate clinical summary for this diabetes study check-in call:\n\n{transcript}"
                }
            ],
            max_tokens=200,
            temperature=0.3
        )
        
        summary = response.choices[0].message.content.strip()
        return summary
        
    except Exception as e:
        logger.error(f"Error generating clinical summary: {e}")
        return f"Call completed - summary generation failed: {str(e)}"

async def save_transcript(stream_sid: str, transcript: str):
    try:
        from_number, to_number = call_state.get_phone_numbers(stream_sid)
        
        # Save transcript
        await db_service.save_call(
            stream_sid=stream_sid, 
            transcript=transcript, 
            from_number=from_number, 
            to_number=to_number
        )
        logger.info(f"saved transcript to database with stream_sid: {stream_sid}")
        
        # Generate and save clinical summary
        if transcript and len(transcript.strip()) > 50:
            try:
                summary = await generate_clinical_summary(transcript)
                
                # Save the summary
                await db_service.db.calls.update_one(
                    {"stream_sid": stream_sid},
                    {"$set": {
                        "ai_summary": summary,
                        "clinical_summary": summary,
                        "call_completed": True,
                        "completion_time": datetime.datetime.utcnow(),
                        "last_updated": datetime.datetime.utcnow()
                    }}
                )
                logger.info(f"saved clinical summary for stream_sid: {stream_sid}")
                
            except Exception as summary_error:
                logger.error(f"Error generating summary: {summary_error}")
                
    except Exception as e:
        logger.error(f"failed to save transcript to database: {e}")

async def run_bot(websocket: WebSocket, stream_sid: str, testing: bool):
    # Get participant info from call state
    call_data = call_state.get_call_data(stream_sid)
    participant_name = "Unknown Participant"
    first_name = ""
    
    if call_data:
        participant_name = call_data.get("participant_name", "Unknown Participant")
        first_name = participant_name.split()[0] if participant_name and participant_name != "Unknown Participant" else ""
        logger.info(f"Starting clinical bot for {participant_name}")
    
    async def extract_transcript(context):
        """Extract transcript from the pipeline task and save it to the database."""
        if not context or not hasattr(context, "get_messages"):
            return ""
        
        messages = context.get_messages()
        transcript = ""
        
        for message in messages:
            role = message.get("role", "unknown")
            content = message.get("content", "")

            if isinstance(content, list):
                txt_content = ""
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        txt_content += item.get("text", "")
                content = txt_content
            
            if content and isinstance(content, str):
                # Use clinical-appropriate labels
                role_label = "PATIENT" if role == "user" else "LOTUS_AI"
                transcript += f"{role_label}: {content}\n\n"
        
        if transcript:
            await save_transcript(stream_sid, transcript)
                 
    transport = FastAPIWebsocketTransport(
        websocket, 
        FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=False,
            vad_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
            vad_audio_passthrough=True,
            serializer=TwilioFrameSerializer(stream_sid=stream_sid),
        )
    )

    llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o")
    
    tts = ElevenLabsTTSService(
        api_key=os.getenv("ELEVENLABS_API_KEY"),
        voice_id="kdmDKE6EkgrWrrykO9Qt",  
        push_silence_after_stopping=False
    )
    
    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"), audio_passthrough=True)

    # Clinical trial focused system message
    messages = [
        {
            "role": "system",
            "content": f"""You are Lotus, an AI assistant conducting a wellness check-in call for {participant_name} 
            in a Type 2 Diabetes clinical trial study. This is a routine follow-up to ensure their wellbeing and medication compliance.

            Your objectives:
            1. Conduct a friendly, professional wellness check
            2. Ask about medication compliance and any missed doses
            3. Inquire about symptoms, side effects, or health concerns
            4. Provide supportive, non-judgmental responses
            5. Suggest contacting the study coordinator for any medical concerns
            6. Keep the conversation focused but allow natural flow

            Be warm, empathetic, and professional. Use the participant's first name when appropriate.
            Keep responses conversational and under 150 words unless more detail is needed.
            
            Start by greeting {first_name or 'them'} warmly and explaining this is their scheduled wellness check-in from the diabetes study."""
        },
    ]

    context = OpenAILLMContext(messages)
    context_aggregator = llm.create_context_aggregator(context)

    # saves conversation in memory
    audiobuffer = AudioBufferProcessor(user_continuous_stream=not testing)

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            context_aggregator.user(),
            llm,
            tts,
            transport.output(),
            audiobuffer,
            context_aggregator.assistant(),
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            audio_in_sample_rate=8000,
            audio_out_sample_rate=8000,
            allow_interruptions=True,
        ),
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        # Start audio recording
        await audiobuffer.start_recording()
        # Start conversation with greeting
        messages.append(
            {
                "role": "system",
                "content": f"Please greet {first_name or 'the participant'} and begin the wellness check-in for type 2 diabetes clinical trial."
            }
        )
        await task.queue_frames([context_aggregator.user().get_context_frame()])
        
    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        # Extract and save transcript
        await extract_transcript(context)
        
        # Mark call as completed
        try:
            await db_service.db.calls.update_one(
                {"stream_sid": stream_sid},
                {"$set": {
                    "call_completed": True,
                    "completion_time": datetime.datetime.utcnow(),
                    "call_status": "completed"
                }}
            )
        except Exception as e:
            logger.error(f"Error updating call completion status: {e}")
        
        await task.cancel()
        call_state.remove_call(stream_sid)

    @audiobuffer.event_handler("on_audio_data")
    async def on_audio_data(buffer, audio, sample_rate, num_channels):
        logger.debug(f"ON_AUDIO_DATA TRIGGERED! Audio length: {len(audio)}, Stream SID: {stream_sid}")
        
        # Enhanced server name with participant context
        participant_info = call_state.get_call_data(stream_sid)
        participant_name = participant_info.get("participant_name", "Unknown") if participant_info else "Unknown"
        server_name = f"{participant_name.replace(' ', '_')}_{stream_sid}"
        
        try:
            await save_audio(server_name, audio, sample_rate, num_channels, stream_sid)
        except Exception as e:
            logger.error(f"ERROR in on_audio_data: {str(e)}")
    
    runner = PipelineRunner(handle_sigint=False, force_gc=True)
    await runner.run(task)